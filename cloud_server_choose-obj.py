# cloud_server.py
# 运行在你的云服务器上

from estimater import *
from datareader import *
import argparse
from flask import Flask, request, jsonify
import numpy as np
import cv2
import lz4.frame
# import base64 # 不再需要回传图片，因此不需要 base64
from ultralytics import YOLO
import time
import os

# --- 全局变量 ---
app = Flask(__name__)
g_K = None
g_shape = None
g_debug_dir = None
g_est_refine_iter = 2 # 🚀 优化1：默认迭代次数降低为 2，显著提速

# 💡 核心配置区
OBJECT_CONFIG = {
    "passive": {
        "yolo_path": "passive.pt",
        "mesh_path": "./demo_data/mydata/mesh/passive.obj", 
    },
    "insert": {
        "yolo_path": "insert.pt",
        "mesh_path": "./demo_data/mydata/mesh/insert.obj", 
    }
}

LOADED_OBJECTS = {} 

# --- 辅助函数：计算包围盒的8个角点 ---
def get_projected_corners(pose, bbox, K):
    """
    计算 3D 包围盒的 8 个顶点在 2D 图像上的投影坐标
    """
    # bbox shape: (2, 3) -> min_xyz, max_xyz
    min_pt = bbox[0]
    max_pt = bbox[1]
    
    # 构建 8 个角点 (3D)
    corners_3d = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]]
    ]) # shape (8, 3)

    # 1. 应用位姿变换 (Model -> Camera)
    # pose is 4x4, corners_3d is 8x3. Need to append 1 for homogeneous
    ones = np.ones((8, 1))
    corners_hom = np.hstack((corners_3d, ones)) # 8x4
    corners_cam = (pose @ corners_hom.T).T # 8x4
    corners_cam = corners_cam[:, :3] # 8x3 (xyz in cam)

    # 2. 投影到 2D (Camera -> Pixel)
    # project: u = fx * x/z + cx, v = fy * y/z + cy
    projected = (K @ corners_cam.T).T # 8x3
    z = projected[:, 2:3] + 1e-5 # 避免除零
    pixels = projected[:, :2] / z
    
    return pixels.astype(int).tolist() # 返回整数列表 [[u,v], ...]

# -----------------

def setup_server():
    global g_K, g_debug_dir, g_est_refine_iter, LOADED_OBJECTS

    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    # 🚀 优化：默认设为 2
    parser.add_argument('--est_refine_iter', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    
    g_debug_dir = args.debug_dir
    g_est_refine_iter = args.est_refine_iter
    os.system(f'rm -rf {g_debug_dir}/*') # 清理日志加快IO

    logging.info(">>> Loading models...")

    for obj_name, config in OBJECT_CONFIG.items():
        logging.info(f"--- Loading: [{obj_name}] ---")
        
        mesh_file = config["mesh_path"]
        if not os.path.exists(mesh_file): continue
            
        mesh_or_scene = trimesh.load(mesh_file)
        if isinstance(mesh_or_scene, trimesh.Scene):
            mesh = mesh_or_scene.dump(concatenate=True)
        else:
            mesh = mesh_or_scene

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        # debug=0 关闭调试输出以提速
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                             mesh=mesh, scorer=scorer, refiner=refiner, 
                             debug_dir=g_debug_dir, debug=0, glctx=glctx)
        
        yolo_model = YOLO(config["yolo_path"])

        LOADED_OBJECTS[obj_name] = {
            "est": est,
            "yolo": yolo_model,
            "to_origin": to_origin,
            "bbox": bbox
        }
    
    logging.info("✅ Server Ready (Optimized Mode)")

@app.route('/init', methods=['POST'])
def init():
    global g_K, g_shape
    try:
        data = request.json
        g_K = np.array(data['K'])
        g_shape = tuple(data['shape']) 
        return jsonify({"status": "initialized"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/process', methods=['POST'])
def process_frame():
    global g_K, g_shape, g_est_refine_iter, LOADED_OBJECTS
    
    if g_K is None: return jsonify({"error": "Not initialized"}), 503

    try:
        target_type = request.form.get('type', 'passive')
        if target_type not in LOADED_OBJECTS:
            return jsonify({"error": f"Unknown object {target_type}"}), 400
            
        obj_data = LOADED_OBJECTS[target_type]
        est = obj_data["est"]
        yolo = obj_data["yolo"]
        to_origin = obj_data["to_origin"]
        bbox = obj_data["bbox"] # 3D bounding box

        # 1. 解码
        rgb_bytes = request.files['rgb'].read()
        img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
        color = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth_bytes = lz4.frame.decompress(request.files['depth'].read())
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(g_shape)

        # 2. YOLO (减少 verbose 输出)
        # 🚀 优化：conf 设为 0.7 保证检出率，减少漏检导致的重试
        results = yolo(color, conf=0.7, verbose=False)
        
        mask = np.zeros(g_shape, dtype=bool)
        if results[0].masks and len(results[0].masks.data) > 0:
            m_data = results[0].masks.data[0].cpu().numpy()
            if m_data.shape[:2] != color.shape[:2]:
                m_data = cv2.resize(m_data, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = m_data.astype(bool)
        
        if mask.sum() < 50:
            return jsonify({"found": False}), 200

        # 3. Pose Estimation
        pose = est.register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=g_est_refine_iter)
        
        # 4. 🚀 优化：不画图，只计算坐标
        # 计算物体中心位姿
        center_pose = pose @ np.linalg.inv(to_origin)
        
        # 计算8个角点在屏幕上的坐标
        corners_2d = get_projected_corners(center_pose, bbox, g_K)

        return jsonify({
            "found": True,
            "pose": pose.tolist(),
            "corners": corners_2d # 返回 8 个点 [[u,v], ...]
        })

    except Exception as e:
        logging.error(f"Err: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    setup_server()
    app.run(host='127.0.0.1', port=6006, threaded=True) # threaded=True 允许并发
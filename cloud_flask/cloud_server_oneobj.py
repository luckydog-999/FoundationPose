# cloud_server.py
# 运行在你的云服务器上

from estimater import *
from datareader import *
import argparse

# --- 新增的库 ---
from flask import Flask, request, jsonify
import numpy as np
import cv2
import lz4.frame
import base64
from ultralytics import YOLO
import time
# -----------------

# --- 全局变量 ---
app = Flask(__name__)
est = None
yolo_model = None
g_K = None           # 'g_' a prefix for 'global'
g_shape = None
g_to_origin = None
g_bbox = None
g_debug_dir = None
g_est_refine_iter = None
# -----------------

def setup_server():
    """
    加载所有模型和配置，这只在服务器启动时运行一次。
    """
    global est, yolo_model, g_K, g_to_origin, g_bbox, g_debug_dir, g_est_refine_iter

    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mydata/mesh/textured_simple.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--debug', type=int, default=1) # Debug 在这里作用不大，但保留
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    
    g_debug_dir = args.debug_dir
    g_est_refine_iter = args.est_refine_iter
    os.system(f'rm -rf {g_debug_dir}/* && mkdir -p {g_debug_dir}/track_vis {g_debug_dir}/ob_in_cam')

    # 1. 加载 FoundationPose 模型
    logging.info("Loading mesh...")
    mesh_or_scene = trimesh.load(args.mesh_file)
    if isinstance(mesh_or_scene, trimesh.Scene):
      mesh = mesh_or_scene.dump(concatenate=True)
    else:
      mesh = mesh_or_scene

    g_to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    g_bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    logging.info("Initializing FoundationPose...")
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=g_debug_dir, debug=args.debug, glctx=glctx)
    logging.info("FoundationPose initialization done")

    # 2. 加载 YOLO 模型
    logging.info("Loading YOLO model...")
    try:
      yolo_model = YOLO('best.pt')
      logging.info("YOLO model loaded successfully.")
    except Exception as e:
      logging.error(f"Failed to load YOLO model 'best.pt'. Error: {e}")
      exit()
      
    logging.info("✅ Server is ready and waiting for initialization data...")

@app.route('/init', methods=['POST'])
def init():
    """
    接收来自客户端的相机内参 K 和图像尺寸。
    """
    global g_K, g_shape
    try:
        data = request.json
        g_K = np.array(data['K'])
        g_shape = tuple(data['shape']) # (height, width)
        logging.info(f"Client initialized. K matrix:\n{g_K}\nShape: {g_shape}")
        return jsonify({"status": "initialized"})
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/process', methods=['POST'])
def process_frame():
    """
    处理来自客户端的单帧数据。
    """
    global g_K, g_shape, est, yolo_model, g_to_origin, g_bbox, g_debug_dir, g_est_refine_iter
    
    if g_K is None or g_shape is None:
        logging.warning("Server not initialized. Client must call /init first.")
        return jsonify({"error": "Server not initialized"}), 503 # Service Unavailable

    try:
        # 1. 解码 RGB 图像 (JPEG)
        # request.files['rgb'] 是一个 FileStorage 对象, .read() 获取 bytes
        rgb_jpg_bytes = request.files['rgb'].read()
        nparr = np.frombuffer(rgb_jpg_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        color = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # FoundationPose 需要 RGB

        # 2. 解码深度图 (lz4 压缩的 float32)
        depth_lz4_bytes = request.files['depth'].read()
        depth_bytes = lz4.frame.decompress(depth_lz4_bytes)
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(g_shape)

        # 3. 运行 YOLO 获取掩码
        # 💡 💡 💡 关键修改在这里 💡 💡 💡
        # 添加 conf=0.85 来过滤掉置信度低于 0.85 的检测
        # 添加 verbose=False 来减少不必要的日志刷屏
        yolo_results = yolo_model(color, conf=0.85, verbose=False)
        
        mask = np.zeros(g_shape, dtype=bool)

        # 检查过滤后是否还有掩码
        if yolo_results[0].masks is not None and len(yolo_results[0].masks.data) > 0:
            # yolo_results 已经被 conf=0.85 过滤
            # 并且结果按置信度排序，所以 data[0] 是置信度最高的那个
            mask_data = yolo_results[0].masks.data[0].cpu().numpy()
            if mask_data.shape[0] != color.shape[0] or mask_data.shape[1] != color.shape[1]:
                mask_data = cv2.resize(mask_data, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = mask_data.astype(bool)
        
        if mask.sum() < 100:
            logging.warning("No valid mask detected (conf < 0.85 or mask too small).")
            return jsonify({"error": "No valid mask detected"}), 200 # 200 OK 但带 error

        # 4. 运行 FoundationPose 位姿估计
        logging.info("Registering pose using YOLO mask...")
        pose = est.register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=g_est_refine_iter)
        
        # 5. (可选) 生成可视化图像并发送回客户端
        center_pose = pose @ np.linalg.inv(g_to_origin)
        vis = draw_posed_3d_box(g_K, img=color, ob_in_cam=center_pose, bbox=g_bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=g_K, thickness=3, transparency=0, is_input_rgb=True)
        
        # 将可视化图像编码为 JPEG
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        _, vis_jpg_bytes = cv2.imencode('.jpg', vis_bgr)
        
        # 使用 base64 编码以便放入 JSON
        vis_base64 = base64.b64encode(vis_jpg_bytes).decode('utf-8')

        # 6. 返回结果
        return jsonify({
            "pose": pose.tolist(),
            "vis_image": vis_base64 # 发送可视化图像
        })

    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    setup_server()
    # 必须用 '0.0.0.0' 才能从外部访问
    # 💡 新的代码
    # ******************************************
    app.run(host='127.0.0.1', port=6006)
    # ****************************************
    
# python cloud_server.py --mesh_file ./demo_data/mydata/mesh/textured_simple.obj
# 第二个速度更快
# python cloud_server.py --mesh_file ./demo_data/mydata/mesh/textured_simple.obj --est_refine_iter 2
# docker_server.py (运行在容器内)
# 职责：接收 RGB + Depth + Mask -> 计算位姿

from estimater import *
from datareader import *
import argparse
from flask import Flask, request, jsonify
import numpy as np
import cv2
import lz4.frame
import logging

app = Flask(__name__)
est = None
g_K = None
g_est_refine_iter = None
g_debug_dir = None

def setup_server():
    global est, g_K, g_est_refine_iter, g_debug_dir

    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/data/mesh/myobj.obj')
    parser.add_argument('--est_refine_iter', type=int, default=2) # 速度优先
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    g_est_refine_iter = args.est_refine_iter
    g_debug_dir = args.debug_dir

    # 加载 Mesh 和 FoundationPose
    logging.info("Loading FoundationPose...")
    mesh = trimesh.load(args.mesh_file)
    if isinstance(mesh, trimesh.Scene): mesh = mesh.dump(concatenate=True)
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=g_debug_dir, debug=0, glctx=glctx)
    logging.info("✅ Docker Server Ready (FoundationPose only)")

@app.route('/process', methods=['POST'])
def process_frame():
    global est, g_K
    try:
        # 1. 接收并解析参数
        K_list = request.form.get('K') # 从 form data 获取 K
        if K_list:
            # K 以字符串形式传来 "fx,0,cx,0,fy,cy,0,0,1"
            g_K = np.fromstring(K_list, sep=',').reshape(3,3)
            
        if g_K is None:
             return jsonify({"error": "K matrix missing"}), 400

        # 2. 解码 RGB
        rgb_bytes = request.files['rgb'].read()
        color = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        # 3. 解码 Depth (lz4)
        depth_lz4 = request.files['depth'].read()
        depth_bytes = lz4.frame.decompress(depth_lz4)
        h, w = color.shape[:2]
        depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(h, w)

        # 4. 解码 Mask (关键修改：接收外部传来的 Mask)
        mask_bytes = request.files['mask'].read()
        mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = mask_img.astype(bool)

        if mask.sum() < 50:
            return jsonify({"error": "Mask too small"}), 200

        # 5. 运行 Pose 估计
        pose = est.register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=g_est_refine_iter)
        
        # 6. 返回结果 (不需要回传图片，本地自己有)
        return jsonify({"pose": pose.tolist()})

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    setup_server()
    app.run(host='0.0.0.0', port=6006)
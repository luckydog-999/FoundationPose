# docker_server.py
# 运行环境：Docker 容器内部 (Linux)
# 作用：调用 GPU (RTX 4050) 进行计算

import socket
import struct
import json
import numpy as np
import cv2
import lz4.frame
import time
import os
import logging
import torch
import gc

# 引入你的库 (假设目录结构已挂载到容器内)
from ultralytics import YOLO
from estimater import * 
from datareader import *
from my_utils.socket_utils import recv_msg, send_msg, recv_json, send_json, recvall

# --- 显存优化配置 ---
torch.set_grad_enabled(False) # 🈲 全局禁用梯度，大幅节省显存

# --- 配置 ---
HOST = '0.0.0.0' # ⚠️ 必须是 0.0.0.0，否则宿主机无法访问 Docker 端口
PORT = 6006

# 根据你的挂载路径修改这里
# 建议在 docker run 时通过 -v 挂载本地代码目录到 /app
OBJECT_CONFIG = {
    "passive": {
        "yolo_path": "passive_best.pt", 
        "mesh_path": "./demo_data/passive/mesh/passive.obj", 
    },
    "insert": {
        "yolo_path": "insert_best.pt",
        "mesh_path": "./demo_data/insert/mesh/insert.obj", 
    }
}

LOADED_OBJECTS = {} 
g_K = None
g_shape = None

# ⚠️ 针对 4050 6GB 显存的优化
# 迭代次数越多越准，但显存和耗时越高。建议先设为 1，稳定后再尝试 2。
g_est_refine_iter = 1 

def get_projected_corners(pose, bbox, K):
    min_pt = bbox[0]
    max_pt = bbox[1]
    corners_3d = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]]
    ])
    ones = np.ones((8, 1))
    corners_hom = np.hstack((corners_3d, ones))
    corners_cam = (pose @ corners_hom.T).T
    corners_cam = corners_cam[:, :3]
    projected = (K @ corners_cam.T).T
    z = projected[:, 2:3] + 1e-5
    pixels = projected[:, :2] / z
    return pixels.astype(int).tolist()

def load_models():
    logging.info(">>> Loading models...")
    
    # 检查显存
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        logging.info(f"💾 GPU Available: {torch.cuda.get_device_name(0)}")
        logging.info(f"💾 Free VRAM: {free_mem:.2f} GB")
        if free_mem < 4.0:
            logging.warning("⚠️ Warning: Low VRAM detected (<4GB). Ensure no other GPU apps are running.")

    for obj_name, config in OBJECT_CONFIG.items():
        logging.info(f"--- Loading: [{obj_name}] ---")
        mesh_file = config["mesh_path"]
        yolo_file = config["yolo_path"]
        
        if not os.path.exists(mesh_file): 
            logging.error(f"❌ Mesh missing: {mesh_file}")
            continue
        if not os.path.exists(yolo_file):
            logging.error(f"❌ YOLO missing: {yolo_file}")
            continue

        mesh = trimesh.load(mesh_file)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                             mesh=mesh, scorer=scorer, refiner=refiner, 
                             debug_dir="./debug", debug=0, glctx=glctx)
        
        yolo_model = YOLO(yolo_file)

        LOADED_OBJECTS[obj_name] = {
            "est": est, "yolo": yolo_model, "to_origin": to_origin, "bbox": bbox
        }
        
        # 加载完一个模型后清理一下缓存
        torch.cuda.empty_cache()
        
    logging.info("✅ Models Ready")

def main():
    global g_K, g_shape
    
    # 简单的日志设置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    set_seed(0)
    
    load_models()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    
    try:
        server_sock.bind((HOST, PORT))
    except Exception as e:
        print(f"Bind Error: {e}")
        return

    server_sock.listen(1)
    print(f"🚀 Docker Server listening on {PORT}...")

    while True:
        conn, addr = server_sock.accept()
        print(f"Connected by {addr} (Host Machine)")
        
        try:
            # 1. Init Handshake
            init_data = recv_json(conn)
            if init_data and 'K' in init_data:
                g_K = np.array(init_data['K'])
                g_shape = tuple(init_data['shape'])
                print("Client Initialized.")
                send_json(conn, {"status": "ok"})
            else:
                print("Init failed.")
                conn.close()
                continue

            # 2. Process Loop
            while True:
                header_data = recvall(conn, 12)
                if not header_data: break
                
                rgb_len, depth_len, type_len = struct.unpack('>III', header_data)
                
                type_bytes = recvall(conn, type_len)
                rgb_bytes = recvall(conn, rgb_len)
                depth_bytes = recvall(conn, depth_len)
                
                if not rgb_bytes or not depth_bytes: break

                target_type = type_bytes.decode('utf-8')
                
                if target_type not in LOADED_OBJECTS:
                    send_json(conn, {"found": False, "err": "Unknown object"})
                    continue

                obj_data = LOADED_OBJECTS[target_type]
                
                # Decode
                img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
                color = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                depth_raw = lz4.frame.decompress(depth_bytes)
                depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(g_shape)

                # YOLO Inference
                # 显存优化：verbose=False 减少打印
                results = obj_data["yolo"](color, conf=0.5, verbose=False)
                
                mask = np.zeros(g_shape, dtype=bool)
                
                if len(results[0].boxes) > 0:
                    if results[0].masks is not None:
                        m_data = results[0].masks.data[0].cpu().numpy()
                        # Resize if necessary
                        if m_data.shape[:2] != color.shape[:2]:
                            m_data = cv2.resize(m_data, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask = m_data.astype(bool)
                
                if mask.sum() < 50:
                    send_json(conn, {"found": False})
                    continue

                # Pose Estimation
                try:
                    # ⚠️ 显存核心区
                    pose = obj_data["est"].register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=g_est_refine_iter)
                    
                    center_pose = pose @ np.linalg.inv(obj_data["to_origin"])
                    corners_2d = get_projected_corners(center_pose, obj_data["bbox"], g_K)
                    
                    send_json(conn, {
                        "found": True,
                        "pose": pose.tolist(),
                        "corners": corners_2d
                    })
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("❌ GPU OOM! Trying to clear cache...")
                        torch.cuda.empty_cache()
                        send_json(conn, {"found": False, "err": "OOM"})
                    else:
                        raise e

        except Exception as e:
            print(f"Connection Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
            print("Connection closed. Waiting...")

if __name__ == '__main__':
    main()
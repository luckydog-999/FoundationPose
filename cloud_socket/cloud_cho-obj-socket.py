# cloud_server_socket.py
# 运行在你的云服务器上

import socket
import struct
import json
import numpy as np
import cv2
import lz4.frame
import time
import os
import argparse
import logging
from ultralytics import YOLO
# 假设你的 FoundationPose 依赖都在这里
from estimater import * 
from datareader import *

# 引入刚才写的辅助工具
from my_utils.socket_utils import recv_msg, send_msg, recv_json, send_json, recvall

# --- 配置 ---
HOST = '0.0.0.0'
PORT = 6006
OBJECT_CONFIG = {
    "passive": {
        # ⚠️ 请确保服务器上真的有这个文件名，如果叫 passive.pt 请修改这里
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
g_est_refine_iter = 2

# --- 辅助函数 ---
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

def load_models(debug_dir):
    logging.info(">>> Loading models...")
    for obj_name, config in OBJECT_CONFIG.items():
        logging.info(f"--- Loading: [{obj_name}] ---")
        mesh_file = config["mesh_path"]
        yolo_file = config["yolo_path"]
        
        if not os.path.exists(mesh_file): 
            logging.error(f"❌ Mesh missing: {mesh_file}")
            continue
        if not os.path.exists(yolo_file):
            logging.error(f"❌ YOLO missing: {yolo_file}")
            continue # 找不到模型就跳过，防止报错
            
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
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                             mesh=mesh, scorer=scorer, refiner=refiner, 
                             debug_dir=debug_dir, debug=0, glctx=glctx)
        
        yolo_model = YOLO(yolo_file)

        LOADED_OBJECTS[obj_name] = {
            "est": est, "yolo": yolo_model, "to_origin": to_origin, "bbox": bbox
        }
    logging.info("✅ Models Ready")

def main():
    global g_K, g_shape, g_est_refine_iter
    
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--est_refine_iter', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()
    g_est_refine_iter = args.est_refine_iter
    set_logging_format()
    set_seed(0)
    
    load_models(args.debug_dir)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    
    try:
        server_sock.bind((HOST, PORT))
    except Exception as e:
        print(f"Bind Error: {e}")
        return

    server_sock.listen(1)
    print(f"🚀 Socket Server listening on {PORT}...")

    while True:
        conn, addr = server_sock.accept()
        print(f"Connected by {addr}")
        
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

                # --- 🟢 YOLO 改进版 ---
                # 1. 降低阈值到 0.5 (适应你的 0.58 盖子)
                results = obj_data["yolo"](color, conf=0.5, verbose=False)
                
                mask = np.zeros(g_shape, dtype=bool)
                
                # 2. 增加调试打印，让你知道发生了什么
                if len(results[0].boxes) > 0:
                    max_conf = results[0].boxes.conf[0].item()
                    print(f"🔍 [YOLO] Found {len(results[0].boxes)} objs. Max Conf: {max_conf:.2f}")
                    
                    if results[0].masks is not None:
                        # 3. 智能获取最高置信度的 Mask (假设 YOLO 已经按 conf 排序，通常是 index 0)
                        # 如果你有多个物体，这里默认取第一个最可信的
                        m_data = results[0].masks.data[0].cpu().numpy()
                        if m_data.shape[:2] != color.shape[:2]:
                            m_data = cv2.resize(m_data, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask = m_data.astype(bool)
                    else:
                        print("⚠️ [YOLO] Boxes found but NO MASKS! Check model type.")
                else:
                    print("❌ [YOLO] Nothing found (Conf < 0.5)")

                # 4. Mask 检查
                if mask.sum() < 50:
                    send_json(conn, {"found": False})
                    continue

                # Pose Estimation
                pose = obj_data["est"].register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=g_est_refine_iter)
                
                center_pose = pose @ np.linalg.inv(obj_data["to_origin"])
                corners_2d = get_projected_corners(center_pose, obj_data["bbox"], g_K)
                
                send_json(conn, {
                    "found": True,
                    "pose": pose.tolist(),
                    "corners": corners_2d
                })

        except Exception as e:
            print(f"Connection Error: {e}")
            import traceback
            traceback.print_exc() # 打印详细报错
        finally:
            conn.close()
            print("Connection closed. Waiting for new client...")

if __name__ == '__main__':
    main()
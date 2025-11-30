# server_optimized.py
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
from estimater import * 
from datareader import *
from my_utils.socket_utils import recv_msg, send_msg, recv_json, send_json, recvall

HOST = '0.0.0.0'
PORT = 6006
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
g_est_refine_iter = 2

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
        if not os.path.exists(config["mesh_path"]) or not os.path.exists(config["yolo_path"]):
            continue
            
        mesh = trimesh.load(config["mesh_path"])
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                             mesh=mesh, scorer=scorer, refiner=refiner, 
                             debug_dir=debug_dir, debug=0, glctx=glctx)
        
        yolo_model = YOLO(config["yolo_path"])
        LOADED_OBJECTS[obj_name] = {
            "est": est, "yolo": yolo_model, "to_origin": to_origin, "bbox": bbox
        }
    logging.info("✅ Models Ready")

def main():
    global g_K, g_shape, g_est_refine_iter
    
    # 保持你原来的参数
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
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print(f"🚀 Socket Server listening on {PORT}...")

    while True:
        conn, addr = server_sock.accept()
        print(f"Connected by {addr}")
        
        try:
            init_data = recv_json(conn)
            if init_data and 'K' in init_data:
                g_K = np.array(init_data['K'])
                g_shape = tuple(init_data['shape'])
                send_json(conn, {"status": "ok"})
            else:
                conn.close()
                continue

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
                    send_json(conn, {"found": False})
                    continue

                obj_data = LOADED_OBJECTS[target_type]
                
                img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
                color = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # --- 关键优化点 ---
                # 原始：depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(g_shape)
                # 优化：Uint16 -> Float32 (米)
                depth_raw = lz4.frame.decompress(depth_bytes)
                depth_mm = np.frombuffer(depth_raw, dtype=np.uint16).reshape(g_shape)
                depth = depth_mm.astype(np.float32) / 1000.0
                # ------------------

                # YOLO (保持原样)
                results = obj_data["yolo"](color, conf=0.5, verbose=False)
                mask = np.zeros(g_shape, dtype=bool)
                
                if len(results[0].boxes) > 0 and results[0].masks is not None:
                    m_data = results[0].masks.data[0].cpu().numpy()
                    if m_data.shape[:2] != color.shape[:2]:
                        m_data = cv2.resize(m_data, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = m_data.astype(bool)

                if mask.sum() < 50:
                    send_json(conn, {"found": False})
                    continue

                # Pose (保持原样)
                pose = obj_data["est"].register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=g_est_refine_iter)
                
                center_pose = pose @ np.linalg.inv(obj_data["to_origin"])
                corners_2d = get_projected_corners(center_pose, obj_data["bbox"], g_K)
                
                send_json(conn, {
                    "found": True,
                    "pose": pose.tolist(),
                    "corners": corners_2d
                })

        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()

if __name__ == '__main__':
    main()
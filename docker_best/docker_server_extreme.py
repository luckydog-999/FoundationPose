import socket
import struct
import json
import numpy as np
import cv2
import lz4.frame
import os
import logging
import torch
import gc
import traceback

from ultralytics import YOLO
from estimater import *
from datareader import *
from my_utils.socket_utils import recvall, send_json, recv_json

# --- 极速配置 ---
torch.set_grad_enabled(False)
HOST = '0.0.0.0'
PORT = 6006

# 追踪参数优化
TRACK_REFINE_ITER = 2   # 追踪时精炼次数 (参考代码用2)
INIT_REFINE_ITER = 5    # 初始化时精炼次数 (参考代码用5)

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
LAST_POSE = {}

def load_models():
    logging.info(">>> Loading models...")
    for obj_name, config in OBJECT_CONFIG.items():
        if not os.path.exists(config["mesh_path"]): 
            logging.warning(f"Mesh not found: {config['mesh_path']}")
            continue
            
        mesh = trimesh.load(config["mesh_path"])
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        
        # 初始化 FoundationPose
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                             mesh=mesh, scorer=scorer, refiner=refiner, 
                             debug_dir="./debug", debug=0, glctx=glctx)
        
        yolo = YOLO(config["yolo_path"])
        
        LOADED_OBJECTS[obj_name] = {
            "est": est, 
            "yolo": yolo, 
            "to_origin": to_origin, 
            "bbox": bbox
        }
        LAST_POSE[obj_name] = None
    logging.info("✅ Models Ready")

def main():
    logging.basicConfig(level=logging.INFO)
    load_models()
    
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print(f"🚀 Extreme Server V3 listening on {PORT}...")

    while True:
        print("Waiting for connection...")
        conn, addr = server_sock.accept()
        print(f"Connected: {addr}")
        
        # 重置状态
        for k in LAST_POSE: LAST_POSE[k] = None
        
        g_K = None
        g_shape = None

        try:
            # 握手
            init_data = recv_json(conn) 
            if init_data and 'K' in init_data:
                g_K = np.array(init_data['K'])
                g_shape = tuple(init_data['shape'])
                send_json(conn, {"status": "ok"})
                print(f"Handshake OK. K shape: {g_K.shape}, Img shape: {g_shape}")
            else:
                conn.close()
                continue
                
            while True:
                # 接收头信息
                header = recvall(conn, 12)
                if not header: break
                rgb_len, depth_len, type_len = struct.unpack('>III', header)
                
                # 接收数据体
                type_bytes = recvall(conn, type_len)
                rgb_bytes = recvall(conn, rgb_len)
                depth_bytes = recvall(conn, depth_len)
                
                target = type_bytes.decode('utf-8')
                if target not in LOADED_OBJECTS: 
                    send_json(conn, {"found": False, "err": "Unknown object"})
                    continue
                    
                obj_data = LOADED_OBJECTS[target]

                # 1. 解码图像
                img = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
                color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 2. 解码深度
                raw_depth = lz4.frame.decompress(depth_bytes)
                depth_uint16 = np.frombuffer(raw_depth, dtype=np.uint16).reshape(g_shape)
                depth = depth_uint16.astype(np.float32) / 1000.0

                pose = None
                
                # --- 核心逻辑修复 ---
                
                # 分支 A: 尝试追踪 (如果上一帧有 Pose)
                if LAST_POSE[target] is not None:
                    try:
                        # 修正：使用 track_one 而不是 track
                        # track_one 通常不需要传入 pose，它内部维护状态，或者根据实现传入
                        # 根据 run_camdemo.py: est.track_one(K=K, rgb=color_image, depth=depth_image, iteration=args.track_refine_iter)
                        pose = obj_data["est"].track_one(K=g_K, rgb=color, depth=depth, iteration=TRACK_REFINE_ITER)
                    except Exception as e:
                        print(f"Track Error: {e}")
                        pose = None

                # 分支 B: 尝试检测并注册 (如果追踪失败)
                if pose is None:
                    # 只有在追踪失败时才运行 YOLO
                    res = obj_data["yolo"](img, conf=0.5, verbose=False) # 使用 BGR 给 YOLO 也可以
                    
                    if len(res[0].boxes) > 0 and res[0].masks:
                        # 获取 Mask
                        mask = res[0].masks.data[0].cpu().numpy().astype(bool)
                        
                        # Resize Mask 如果尺寸不匹配 (YOLO 有时会输出不同尺寸)
                        if mask.shape[:2] != color.shape[:2]:
                             mask = cv2.resize(mask.astype(np.uint8), (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                        
                        try:
                            # 运行注册 (重型计算)
                            print(f"[{target}] Lost track, detecting...")
                            pose = obj_data["est"].register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=INIT_REFINE_ITER)
                        except Exception as e:
                            print(f"Register Error: {e}")
                            pose = None
                    else:
                        # YOLO 也没找到
                        pass

                # 更新状态并发送
                if pose is not None:
                    LAST_POSE[target] = pose
                    # 发送 Pose 矩阵 (4x4)
                    send_json(conn, {"found": True, "pose": pose.tolist()})
                else:
                    LAST_POSE[target] = None
                    send_json(conn, {"found": False})

        except Exception as e:
            print(f"❌ Connection Loop Error: {e}")
            traceback.print_exc()
            conn.close()

if __name__ == '__main__':
    main()
# # docker_server.py
# import socket
# import struct
# import json
# import numpy as np
# import cv2
# import lz4.frame
# import os
# import logging
# import torch
# import gc

# from ultralytics import YOLO
# from estimater import * 
# from datareader import *
# from my_utils.socket_utils import recv_msg, send_msg, recv_json, send_json, recvall

# torch.set_grad_enabled(False)
# HOST = '0.0.0.0'
# PORT = 6006

# OBJECT_CONFIG = {
#     "passive": {
#         "yolo_path": "passive_best.pt", 
#         "mesh_path": "./demo_data/passive/mesh/passive.obj", 
#     },
#     "insert": {
#         "yolo_path": "insert_best.pt",
#         "mesh_path": "./demo_data/insert/mesh/insert.obj", 
#     }
# }

# LOADED_OBJECTS = {} 
# g_K = None
# g_shape = None
# g_est_refine_iter = 1 

# def get_projected_corners(pose, bbox, K):
#     # (保留原函数不变)
#     min_pt = bbox[0]
#     max_pt = bbox[1]
#     corners_3d = np.array([
#         [min_pt[0], min_pt[1], min_pt[2]],
#         [min_pt[0], min_pt[1], max_pt[2]],
#         [min_pt[0], max_pt[1], min_pt[2]],
#         [min_pt[0], max_pt[1], max_pt[2]],
#         [max_pt[0], min_pt[1], min_pt[2]],
#         [max_pt[0], min_pt[1], max_pt[2]],
#         [max_pt[0], max_pt[1], min_pt[2]],
#         [max_pt[0], max_pt[1], max_pt[2]]
#     ])
#     ones = np.ones((8, 1))
#     corners_hom = np.hstack((corners_3d, ones))
#     corners_cam = (pose @ corners_hom.T).T
#     corners_cam = corners_cam[:, :3]
#     projected = (K @ corners_cam.T).T
#     z = projected[:, 2:3] + 1e-5
#     pixels = projected[:, :2] / z
#     return pixels.astype(int).tolist()

# def load_models():
#     logging.info(">>> Loading models...")
#     for obj_name, config in OBJECT_CONFIG.items():
#         logging.info(f"--- Loading: [{obj_name}] ---")
#         if not os.path.exists(config["mesh_path"]) or not os.path.exists(config["yolo_path"]):
#             logging.error(f"❌ File missing for {obj_name}")
#             continue

#         # 加载 Mesh
#         mesh = trimesh.load(config["mesh_path"])
#         to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
#         bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

#         # 初始化 Estimater
#         scorer = ScorePredictor()
#         refiner = PoseRefinePredictor()
#         glctx = dr.RasterizeCudaContext()
#         est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
#                              mesh=mesh, scorer=scorer, refiner=refiner, 
#                              debug_dir="./debug", debug=0, glctx=glctx)
        
#         yolo_model = YOLO(config["yolo_path"])

#         LOADED_OBJECTS[obj_name] = {
#             "est": est, "yolo": yolo_model, "to_origin": to_origin, "bbox": bbox
#         }
#         torch.cuda.empty_cache()
#     logging.info("✅ Models Ready")

# def main():
#     global g_K, g_shape
#     logging.basicConfig(level=logging.INFO)
#     load_models()

#     server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
#     server_sock.bind((HOST, PORT))
#     server_sock.listen(1)
#     print(f"🚀 Docker Server listening on {PORT}...")

#     while True:
#         conn, addr = server_sock.accept()
#         print(f"Connected by {addr}")
        
#         try:
#             # Init
#             init_data = recv_json(conn)
#             if init_data and 'K' in init_data:
#                 g_K = np.array(init_data['K'])
#                 g_shape = tuple(init_data['shape'])
#                 print(f"Client Init: {g_shape}")
#                 send_json(conn, {"status": "ok"})
#             else:
#                 conn.close()
#                 continue

#             while True:
#                 header_data = recvall(conn, 12)
#                 if not header_data: break
#                 rgb_len, depth_len, type_len = struct.unpack('>III', header_data)
                
#                 type_bytes = recvall(conn, type_len)
#                 rgb_bytes = recvall(conn, rgb_len)
#                 depth_bytes = recvall(conn, depth_len)
#                 if not rgb_bytes: break

#                 target_type = type_bytes.decode('utf-8')
                
#                 # Check object
#                 if target_type not in LOADED_OBJECTS:
#                     print(f"Warning: Unknown target {target_type}")
#                     send_json(conn, {"found": False, "err": "Unknown Obj"})
#                     continue

#                 obj_data = LOADED_OBJECTS[target_type]
                
#                 # Decode
#                 img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
#                 color = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
#                 depth_raw = lz4.frame.decompress(depth_bytes)
#                 depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(g_shape)

#                 # YOLO
#                 results = obj_data["yolo"](color, conf=0.5, verbose=False)
#                 mask = np.zeros(g_shape, dtype=bool)
                
#                 yolo_found = False
#                 if len(results[0].boxes) > 0:
#                     yolo_found = True
#                     if results[0].masks is not None:
#                         m_data = results[0].masks.data[0].cpu().numpy()
#                         if m_data.shape[:2] != color.shape[:2]:
#                             m_data = cv2.resize(m_data, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
#                         mask = m_data.astype(bool)
                
#                 if not yolo_found:
#                     # 调试：告诉客户端 YOLO 没看到
#                     send_json(conn, {"found": False, "err": "YOLO Fail"})
#                     continue

#                 if mask.sum() < 50:
#                     send_json(conn, {"found": False, "err": "Mask too small"})
#                     continue

#                 try:
#                     pose = obj_data["est"].register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=g_est_refine_iter)
                    
#                     center_pose = pose @ np.linalg.inv(obj_data["to_origin"])
#                     corners_2d = get_projected_corners(center_pose, obj_data["bbox"], g_K)
                    
#                     send_json(conn, {
#                         "found": True,
#                         "pose": pose.tolist(),
#                         "corners": corners_2d
#                     })
#                 except Exception as e:
#                     print(f"Pose Error: {e}")
#                     # 如果崩了，返回 False 而不是断开连接
#                     send_json(conn, {"found": False, "err": "Pose Calc Fail"})

#         except Exception as e:
#             print(f"Conn Error: {e}")
#         finally:
#             conn.close()

# if __name__ == '__main__':
#     main()

# # docker_server.py
# import socket
# import struct
# import json
# import numpy as np
# import cv2
# import lz4.frame
# import os
# import logging
# import torch
# import gc

# from ultralytics import YOLO
# from estimater import *
# from datareader import *
# from my_utils.socket_utils import recvall, send_json

# # --- 显存救星 ---
# torch.set_grad_enabled(False)
# HOST = '0.0.0.0'
# PORT = 6006

# # 4050 必须设为 1，否则显存爆
# g_est_refine_iter = 1 

# OBJECT_CONFIG = {
#     "passive": {
#         "yolo_path": "passive_best.pt", 
#         "mesh_path": "./demo_data/passive/mesh/passive.obj", 
#     },
#     "insert": {
#         "yolo_path": "insert_best.pt",
#         "mesh_path": "./demo_data/insert/mesh/insert.obj", 
#     }
# }

# LOADED_OBJECTS = {}
# # 记录上一帧的姿态，用于跟踪
# LAST_POSE = {"passive": None, "insert": None}

# def get_projected_corners(pose, bbox, K):
#     min_pt = bbox[0]
#     max_pt = bbox[1]
#     corners_3d = np.array([
#         [min_pt[0], min_pt[1], min_pt[2]],
#         [min_pt[0], min_pt[1], max_pt[2]],
#         [min_pt[0], max_pt[1], min_pt[2]],
#         [min_pt[0], max_pt[1], max_pt[2]],
#         [max_pt[0], min_pt[1], min_pt[2]],
#         [max_pt[0], min_pt[1], max_pt[2]],
#         [max_pt[0], max_pt[1], min_pt[2]],
#         [max_pt[0], max_pt[1], max_pt[2]]
#     ])
#     ones = np.ones((8, 1))
#     corners_hom = np.hstack((corners_3d, ones))
#     corners_cam = (pose @ corners_hom.T).T
#     corners_cam = corners_cam[:, :3]
#     projected = (K @ corners_cam.T).T
#     z = projected[:, 2:3] + 1e-5
#     pixels = projected[:, :2] / z
#     return pixels.astype(int).tolist()

# def load_models():
#     logging.info(">>> Loading models...")
#     for obj_name, config in OBJECT_CONFIG.items():
#         if not os.path.exists(config["mesh_path"]): continue
        
#         logging.info(f"Loading {obj_name}...")
#         mesh = trimesh.load(config["mesh_path"])
#         to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
#         bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

#         scorer = ScorePredictor()
#         refiner = PoseRefinePredictor()
#         glctx = dr.RasterizeCudaContext()
#         est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
#                              mesh=mesh, scorer=scorer, refiner=refiner, 
#                              debug_dir="./debug", debug=0, glctx=glctx)
#         yolo = YOLO(config["yolo_path"])
#         LOADED_OBJECTS[obj_name] = {"est": est, "yolo": yolo, "to_origin": to_origin, "bbox": bbox}
#         torch.cuda.empty_cache()
#     logging.info("✅ Ready")

# def main():
#     logging.basicConfig(level=logging.INFO)
#     load_models()
    
#     server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
#     server_sock.bind((HOST, PORT))
#     server_sock.listen(1)
#     print(f"🚀 Waiting for connection on {PORT}...")

#     while True:
#         conn, addr = server_sock.accept()
#         print(f"Connected: {addr}")
        
#         # 握手
#         try:
#             init_data = json.loads(conn.recv(1024).decode())
#             g_K = np.array(init_data['K'])
#             g_shape = tuple(init_data['shape'])
#             conn.send(json.dumps({"status": "ok"}).encode())
#         except:
#             conn.close()
#             continue

#         while True:
#             try:
#                 # 接收头
#                 header = recvall(conn, 12)
#                 if not header: break
#                 rgb_len, depth_len, type_len = struct.unpack('>III', header)
                
#                 # 接收体
#                 type_bytes = recvall(conn, type_len)
#                 rgb_bytes = recvall(conn, rgb_len)
#                 depth_bytes = recvall(conn, depth_len)
                
#                 target = type_bytes.decode('utf-8')
#                 if target not in LOADED_OBJECTS: continue
#                 obj_data = LOADED_OBJECTS[target]

#                 # 解码
#                 img = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
#                 color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 depth = np.frombuffer(lz4.frame.decompress(depth_bytes), dtype=np.float32).reshape(g_shape)

#                 pose = None
                
#                 # === 🚀 核心策略: 先尝试跟踪 (Track) ===
#                 if LAST_POSE[target] is not None:
#                     try:
#                         # 使用上一帧的姿态进行微调，不需要 YOLO，速度极快
#                         pose = obj_data["est"].track(K=g_K, rgb=color, depth=depth, 
#                                                      pose=LAST_POSE[target], iteration=g_est_refine_iter)
                        
#                         # 防丢检测：如果姿态太离谱(比如跑到相机后面去了)，认为丢失
#                         if np.isnan(pose).any() or pose[2,3] < 0.1:
#                             pose = None
#                             LAST_POSE[target] = None
#                             print(f"[{target}] Lost track, resetting...")
#                     except:
#                         pose = None
#                         LAST_POSE[target] = None

#                 # === 🐢 备用策略: 跟踪失败，使用 YOLO 重检测 ===
#                 if pose is None:
#                     # YOLO 也是大消耗，只在丢失时运行
#                     res = obj_data["yolo"](color, conf=0.5, verbose=False)
#                     mask = None
#                     if len(res[0].boxes) > 0 and res[0].masks:
#                         m = res[0].masks.data[0].cpu().numpy()
#                         if m.shape[:2] != color.shape[:2]:
#                             m = cv2.resize(m, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
#                         mask = m.astype(bool)
                    
#                     if mask is not None and mask.sum() > 100:
#                         try:
#                             pose = obj_data["est"].register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=g_est_refine_iter)
#                         except: pass

#                 # === 发送结果 ===
#                 if pose is not None:
#                     LAST_POSE[target] = pose # 更新这一帧，给下一帧用
#                     center = pose @ np.linalg.inv(obj_data["to_origin"])
#                     corns = get_projected_corners(center, obj_data["bbox"], g_K)
#                     send_json(conn, {"found": True, "pose": pose.tolist(), "corners": corns})
#                     print(f"\r[{target}] Tracking... Z={pose[2,3]:.2f}", end="")
#                 else:
#                     send_json(conn, {"found": False})
#                     print(f"\r[{target}] Searching...", end="")

#             except Exception as e:
#                 print(e)
#                 break
        
#         conn.close()
#         LAST_POSE["passive"] = None
#         LAST_POSE["insert"] = None

# if __name__ == '__main__':
#     main()

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
import time

from ultralytics import YOLO
from estimater import *
from datareader import *
# 假设你的 recvall, send_json 在这里，如果报错请把工具函数贴进来
from my_utils.socket_utils import recvall, send_json

# --- 🔥 显存与速度优化配置 ---
torch.set_grad_enabled(False)
HOST = '0.0.0.0'
PORT = 6006

# 4050 显卡显存较小，保持为 1 最快，如果抖动厉害可改为 2
g_est_refine_iter = 1 

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
# 全局变量记录上一帧姿态
LAST_POSE = {"passive": None, "insert": None}

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
    for obj_name, config in OBJECT_CONFIG.items():
        if not os.path.exists(config["mesh_path"]): continue
        
        logging.info(f"Loading {obj_name}...")
        mesh = trimesh.load(config["mesh_path"])
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                             mesh=mesh, scorer=scorer, refiner=refiner, 
                             debug_dir="./debug", debug=0, glctx=glctx)
        
        # 预加载 YOLO 到 GPU
        yolo = YOLO(config["yolo_path"])
        
        LOADED_OBJECTS[obj_name] = {"est": est, "yolo": yolo, "to_origin": to_origin, "bbox": bbox}
        torch.cuda.empty_cache()
    logging.info("✅ Models Ready")

def main():
    logging.basicConfig(level=logging.INFO)
    load_models()
    
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print(f"🚀 High-Speed Server listening on {PORT}...")

    while True:
        conn, addr = server_sock.accept()
        print(f"Connected: {addr}")
        
        # 握手
        try:
            init_data = json.loads(conn.recv(1024).decode())
            g_K = np.array(init_data['K'])
            g_shape = tuple(init_data['shape'])
            conn.send(json.dumps({"status": "ok"}).encode())
        except Exception as e:
            print(f"Handshake error: {e}")
            conn.close()
            continue

        while True:
            try:
                # 1. 接收头
                header = recvall(conn, 12)
                if not header: break
                rgb_len, depth_len, type_len = struct.unpack('>III', header)
                
                # 2. 接收体
                type_bytes = recvall(conn, type_len)
                rgb_bytes = recvall(conn, rgb_len)
                depth_bytes = recvall(conn, depth_len)
                
                target = type_bytes.decode('utf-8')
                if target not in LOADED_OBJECTS: continue
                obj_data = LOADED_OBJECTS[target]

                # 3. 解码 (CPU 耗时点，但必须做)
                img = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
                color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                depth = np.frombuffer(lz4.frame.decompress(depth_bytes), dtype=np.float32).reshape(g_shape)

                pose = None
                status_msg = "Searching"

                # =========================================================
                # 🚀 高速通道 (Fast Track Mode)
                # =========================================================
                if LAST_POSE[target] is not None:
                    try:
                        # 直接基于上一帧姿态进行 Refine，跳过 YOLO
                        pose = obj_data["est"].track(K=g_K, rgb=color, depth=depth, 
                                                     pose=LAST_POSE[target], iteration=g_est_refine_iter)
                        
                        # --- 🛡️ 防丢检测 (Sanity Check) ---
                        # 1. 检查 NaN
                        if np.isnan(pose).any():
                            raise ValueError("Pose implies NaN")
                        
                        # 2. 检查距离 (防止飞到无穷远或相机背后)
                        # pose[2, 3] 是物体在相机坐标系下的 Z 轴距离 (米)
                        if pose[2, 3] < 0.1 or pose[2, 3] > 3.0:
                            raise ValueError(f"Z-distance abnormal: {pose[2,3]:.2f}")

                        # 3. (可选) 检查瞬移：如果两帧之间移动超过 20cm，认为跟踪失效
                        prev_trans = LAST_POSE[target][:3, 3]
                        curr_trans = pose[:3, 3]
                        dist = np.linalg.norm(curr_trans - prev_trans)
                        if dist > 0.2: 
                            raise ValueError(f"Moved too fast: {dist:.2f}m")

                        status_msg = "Tracking"
                        
                    except Exception as e:
                        # 跟踪失败，降级回检测模式
                        # print(f"[{target}] Lost track: {e}")
                        pose = None
                        LAST_POSE[target] = None
                        status_msg = "Lost"

                # =========================================================
                # 🐢 慢速通道 (Detection Mode) - 只有跟踪丢了才跑
                # =========================================================
                if pose is None:
                    # 运行 YOLO
                    res = obj_data["yolo"](color, conf=0.5, verbose=False)
                    mask = None
                    if len(res[0].boxes) > 0 and res[0].masks:
                        # 找最大的 mask 或者置信度最高的
                        m = res[0].masks.data[0].cpu().numpy()
                        if m.shape[:2] != color.shape[:2]:
                            m = cv2.resize(m, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask = m.astype(bool)
                    
                    if mask is not None and mask.sum() > 50:
                        try:
                            # 重新注册姿态 (Register)
                            pose = obj_data["est"].register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=g_est_refine_iter)
                            status_msg = "Detected"
                        except: 
                            pass

                # =========================================================
                # 📤 结果发送
                # =========================================================
                if pose is not None:
                    LAST_POSE[target] = pose # 更新上一帧，供下一帧 Tracking 使用
                    
                    # 这里的 pose 是 Model -> Camera
                    # 我们需要把 bbox 变换后发回去
                    center_pose = pose @ np.linalg.inv(obj_data["to_origin"])
                    corns = get_projected_corners(center_pose, obj_data["bbox"], g_K)
                    
                    send_json(conn, {"found": True, "pose": pose.tolist(), "corners": corns})
                    print(f"\r[{target}] {status_msg} | Z={pose[2,3]:.3f}m", end="")
                else:
                    LAST_POSE[target] = None # 确保下一帧重新检测
                    send_json(conn, {"found": False})
                    print(f"\r[{target}] Searching...", end="")

            except Exception as e:
                print(f"\nLoop Error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        conn.close()
        # 断开连接时清理显存
        LAST_POSE["passive"] = None
        LAST_POSE["insert"] = None
        torch.cuda.empty_cache()
        print("\nConnection closed. VRAM cleared.")

if __name__ == '__main__':
    main()
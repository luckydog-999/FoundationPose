import socket
import struct
import json
import numpy as np
import cv2
import lz4.frame
import os
import logging
import torch
import traceback
import trimesh

from ultralytics import YOLO
from estimater import *
from datareader import *
from my_utils.socket_utils import recvall, send_json, recv_json

# --- 高性能配置 (High-End GPU Config) ---
torch.set_grad_enabled(False)
HOST = '0.0.0.0'
PORT = 6006

# 🚀 精度优化：针对 3090/4090 等高端卡大幅增加迭代次数
TRACK_REFINE_ITER = 6    # 追踪精炼次数 (从2提升到6，更稳)
INIT_REFINE_ITER = 15    # 初始化精炼次数 (从5提升到15，首次识别极其精准)

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

# --- 辅助：计算3D包围盒投影 ---
def get_projected_bbox(pose, bbox_3d, K):
    """
    计算包围盒8个顶点在图像上的2D坐标
    bbox_3d: (8, 3) 顶点的局部坐标
    """
    # 1. 变换到相机坐标系 (4x4 @ 4x8)
    ones = np.ones((8, 1))
    corners_hom = np.hstack((bbox_3d, ones)) # 8x4
    corners_cam = (pose @ corners_hom.T).T   # 8x4
    corners_cam = corners_cam[:, :3]         # 8x3

    # 2. 投影到像素坐标 (u = fx*x/z + cx)
    projected = (K @ corners_cam.T).T        # 8x3
    z = projected[:, 2:3] + 1e-5             # 避免除以0
    pixels = projected[:, :2] / z
    
    return pixels.astype(int).tolist()

def load_models():
    logging.info(">>> Loading High-Fidelity Models...")
    for obj_name, config in OBJECT_CONFIG.items():
        if not os.path.exists(config["mesh_path"]): 
            logging.warning(f"Mesh not found: {config['mesh_path']}")
            continue
            
        mesh = trimesh.load(config["mesh_path"])
        # 获取 Oriented Bounding Box 的变换和尺寸
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        
        # 生成 8 个角点 (在物体局部坐标系下)
        min_pt = -extents / 2
        max_pt = extents / 2
        corners = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]]
        ])
        
        # 修正中心偏移 (因为 mesh.vertices 是原始坐标)
        # 这里的 bbox_corners 需要配合 center_pose 使用，
        # 或者我们直接保存原始 mesh 坐标系下的 OBB 角点？
        # 为简单起见，FoundationPose 输出的是 Model->Camera，
        # 我们这里计算相对于 Model 原点的 OBB 角点。
        # 变换矩阵 inv(to_origin) 将 OBB 中心对齐回 Mesh 原点
        obb_transform = np.linalg.inv(to_origin)
        corners_hom = np.hstack((corners, np.ones((8, 1))))
        corners_model_space = (obb_transform @ corners_hom.T).T[:, :3]

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                             mesh=mesh, scorer=scorer, refiner=refiner, 
                             debug_dir="./debug", debug=0, glctx=glctx)
        
        yolo = YOLO(config["yolo_path"])
        
        LOADED_OBJECTS[obj_name] = {
            "est": est, 
            "yolo": yolo, 
            "bbox_corners": corners_model_space # 保存模型坐标系下的8个角点
        }
        LAST_POSE[obj_name] = None
    logging.info("✅ High-End Models Ready")

def main():
    logging.basicConfig(level=logging.INFO)
    load_models()
    
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print(f"🚀 High-End Server listening on {PORT}...")

    while True:
        print("Waiting for high-performance client...")
        conn, addr = server_sock.accept()
        print(f"Connected: {addr}")
        
        for k in LAST_POSE: LAST_POSE[k] = None
        g_K = None
        g_shape = None

        try:
            init_data = recv_json(conn) 
            if init_data and 'K' in init_data:
                g_K = np.array(init_data['K'])
                g_shape = tuple(init_data['shape'])
                send_json(conn, {"status": "ok"})
                print(f"Client Init: {g_shape[1]}x{g_shape[0]}")
            else:
                conn.close()
                continue
                
            while True:
                header = recvall(conn, 12)
                if not header: break
                rgb_len, depth_len, type_len = struct.unpack('>III', header)
                
                type_bytes = recvall(conn, type_len)
                rgb_bytes = recvall(conn, rgb_len)
                depth_bytes = recvall(conn, depth_len)
                
                target = type_bytes.decode('utf-8')
                if target not in LOADED_OBJECTS: 
                    send_json(conn, {"found": False, "err": "Unknown object"})
                    continue
                    
                obj_data = LOADED_OBJECTS[target]

                # 1. 解码
                img = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
                color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                raw_depth = lz4.frame.decompress(depth_bytes)
                depth_uint16 = np.frombuffer(raw_depth, dtype=np.uint16).reshape(g_shape)
                depth = depth_uint16.astype(np.float32) / 1000.0

                pose = None
                
                # --- 追踪 ---
                if LAST_POSE[target] is not None:
                    try:
                        # 增加迭代次数以提高平滑度和精度
                        pose = obj_data["est"].track_one(K=g_K, rgb=color, depth=depth, iteration=TRACK_REFINE_ITER)
                    except Exception:
                        pose = None

                # --- 丢失检测 ---
                if pose is None:
                    # 稍微提高一点 conf，减少误检带来的画面抖动
                    res = obj_data["yolo"](img, conf=0.6, verbose=False)
                    
                    if len(res[0].boxes) > 0 and res[0].masks:
                        mask = res[0].masks.data[0].cpu().numpy().astype(bool)
                        if mask.shape[:2] != color.shape[:2]:
                             mask = cv2.resize(mask.astype(np.uint8), (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                        
                        try:
                            # 注册时使用高迭代次数，确保一旦抓住就非常准
                            pose = obj_data["est"].register(K=g_K, rgb=color, depth=depth, ob_mask=mask, iteration=INIT_REFINE_ITER)
                        except Exception:
                            pose = None

                # --- 结果打包 ---
                response = {"found": False}
                if pose is not None:
                    LAST_POSE[target] = pose
                    
                    # 计算漂亮的 3D 包围盒
                    corners_2d = get_projected_bbox(pose, obj_data["bbox_corners"], g_K)
                    
                    response = {
                        "found": True, 
                        "pose": pose.tolist(),
                        "corners": corners_2d # 发送8个角点给前端画框
                    }
                else:
                    LAST_POSE[target] = None
                
                send_json(conn, response)

        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()
            conn.close()

if __name__ == '__main__':
    main()
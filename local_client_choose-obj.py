# local_client.py
# 运行在你的本地电脑上

import pyzed.sl as sl
import cv2
import numpy as np
import requests
import lz4.frame
import time
import logging
from scipy.spatial.transform import Rotation as R
import os
import datetime

# --- 配置 ---
SERVER_URL = "https://u474219-8f05-a40ec3b8.bjb1.seetacloud.com:8443"
INIT_URL = f"{SERVER_URL}/init"
PROCESS_URL = f"{SERVER_URL}/process"
# ------------

# 💡 绘图辅助函数：根据8个点画线框
def draw_3d_box_from_corners(img, corners, color=(0, 255, 0), thickness=2):
    """
    corners: list of 8 points [[x,y], ...]
    Order assumed: 
    0-3: Bottom face (0->1->3->2->0)
    4-7: Top face (4->5->7->6->4)
    Connect pillars (0-4, 1-5, 2-6, 3-7)
    此顺序基于 Trimesh bounds 的一般顺序 (min_xyz 到 max_xyz 的排列)
    """
    if len(corners) != 8: return img
    
    pts = np.array(corners, dtype=np.int32)
    
    # 定义连接关系
    # 注意：trimesh bounds 生成的顺序通常是二进制顺序
    # 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111
    # 这里的 0,1,2... 是上面的 index
    
    # 具体的连接边
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0), # 面1 (例如左面)
        (4, 5), (5, 7), (7, 6), (6, 4), # 面2 (例如右面)
        (0, 4), (1, 5), (2, 6), (3, 7)  # 连接两个面的棱
    ]
    
    for s, e in edges:
        cv2.line(img, tuple(pts[s]), tuple(pts[e]), color, thickness)
        
    return img

def create_transform_matrix(tx, ty, tz, rx_deg, ry_deg, rz_deg, euler_order='xyz'):
    try:
        r = R.from_euler(euler_order, [rx_deg, ry_deg, rz_deg], degrees=True)
        rotation_matrix = r.as_matrix()
    except ValueError:
        return np.eye(4)
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = rotation_matrix
    transform_matrix[0:3, 3] = [tx, ty, tz]
    return transform_matrix

def matrix_to_six_dof(matrix, euler_order='xyz'):
    tx, ty, tz = matrix[0:3, 3]
    r = R.from_matrix(matrix[0:3, 0:3])
    rx, ry, rz = r.as_euler(euler_order, degrees=True)
    return tx, ty, tz, rx, ry, rz

def get_tool_in_base_pose_manual():
    # 你的机械臂位姿 (保持不变)
    input_tx, input_ty, input_tz = 0.359, 0.623, 0.262
    input_rx, input_ry, input_rz = -168.08, 5.1, -50.06
    return create_transform_matrix(input_tx, input_ty, input_tz, input_rx, input_ry, input_rz)

def main():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s') # 简化日志
    
    # 1. ZED Init
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # 🚀 优化：改用 PERFORMANCE 模式，深度计算更快
    init_params.sdk_verbose = 0
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS: exit()

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])
    shape = (cam_info.camera_configuration.resolution.height, cam_info.camera_configuration.resolution.width)

    # 🚀 优化：使用 Session 长连接
    session = requests.Session()

    try:
        session.post(INIT_URL, json={"K": K.tolist(), "shape": shape}, timeout=5)
    except Exception as e:
        print(f"Server Init Error: {e}")
        return

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    runtime_params.enable_depth = True

    T_cam_tool = create_transform_matrix(
        -0.06308799, -0.12889982, -0.00412758, 358.2564, 359.5684, 358.3827
    )
    
    target_object = "passive"
    is_paused = False

    print(">>> 优化版启动：按 '1'/'2' 切换物体, 'p' 暂停 <<<")

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            start_total = time.time()
            
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            color_bgr = image_mat.get_data()[..., :3]

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): is_paused = not is_paused
            elif key == ord('1'): target_object = "passive"
            elif key == ord('2'): target_object = "insert"

            # 暂停显示
            if is_paused:
                cv2.putText(color_bgr, "PAUSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Local View", color_bgr)
                continue

            # 准备数据
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth = depth_mat.get_data()
            depth[np.isnan(depth)] = 0.0
            depth[np.isinf(depth)] = 0.0

            # 🚀 优化：降低 JPEG 质量到 70 (肉眼几乎看不出区别，体积减半)
            _, rgb_jpg = cv2.imencode('.jpg', color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            depth_lz4 = lz4.frame.compress(depth.tobytes())

            try:
                # 发送请求
                t0 = time.time()
                resp = session.post(PROCESS_URL, 
                                    files={'rgb': ('i.jpg', rgb_jpg, 'image/jpeg'), 
                                           'depth': ('d.lz4', depth_lz4, 'application/octet-stream')},
                                    data={'type': target_object},
                                    timeout=3) # 设置短超时，防止卡死
                net_time = (time.time() - t0) * 1000

                if resp.status_code == 200:
                    res_json = resp.json()
                    
                    if res_json.get("found", False):
                        # 1. 获取位姿
                        T_obj_cam = np.array(res_json['pose'])
                        
                        # 2. 本地画图 (利用服务器传回的 corners)
                        corners = res_json.get("corners", [])
                        draw_3d_box_from_corners(color_bgr, corners, color=(0, 255, 0))
                        
                        # 3. 计算机械臂位姿
                        T_tool_base = get_tool_in_base_pose_manual()
                        T_obj_base = T_tool_base @ T_cam_tool @ T_obj_cam
                        tx, ty, tz, rx, ry, rz = matrix_to_six_dof(T_obj_base)

                        # 4. 显示信息
                        fps = 1.0 / (time.time() - start_total)
                        info_text = f"FPS: {fps:.1f} | Net: {net_time:.0f}ms | {target_object}"
                        cv2.putText(color_bgr, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(color_bgr, f"X:{tx:.3f} Y:{ty:.3f} Z:{tz:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        print(f"\rFPS: {fps:.1f} | {target_object} Found", end="")
                    else:
                        cv2.putText(color_bgr, f"{target_object} NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Local View", color_bgr)

            except Exception as e:
                print(f" Net Err: {e}")
                cv2.imshow("Local View", color_bgr)
        
        else:
            time.sleep(0.01)

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
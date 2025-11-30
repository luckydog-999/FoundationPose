# client_ar_tracking.py
# 🚀 终极版：利用 ZED 里程计实现 60FPS 流畅 AR 效果

import pyzed.sl as sl
import cv2
import numpy as np
import socket
import struct
import lz4.frame
import time
import threading
import queue

# 引用您的工具函数
from my_utils.socket_utils import recv_json, send_json
from my_utils.math_utils import (
    matrix_to_six_dof, 
    get_tool_in_base_pose_manual,
    create_transform_matrix,
    draw_axis 
)

# --- 配置 ---
SERVER_IP = "127.0.0.1"
SERVER_PORT = 6006
JPEG_QUALITY = 90  # 保持高画质

# --- 全局状态 ---
# 存储最新的物体在"世界坐标系"下的位姿 (4x4 Matrix)
g_obj_world_pose = None 
g_last_update_time = 0
g_running = True

# 请求队列 (用于线程通信)
g_request_queue = queue.Queue(maxsize=1) 

def network_worker(K_list, shape):
    """网络线程：负责慢速的上传和接收，不卡主界面"""
    global g_obj_world_pose, g_last_update_time, g_running
    
    print("🔵 Network thread started...")
    
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        client_sock.connect((SERVER_IP, SERVER_PORT))
        send_json(client_sock, {"K": K_list, "shape": shape})
        if recv_json(client_sock).get("status") != "ok": 
            print("❌ Server Handshake failed")
            return
    except Exception as e:
        print(f"❌ Connect Error: {e}")
        return

    while g_running:
        try:
            # 1. 从队列取数据 (阻塞等待)
            # data: (rgb_encoded, depth_encoded, t_cam_world_at_capture, target_name)
            data = g_request_queue.get(timeout=1.0) 
            
            rgb_enc, depth_enc, cam_pose_at_capture, target = data
            t_bytes = target.encode('utf-8')
            
            # 2. 发送
            head = struct.pack('>III', len(rgb_enc), len(depth_enc), len(t_bytes))
            client_sock.sendall(head + t_bytes + rgb_enc + depth_enc)
            
            # 3. 接收结果 (耗时操作)
            res = recv_json(client_sock)
            
            if res and res.get("found"):
                # T_obj_cam: 物体相对于那一帧相机的位姿
                T_obj_cam = np.array(res['pose'])
                
                # 🌟 核心魔法：算出物体在世界坐标系的绝对位置
                # T_obj_world = T_cam_world * T_obj_cam
                T_obj_world = np.dot(cam_pose_at_capture, T_obj_cam)
                
                # 更新全局变量
                g_obj_world_pose = T_obj_world
                g_last_update_time = time.time()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Network Error: {e}")
            break
            
    client_sock.close()

def main():
    global g_running, g_obj_world_pose

    # 1. 初始化 ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS: exit()

    # 🌟 关键：开启位置追踪 (Odometry)
    # 这允许 ZED 知道相机移动了多少
    track_params = sl.PositionalTrackingParameters()
    if zed.enable_positional_tracking(track_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Positional Tracking failed to start!")
        exit()

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])
    h = cam_info.camera_configuration.resolution.height
    w = cam_info.camera_configuration.resolution.width

    # 2. 启动网络线程
    t_net = threading.Thread(target=network_worker, args=(K.tolist(), (h, w)))
    t_net.daemon = True
    t_net.start()
    
    # 3. 准备手眼矩阵
    T_cam_tool = create_transform_matrix(
        -0.06308799, -0.12889982, -0.00412758, 358.2564, 359.5684, 358.3827
    )

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    runtime.enable_depth = True
    
    # ZED Pose 对象
    zed_pose = sl.Pose()
    
    target = "passive"
    print("🚀 AR Client Running... (Press 'q' to quit)")

    # FPS 统计
    local_frames = 0
    start_time = time.time()

    while g_running:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            local_frames += 1
            
            # --- A. 获取当前帧和当前位姿 ---
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            
            # 获取当前时刻相机的世界位姿 (T_cam_current)
            state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
            # 注意：Transform 转 numpy matrix
            T_cam_world_current = zed_pose.pose_data(sl.Transform()).m 
            
            # 准备图像
            img = image_mat.get_data()[:, :, :3]
            img = np.ascontiguousarray(img)

            # --- B. 尝试把数据塞给网络线程 (如果它空闲) ---
            if g_request_queue.empty():
                depth = depth_mat.get_data()
                depth[np.isnan(depth)] = 0.0
                
                # 压缩 (Uint16)
                depth_mm = (depth * 1000).astype(np.uint16)
                d_lz4 = lz4.frame.compress(depth_mm.tobytes())
                _, rgb_j = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                
                # 🌟 关键：把这张图对应的“拍摄时刻相机位姿”也传过去
                # 必须用 copy() 否则可能会变
                pose_snapshot = T_cam_world_current.copy()
                
                g_request_queue.put((rgb_j.tobytes(), d_lz4, pose_snapshot, target))

            # --- C. 渲染 (AR 补偿) ---
            # 即使服务器没返回，我们也能用 g_obj_world_pose 和当前的 T_cam_world_current 算出物体应该在哪
            if g_obj_world_pose is not None:
                # T_obj_cam_new = inv(T_cam_world_new) * T_obj_world
                T_world_cam_current = np.linalg.inv(T_cam_world_current)
                T_obj_cam_render = np.dot(T_world_cam_current, g_obj_world_pose)
                
                # 1. 画轴 (看起来像吸在桌子上不动)
                draw_axis(img, T_obj_cam_render, K)
                
                # 2. 计算基座坐标 (显示给机械臂用)
                T_tool_base = get_tool_in_base_pose_manual()
                # T_obj_base 不会变，因为它是世界坐标
                # 但为了显示稳定，我们直接用计算好的
                T_obj_base = T_tool_base @ T_cam_tool @ T_obj_cam_render
                
                x, y, z, rx, ry, rz = matrix_to_six_dof(T_obj_base)
                
                # 提示信息
                time_diff = time.time() - g_last_update_time
                status_color = (0, 255, 0) if time_diff < 1.0 else (0, 255, 255) # 超过1秒没更新变黄
                
                cv2.putText(img, f"Base: {x:.3f} {y:.3f} {z:.3f}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(img, f"Delay: {time_diff:.1f}s", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

            # --- D. 控制与显示 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): g_running = False
            elif key == ord('1'): target = "passive"
            elif key == ord('2'): target = "insert"

            # 显示本地 FPS (应该是 30-60)
            if local_frames % 30 == 0:
                dt = time.time() - start_time
                local_fps = local_frames / dt
                local_frames = 0
                start_time = time.time()
                # print(f"Local FPS: {local_fps:.1f}") # 可选打印

            cv2.imshow("AR Tracking Client", img)
            
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
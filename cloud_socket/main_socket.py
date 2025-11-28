# local_client_socket.py
import pyzed.sl as sl
import cv2
import numpy as np
import socket
import struct
import lz4.frame
import time
import sys

# ✅ 修正导入：直接从文件导入函数，避免 'module not callable'
from my_utils.socket_utils import recv_json, send_json
from my_utils.math_utils import (
    create_transform_matrix, 
    matrix_to_six_dof, 
    get_tool_in_base_pose_manual,
    draw_axis # <--- 用这个新的轻量级画图函数
)

# --- 配置 ---
SERVER_IP = "127.0.0.1" # 请确保用了 VSCode 端口转发
SERVER_PORT = 6006

def main():
    # 1. 初始化 ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    init_params.sdk_verbose = 0
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS: exit()

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])
    shape = (cam_info.camera_configuration.resolution.height, cam_info.camera_configuration.resolution.width)

    # 2. 连接服务器
    print(f">>> Connecting to {SERVER_IP}:{SERVER_PORT}...")
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        client_sock.connect((SERVER_IP, SERVER_PORT))
        # 握手
        send_json(client_sock, {"K": K.tolist(), "shape": shape})
        if recv_json(client_sock).get("status") != "ok": return
        print(">>> Server Ready. Let's go!")
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    # 3. 准备手眼矩阵 (Cam -> Tool)
    # 请确认这是你标定好的准确数值！
    T_cam_tool = create_transform_matrix(
        -0.06308799, -0.12889982, -0.00412758, 358.2564, 359.5684, 358.3827
    )

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    runtime.enable_depth = True
    target = "passive"
    
    print("\n" + "="*50)
    print("   按 '1': 找 Passive (圆柱)")
    print("   按 '2': 找 Insert (盖子)")
    print("   按 'p': 暂停画面")
    print("   按 'q': 退出")
    print("="*50 + "\n")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            start_t = time.time()
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            # 必须转连续内存，否则OpenCV画图会崩
            img = np.ascontiguousarray(image_mat.get_data()[:, :, :3])

            # 按键控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('1'): target = "passive"
            elif key == ord('2'): target = "insert"
            
            # 数据准备
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth = depth_mat.get_data()
            depth[np.isnan(depth)] = 0.0
            
            # 压缩发送 (JPEG 95 保证精度)
            _, rgb_j = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            d_lz4 = lz4.frame.compress(depth.tobytes())
            t_bytes = target.encode('utf-8')
            
            try:
                # 发送
                head = struct.pack('>III', len(rgb_j), len(d_lz4), len(t_bytes))
                client_sock.sendall(head + t_bytes + rgb_j.tobytes() + d_lz4)
                
                # 接收
                res = recv_json(client_sock)
                
                if res and res.get("found"):
                    # 1. 拿到原始位姿 (Object -> Cam)
                    T_obj_cam = np.array(res['pose'])
                    
                    # 2. 画坐标轴 (比画框快，且更清晰)
                    draw_axis(img, T_obj_cam, K)
                    
                    # 3. 核心：计算基座坐标 (Object -> Base)
                    # 链式法则: Base_T_Obj = Base_T_Tool * Tool_T_Cam * Cam_T_Obj
                    T_tool_base = get_tool_in_base_pose_manual()
                    T_obj_base = T_tool_base @ T_cam_tool @ T_obj_cam
                    
                    # 转成欧拉角方便看
                    x, y, z, rx, ry, rz = matrix_to_six_dof(T_obj_base)

                    # 4. 打印给用户看 (重点！！！)
                    print(f"\r✅ [{target}] Base Pose: X={x:.4f}, Y={y:.4f}, Z={z:.4f} | Rz={rz:.1f}°", end="")
                    
                    # 屏幕上也显示
                    cv2.putText(img, f"Base X:{x:.3f} Y:{y:.3f} Z:{z:.3f}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"RX:{rx:.1f} RY:{ry:.1f} RZ:{rz:.1f}", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    print(f"\r❌ [{target}] Searching...", end="")
                    cv2.putText(img, "NOT FOUND", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Client", img)

            except Exception as e:
                print(f"\nError: {e}")
                break
        else:
            time.sleep(0.001)

    client_sock.close()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
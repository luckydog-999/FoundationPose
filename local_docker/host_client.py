# host_client.py
# 运行环境：Windows 宿主机
# 作用：读取 ZED 相机 -> 发送给 localhost Docker -> 接收位姿 -> 画图

import pyzed.sl as sl
import cv2
import numpy as np
import socket
import struct
import lz4.frame
import time

# 确保本地文件夹里有 my_utils
from my_utils.socket_utils import recv_json, send_json
from my_utils.math_utils import (
    create_transform_matrix, 
    matrix_to_six_dof, 
    get_tool_in_base_pose_manual,
    draw_axis
)

# --- 配置 ---
# 因为是本地 Docker，IP 永远是 127.0.0.1
SERVER_IP = "127.0.0.1" 
SERVER_PORT = 6006 

def main():
    # 1. 初始化 ZED
    print(">>> Initializing ZED Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # PERFORMANCE 模式对 4050 压力小一些
    init_params.sdk_verbose = 0
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"❌ Failed to open camera: {err}")
        exit()

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])
    shape = (cam_info.camera_configuration.resolution.height, cam_info.camera_configuration.resolution.width)

    # 2. 连接 Docker 本地服务器
    print(f">>> Connecting to Docker at {SERVER_IP}:{SERVER_PORT}...")
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        client_sock.connect((SERVER_IP, SERVER_PORT))
        # 握手
        send_json(client_sock, {"K": K.tolist(), "shape": shape})
        res = recv_json(client_sock)
        if not res or res.get("status") != "ok": 
            print("❌ Handshake failed.")
            return
        print("✅ Connected to Docker!")
    except Exception as e:
        print(f"❌ Connection Failed. Make sure Docker is running and port {SERVER_PORT} is mapped.")
        print(f"Error details: {e}")
        return

    # 3. 手眼标定矩阵 (请确认此数值准确)
    T_cam_tool = create_transform_matrix(
        -0.06308799, -0.12889982, -0.00412758, 358.2564, 359.5684, 358.3827
    )

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    runtime.enable_depth = True
    target = "passive" # 默认物体
    
    print("\n" + "="*50)
    print(f"   🚀 Running on Localhost (Docker)")
    print("   [1]: Target Passive")
    print("   [2]: Target Insert")
    print("   [q]: Quit")
    print("="*50 + "\n")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            start_t = time.time()
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            img = np.ascontiguousarray(image_mat.get_data()[:, :, :3])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('1'): target = "passive"
            elif key == ord('2'): target = "insert"
            
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth = depth_mat.get_data()
            depth[np.isnan(depth)] = 0.0
            
            # 压缩发送 (本地传输速度快，JPGE 95 几乎无损)
            _, rgb_j = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            d_lz4 = lz4.frame.compress(depth.tobytes())
            t_bytes = target.encode('utf-8')
            
            try:
                head = struct.pack('>III', len(rgb_j), len(d_lz4), len(t_bytes))
                client_sock.sendall(head + t_bytes + rgb_j.tobytes() + d_lz4)
                
                res = recv_json(client_sock)
                
                fps = 1.0 / (time.time() - start_t)
                cv2.putText(img, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if res and res.get("found"):
                    T_obj_cam = np.array(res['pose'])
                    
                    # 画图
                    draw_axis(img, T_obj_cam, K)
                    if "corners" in res:
                        pts = np.array(res["corners"])
                        for p in pts:
                            cv2.circle(img, tuple(p), 3, (255, 0, 0), -1)

                    # 计算基座坐标
                    T_tool_base = get_tool_in_base_pose_manual()
                    T_obj_base = T_tool_base @ T_cam_tool @ T_obj_cam
                    x, y, z, rx, ry, rz = matrix_to_six_dof(T_obj_base)

                    print(f"\r✅ [{target}] X={x:.4f} Y={y:.4f} Z={z:.4f} Rz={rz:.1f}°", end="")
                    
                    cv2.putText(img, f"Base X:{x:.3f} Y:{y:.3f} Z:{z:.3f}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    err_msg = res.get("err", "") if res else ""
                    print(f"\r❌ [{target}] Searching... {err_msg}", end="")
                    cv2.putText(img, "NOT FOUND", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Local Client", img)

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
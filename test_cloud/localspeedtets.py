import pyzed.sl as sl
import cv2
import numpy as np
import socket
import struct
import lz4.frame
import time
import json

# ==========================================
# 1. 严格复刻 socket_utils.py 的内容
# ==========================================
def send_msg(sock, data):
    msg = struct.pack('>I', len(data)) + data
    sock.sendall(msg)

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen: return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)

def send_json(sock, data_dict):
    json_bytes = json.dumps(data_dict).encode('utf-8')
    send_msg(sock, json_bytes)

def recv_json(sock):
    data = recv_msg(sock)
    if data is None: return None
    return json.loads(data.decode('utf-8'))
# ==========================================

# --- 配置 ---
SERVER_IP = "127.0.0.1" # ⚠️ 记得改回你的服务器IP
SERVER_PORT = 6006

def main():
    # 1. ZED 初始化
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    init_params.camera_fps = 30
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED Open Failed")
        exit()

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])
    shape = (cam_info.camera_configuration.resolution.height, cam_info.camera_configuration.resolution.width)

    # 2. 连接
    print(f">>> Connecting to {SERVER_IP}:{SERVER_PORT}...")
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        client_sock.connect((SERVER_IP, SERVER_PORT))
        # [握手] 使用 socket_utils 协议
        send_json(client_sock, {"K": K.tolist(), "shape": shape})
        res = recv_json(client_sock)
        if res is None or res.get("status") != "ok": 
            print("Handshake failed")
            return
        print(">>> Server Ready. Speed Test Started!")
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    target = "test_obj"
    
    prev_time = time.time()

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            img = np.ascontiguousarray(image_mat.get_data()[:, :, :3])

            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth = depth_mat.get_data()
            depth[np.isnan(depth)] = 0.0
            
            # --- 压缩 ---
            _, rgb_j = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            d_lz4 = lz4.frame.compress(depth.tobytes())
            t_bytes = target.encode('utf-8')
            
            try:
                # --- [发送核心] 严格复刻 main_socket.py 协议 ---
                # 12字节头 (3个int)
                head = struct.pack('>III', len(rgb_j), len(d_lz4), len(t_bytes))
                
                # 发送: 头 + Type + RGB + Depth
                # 注意: rgb_j 和 d_lz4 是 bytes，t_bytes 也是 bytes，可以直接拼接
                client_sock.sendall(head + t_bytes + rgb_j.tobytes() + d_lz4)
                
                # --- 等待响应 (socket_utils 协议) ---
                _ = recv_json(client_sock)
                
                # --- FPS ---
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time

                cv2.putText(img, f"Client FPS: {fps:.1f}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Client View", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            except Exception as e:
                print(f"\nError: {e}")
                break

    client_sock.close()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
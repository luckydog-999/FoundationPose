import pyzed.sl as sl
import cv2
import numpy as np
import socket
import struct
import lz4.frame
import time
import json
import traceback

# --- 配置 ---
SERVER_IP = "127.0.0.1" 
SERVER_PORT = 6006       # 务必确认服务端也是 6006
TARGET_OBJECT = "insert" # 必须与服务端 load_models 里的名字一致

# --- Socket 工具 ---
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

def recv_json(sock):
    raw_len = recvall(sock, 4)
    if not raw_len: return None
    msglen = struct.unpack('>I', raw_len)[0]
    return json.loads(recvall(sock, msglen).decode('utf-8'))

def send_json(sock, data):
    send_msg(sock, json.dumps(data).encode('utf-8'))

# --- 仿 FoundationPose 风格绘图 (轻量化移植版) ---
def draw_axis_foundation(img, pose, K):
    """
    移植自 FoundationPose 的 draw_xyz_axis，去除 OpenGL 依赖，
    改为纯 OpenCV 实现，效果一致。
    """
    # 坐标轴长度 0.1米 (10cm)
    scale = 0.1 
    points_3d = np.float32([
        [0, 0, 0],      # 原点
        [scale, 0, 0],  # X
        [0, scale, 0],  # Y
        [0, 0, scale]   # Z
    ])
    
    # 3D -> 2D 投影
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    # 相机坐标系下的点
    points_cam = (R @ points_3d.T).T + t
    
    # 避免除以零 (物体在相机背面时)
    if np.any(points_cam[:, 2] <= 0.001): 
        return

    # 投影公式: x = (X * fx / Z) + cx
    z = points_cam[:, 2]
    x = (points_cam[:, 0] * K[0,0] / z) + K[0,2]
    y = (points_cam[:, 1] * K[1,1] / z) + K[1,2]
    
    pts_2d = np.stack([x, y], axis=1).astype(int)
    
    origin = tuple(pts_2d[0])
    
    # 绘图：FoundationPose 风格是 X红, Y绿, Z蓝，粗细为 3
    # OpenCV 是 BGR 顺序，所以颜色代码是 (B, G, R)
    cv2.line(img, origin, tuple(pts_2d[1]), (0, 0, 255), 3)   # X轴 - 红色
    cv2.line(img, origin, tuple(pts_2d[2]), (0, 255, 0), 3)   # Y轴 - 绿色
    cv2.line(img, origin, tuple(pts_2d[3]), (255, 0, 0), 3)   # Z轴 - 蓝色
    
    # 画个中心点
    cv2.circle(img, origin, 5, (0, 255, 255), -1)

def main():
    # 1. 初始化 ZED
    print("Opening ZED...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    # ⚠️ 必须与服务端一致的分辨率，推荐 VGA 跑高帧率
    init_params.camera_resolution = sl.RESOLUTION.VGA 
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    init_params.camera_fps = 60 # 尝试 60 FPS
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ ZED Open Failed")
        return

    cam_info = zed.get_camera_information()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height
    
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])

    print(f"ZED Ready: {width}x{height} @ {init_params.camera_fps}FPS")
    
    # 2. 连接服务端
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        print(f"Connecting to {SERVER_IP}:{SERVER_PORT}...")
        client_sock.connect((SERVER_IP, SERVER_PORT))
        # 握手：发送相机参数
        send_json(client_sock, {"K": K.tolist(), "shape": (height, width)})
        res = recv_json(client_sock)
        if not res or res.get("status") != "ok":
            print("❌ Handshake failed!")
            return
        print("✅ Linked!")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    target_bytes = TARGET_OBJECT.encode('utf-8')

    print(">>> Starting Loop. Press 'q' to exit.")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            t0 = time.time()
            
            # 取数据
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            
            # 🟢【关键修复】强制转为连续内存
            # ZED 默认是 BGRA (4通道)，切片取 BGR 后内存不连续，必须 ascontiguousarray
            img_bgra = image_mat.get_data()
            img = np.ascontiguousarray(img_bgra[:, :, :3]) 
            
            depth = depth_mat.get_data()
            
            # 数据清洗
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            # 压缩发送
            _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            depth_uint16 = (depth * 1000).astype(np.uint16)
            depth_compressed = lz4.frame.compress(depth_uint16.tobytes())
            
            header = struct.pack('>III', len(img_encoded), len(depth_compressed), len(target_bytes))
            client_sock.sendall(header + target_bytes + img_encoded.tobytes() + depth_compressed)
            
            # 接收结果
            res = recv_json(client_sock)
            
            # 计算 FPS
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0
            
            # 绘图
            if res and res.get("found"):
                pose = np.array(res['pose'])
                try:
                    draw_axis_foundation(img, pose, K)
                    dist = pose[2, 3]
                    cv2.putText(img, f"Dist: {dist:.2f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as draw_err:
                    print(f"Draw Err: {draw_err}")
            else:
                cv2.putText(img, "SEARCHING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 此时 img 已经是连续内存，cv2.putText 绝对不会再报错
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Client V3", img)
            if cv2.waitKey(1) == ord('q'):
                break

    client_sock.close()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import pyzed.sl as sl
import cv2
import numpy as np
import socket
import struct
import lz4.frame
import time
import json

# --- 配置 ---
SERVER_IP = "127.0.0.1" 
SERVER_PORT = 6006
TARGET_OBJECT = "insert" 

# --- Socket Utils ---
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

# --- 🎨 美观可视化绘制 ---

def draw_3d_bbox(img, corners):
    """
    绘制半透明的3D包围盒，视觉效果更佳
    corners: list of [x, y] from server (8 points)
    """
    if corners is None or len(corners) != 8: return

    # 定义立方体的12条棱 (基于 0-7 的顶点顺序)
    # 通常顺序是: 前面4个(0-3), 后面4个(4-7)
    # 连接关系取决于服务端生成 bbox 的顺序，这里假设是标准的 trimesh 顺序
    lines = [
        (0, 1), (1, 3), (3, 2), (2, 0), # 前面
        (4, 5), (5, 7), (7, 6), (6, 4), # 后面
        (0, 4), (1, 5), (2, 6), (3, 7)  # 中间连接
    ]
    
    pts = np.array(corners, dtype=np.int32)
    
    # 1. 绘制线条 (亮绿色，抗锯齿)
    color_line = (0, 255, 127) # SpringGreen
    thickness = 2
    
    for start, end in lines:
        pt1 = tuple(pts[start])
        pt2 = tuple(pts[end])
        cv2.line(img, pt1, pt2, color_line, thickness, cv2.LINE_AA)
        
    # 2. 绘制角点 (小圆点)
    for pt in pts:
        cv2.circle(img, tuple(pt), 4, (0, 200, 255), -1, cv2.LINE_AA)

def draw_axis_smooth(img, pose, K):
    """绘制平滑的坐标轴"""
    scale = 0.08 # 坐标轴长度 8cm
    points_3d = np.float32([[0,0,0], [scale,0,0], [0,scale,0], [0,0,scale]])
    
    R, t = pose[:3, :3], pose[:3, 3]
    points_cam = (R @ points_3d.T).T + t
    
    if np.any(points_cam[:, 2] <= 0.001): return

    z = points_cam[:, 2]
    x = (points_cam[:, 0] * K[0,0] / z) + K[0,2]
    y = (points_cam[:, 1] * K[1,1] / z) + K[1,2]
    pts_2d = np.stack([x, y], axis=1).astype(int)
    
    origin = tuple(pts_2d[0])
    # BGR
    cv2.line(img, origin, tuple(pts_2d[1]), (50, 50, 255), 3, cv2.LINE_AA) # X Red
    cv2.line(img, origin, tuple(pts_2d[2]), (50, 255, 50), 3, cv2.LINE_AA) # Y Green
    cv2.line(img, origin, tuple(pts_2d[3]), (255, 100, 50), 3, cv2.LINE_AA) # Z Blue

def main():
    print(">>> Initializing ZED (High Quality Mode)...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    
    # 🔥 升级1: 分辨率提升到 HD720 (如果带宽允许，可尝试 HD1080)
    init_params.camera_resolution = sl.RESOLUTION.HD720 
    init_params.coordinate_units = sl.UNIT.METER
    
    # 🔥 升级2: 开启 NEURAL 深度模式 (仅高端 N卡 可用，精度最高)
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
    # 如果 NEURAL 跑不动，改为 sl.DEPTH_MODE.ULTRA
    
    init_params.camera_fps = 60 
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ ZED Open Failed. Check USB 3.0 or CUDA.")
        return

    cam_info = zed.get_camera_information()
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])

    print(f"✅ ZED Ready: {width}x{height} @ Neural Mode")
    
    # 连接 Server
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        client_sock.connect((SERVER_IP, SERVER_PORT))
        send_json(client_sock, {"K": K.tolist(), "shape": (height, width)})
        res = recv_json(client_sock)
        if not res or res.get("status") != "ok": return
    except Exception as e:
        print(f"Conn Err: {e}")
        return

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime = sl.RuntimeParameters()
    # 开启置信度阈值，过滤深度噪声
    runtime.confidence_threshold = 95 
    
    target_bytes = TARGET_OBJECT.encode('utf-8')

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            t0 = time.time()
            
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            
            img_bgra = image_mat.get_data()
            img = np.ascontiguousarray(img_bgra[:, :, :3]) 
            depth = depth_mat.get_data()
            depth = np.nan_to_num(depth, nan=0.0)

            # 🔥 升级3: 提高 JPEG 质量 (70 -> 90) 减少压缩伪影
            _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            depth_uint16 = (depth * 1000).astype(np.uint16)
            depth_compressed = lz4.frame.compress(depth_uint16.tobytes())
            
            header = struct.pack('>III', len(img_encoded), len(depth_compressed), len(target_bytes))
            client_sock.sendall(header + target_bytes + img_encoded.tobytes() + depth_compressed)
            
            res = recv_json(client_sock)
            
            fps = 1.0 / (time.time() - t0)
            
            if res and res.get("found"):
                pose = np.array(res['pose'])
                corners = res.get('corners') # 接收8个角点
                
                # 绘制 3D 框
                draw_3d_bbox(img, corners)
                # 绘制 坐标轴
                draw_axis_smooth(img, pose, K)
                
                # 绘制距离信息标签
                dist = pose[2, 3]
                label = f"{TARGET_OBJECT}: {dist:.3f}m"
                cv2.putText(img, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, "SEARCHING...", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(img, f"FPS: {fps:.1f} (High-Res)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)
            
            # 缩放一点显示，不然 720p/1080p 在某些屏幕上太大
            display_img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
            cv2.imshow("High-End Client", display_img)
            
            if cv2.waitKey(1) == ord('q'):
                break

    client_sock.close()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
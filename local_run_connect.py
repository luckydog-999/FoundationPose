import pyzed.sl as sl
import cv2
import numpy as np
import requests
import lz4.frame
import time
import socket
import json
import threading
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# --- 配置 ---
DOCKER_URL = "http://127.0.0.1:6006/process" # 本地 Docker 地址
YOLO_PATH = "best.pt"                        # 你的 YOLO 模型路径

# --- 机械臂通信配置 ---
ROBOT_IP = '192.168.1.20'  # Qt 发送端电脑 IP
ROBOT_PORT = 6666          # Qt 发送端端口

# --- 坐标变换辅助函数 ---
def create_transform_matrix(tx, ty, tz, rx, ry, rz):
    """
    根据 x,y,z, rx,ry,rz (角度制) 生成 4x4 变换矩阵
    """
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    m = np.eye(4)
    m[:3, :3] = r.as_matrix()
    m[:3, 3] = [tx, ty, tz]
    return m

def matrix_to_six_dof(m):
    tx, ty, tz = m[:3, 3]
    rx, ry, rz = R.from_matrix(m[:3, :3]).as_euler('xyz', degrees=True)
    return tx, ty, tz, rx, ry, rz

# --- 核心类：负责后台接收机械臂位姿 ---
class RobotClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.latest_matrix = None # 存储最新的 4x4 变换矩阵
        self.lock = threading.Lock() # 线程锁，防止读写冲突
        self.running = False
        self.connected = False

    def connect_and_listen(self):
        """连接并启动监听循环（放入子线程运行）"""
        while True: # 断线重连大循环
            try:
                print(f"[RobotClient] 正在连接 {self.ip}:{self.port} ...")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.ip, self.port))
                self.connected = True
                self.running = True
                print("[RobotClient] 连接成功，等待数据...")

                while self.running:
                    # 阻塞接收数据
                    data = self.sock.recv(4096) 
                    if not data:
                        print("[RobotClient] 对方断开连接")
                        break
                    
                    try:
                        # 解析数据 (处理粘包是个复杂话题，这里假设每次发送完整的 JSON)
                        # 如果 Qt 发送频率很高，建议 Qt 端每条消息加换行符，这里用 readline
                        json_str = data.decode('utf-8').strip()
                        # 防止多个JSON连在一起导致解析失败，只取最后一个有效的（粗略处理）
                        if "}{" in json_str:
                            json_str = "{" + json_str.split("}{")[-1]

                        msg = json.loads(json_str)

                        if msg.get("command") == "robot_flange_pose":
                            # 假设 pose 格式为 [x, y, z, rx, ry, rz]
                            pose = msg["pose"] 
                            
                            # 转换为矩阵
                            mat = create_transform_matrix(pose[0], pose[1], pose[2], 
                                                          pose[3], pose[4], pose[5])
                            
                            # 线程安全地更新数据
                            with self.lock:
                                self.latest_matrix = mat
                            
                            # 可选：打印调试
                            # print(f"[RobotClient] 更新位姿: {pose}")

                    except json.JSONDecodeError:
                        pass # 忽略解析错误
                    except Exception as e:
                        print(f"[RobotClient] 数据处理错误: {e}")

            except Exception as e:
                print(f"[RobotClient] 连接错误: {e}, 3秒后重试...")
                self.connected = False
                time.sleep(3)
            finally:
                if self.sock: self.sock.close()

    def start(self):
        """启动子线程"""
        t = threading.Thread(target=self.connect_and_listen, daemon=True)
        t.start()

    def get_latest_pose(self):
        """主线程调用此函数获取最新位姿"""
        with self.lock:
            return self.latest_matrix if self.latest_matrix is not None else None

# --- 主程序 ---
def main():
    # 1. 启动机械臂通信线程
    robot_client = RobotClient(ROBOT_IP, ROBOT_PORT)
    robot_client.start()

    # 2. 本地加载 YOLO
    print("正在本地加载 YOLO...")
    model = YOLO(YOLO_PATH)
    
    # 3. 初始化 ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED 打开失败")
        return

    # 获取内参
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    K_str = f"{calib.fx},0,{calib.cx},0,{calib.fy},{calib.cy},0,0,1"
    
    # 手眼标定矩阵 (ZED 到 机械臂法兰)
    # 注意：这个矩阵必须非常精确，建议你确认这是 Eye-in-Hand (相机在手上) 还是 Eye-to-Hand
    T_cam_tool = create_transform_matrix(-0.063, -0.128, -0.004, 358.2, 359.5, 358.3)

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    print(">>> 系统就绪，等待机械臂数据... (按 'q' 退出) <<<")

    while True:
        # 检查 ZED 图像
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            
            img_bgr = image_mat.get_data()[..., :3]
            depth = depth_mat.get_data()
            depth[np.isnan(depth)] = 0
            depth[np.isinf(depth)] = 0

            # --- A. 本地 YOLO 推理 ---
            results = model(img_bgr, conf=0.85, verbose=False)
            
            mask_img = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            has_detection = False
            
            if results[0].masks is not None and len(results[0].masks.data) > 0:
                m = results[0].masks.data[0].cpu().numpy()
                if m.shape != img_bgr.shape[:2]:
                    m = cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_img = (m * 255).astype(np.uint8)
                has_detection = True
            
            if not has_detection:
                cv2.putText(img_bgr, "No Object", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Local View", img_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # --- B. 获取最新的机械臂位姿 (关键步骤) ---
            # 这里实现了 "一定要等发送" 的逻辑
            # 如果 robot_client 还没有收到过数据，T_tool_base 就是 None
            T_tool_base = robot_client.get_latest_pose()

            if T_tool_base is None:
                print("等待机械臂位姿数据中...")
                cv2.putText(img_bgr, "Waiting for Robot Data...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Local View", img_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue # 跳过本次循环，直到收到数据

            # --- C. 发送视觉数据到 Docker 计算物体相对相机的位姿 ---
            _, rgb_enc = cv2.imencode('.jpg', img_bgr)
            _, mask_enc = cv2.imencode('.png', mask_img)
            depth_enc = lz4.frame.compress(depth.tobytes())

            files = {
                'rgb': ('i.jpg', rgb_enc.tobytes(), 'image/jpeg'),
                'mask': ('m.png', mask_enc.tobytes(), 'image/png'),
                'depth': ('d.lz4', depth_enc, 'application/octet-stream')
            }
            data = {'K': K_str}

            try:
                t0 = time.time()
                # 这里的 POST 请求是耗时的，确保 Docker 性能足够好
                res = requests.post(DOCKER_URL, files=files, data=data, timeout=30)
                
                if res.status_code == 200 and 'pose' in res.json():
                    # 物体相对于相机的位姿 (由 Docker 里的 FoundationPose/ICP 计算得出)
                    pose_obj_cam = np.array(res.json()['pose']) # 4x4 矩阵
                    
                    # --- D. 最终坐标计算 ---
                    # 链式法则：基座 -> 末端法兰 -> 相机 -> 物体
                    # T_final = T_base_tool * T_tool_cam * T_cam_obj
                    # 注意：你的 get_tool_base 原来好像是 T_base_tool (从基座看末端)
                    
                    T_final = T_tool_base @ T_cam_tool @ pose_obj_cam
                    
                    tx, ty, tz, rx, ry, rz = matrix_to_six_dof(T_final)
                    
                    dt = (time.time() - t0) * 1000
                    print(f"[{dt:.0f}ms] 最终抓取位姿 (Base系): X={tx:.3f} Y={ty:.3f} Z={tz:.3f}")

                    # 可视化
                    text = f"Base Pose: X:{tx:.2f} Y:{ty:.2f} Z:{tz:.2f}"
                    cv2.putText(img_bgr, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"Docker/计算错误: {e}")

            cv2.imshow("Local View", img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
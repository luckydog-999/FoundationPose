# local_client.py
# 运行在你的本地电脑上

import pyzed.sl as sl
import cv2
import numpy as np
import requests
import lz4.frame
import time
import base64
import logging
# 必须安装: pip install scipy
from scipy.spatial.transform import Rotation as R
import os
import datetime

# --- 配置 ---
SERVER_URL = "https://u474219-8f05-a40ec3b8.bjb1.seetacloud.com:8443"
INIT_URL = f"{SERVER_URL}/init"
PROCESS_URL = f"{SERVER_URL}/process"
# ------------

# 💡 --- 辅助函数 ---

def create_transform_matrix(tx, ty, tz, rx_deg, ry_deg, rz_deg, euler_order='xyz'):
    """
    将 [tx, ty, tz, rx, ry, rz] 转换为 4x4 矩阵
    """
    try:
        # 默认使用 'xyz' 顺序 (Extrinsic)
        r = R.from_euler(euler_order, [rx_deg, ry_deg, rz_deg], degrees=True)
        rotation_matrix = r.as_matrix()
    except ValueError as e:
        logging.error(f"创建旋转矩阵失败: {e}")
        return np.eye(4)
        
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = rotation_matrix
    transform_matrix[0:3, 3] = [tx, ty, tz]
    return transform_matrix

def matrix_to_six_dof(matrix, euler_order='xyz'):
    """
    将 4x4 矩阵转换回 [tx, ty, tz, rx, ry, rz] 格式便于阅读
    """
    tx, ty, tz = matrix[0:3, 3]
    r = R.from_matrix(matrix[0:3, 0:3])
    # 返回角度 (degrees)
    rx, ry, rz = r.as_euler(euler_order, degrees=True)
    return tx, ty, tz, rx, ry, rz

def get_tool_in_base_pose_manual():
    """
    【手动输入区域】
    请输入机械臂示教器上显示的法兰（工具）相对于基座的位姿。
    """
    # ==========================================
    # 👇 请在这里修改你的机械臂实时位姿数值 👇
    # ==========================================
    
    # 平移 (单位: 米)
    input_tx = 0.359   
    input_ty = 0.623
    input_tz = 0.262
    
    # 旋转 (单位: 度) 
    # 如果你的机械臂是 UR/Fanuc 等通用型号，通常对应 xyz 或 zyx 顺序
    input_rx = -168.08
    input_ry = 5.1
    input_rz = -50.06
    
    # 将输入的 6 个数转换为矩阵
    # 注意：如果不确定旋转顺序，大部分工业机器人标准可以用 'xyz' 或 'zyx' 尝试
    T_tool_base = create_transform_matrix(input_tx, input_ty, input_tz, 
                                          input_rx, input_ry, input_rz, 
                                          euler_order='xyz')
    
    return T_tool_base

# ------------------------

def main():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=f"{log_dir}/local_client_{timestamp}.log")
    
    # 1. 初始化 ZED 相机
    logging.info("Initializing ZED camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.sdk_verbose = 0

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        logging.error(f"Failed to open ZED camera: {err}")
        exit()

    # 2. 获取相机内参
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])
    shape = (cam_info.camera_configuration.resolution.height, cam_info.camera_configuration.resolution.width)

    # 3. 初始化服务器
    logging.info("Sending initialization data to server...")
    try:
        requests.post(INIT_URL, json={"K": K.tolist(), "shape": shape}, timeout=10)
    except Exception as e:
        logging.error(f"Server init failed: {e}")
        zed.close()
        return

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    runtime_params.enable_depth = True

    # 💡 --- 定义固定的 T_cam_tool (来自你的 Halcon 标定文件) ---
    # Halcon Type 2 -> 使用 'xyz'
    T_cam_tool = create_transform_matrix(
        tx=-0.06308799, ty=-0.12889982, tz=-0.00412758,
        rx_deg=358.2564, ry_deg=359.5684, rz_deg=358.3827,
        euler_order='xyz'
    )
    logging.info("Cam-in-Tool matrix loaded.")
    
    # 🚩 暂停状态标志位
    is_paused = False
    logging.info(">>> 提示: 按 'p' 键暂停/恢复位姿计算; 按 'q' 键退出 <<<")

    # 4. 实时循环
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            
            # A. 获取图像
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            color_bgra = image_mat.get_data()
            color_bgr = color_bgra[..., :3]

            # 💡 键盘控制逻辑
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                is_paused = not is_paused
                state = "PAUSED" if is_paused else "RUNNING"
                logging.info(f"State changed to: {state}")

            # 如果处于暂停状态，只显示图像，不发送请求，不打印位姿
            if is_paused:
                # 在画面上写字提示暂停
                cv2.putText(color_bgr, "PAUSED - Press 'p' to resume", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Result from Cloud Server", color_bgr)
                time.sleep(0.03) # 省点CPU
                continue

            # --- 以下是正常运行逻辑 ---
            start_time = time.time()
            
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth = depth_mat.get_data()
            depth[np.isnan(depth)] = 0.0
            depth[np.isinf(depth)] = 0.0

            _, rgb_jpg_bytes = cv2.imencode('.jpg', color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            depth_lz4_bytes = lz4.frame.compress(depth.tobytes())

            files = {
                'rgb': ('image.jpg', rgb_jpg_bytes.tobytes(), 'image/jpeg'),
                'depth': ('depth.lz4', depth_lz4_bytes, 'application/octet-stream')
            }
            
            try:
                response = requests.post(PROCESS_URL, files=files, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    if "vis_image" in data:
                        # 解码可视化图
                        vis_bytes = base64.b64decode(data['vis_image'])
                        vis_img = cv2.imdecode(np.frombuffer(vis_bytes, np.uint8), cv2.IMREAD_COLOR)
                        
                        # 1. 获取 T_obj_cam
                        T_obj_cam = np.array(data['pose'])
                        
                        # 2. 获取 T_tool_base (读取你手动填写的6个数生成的矩阵)
                        T_tool_base = get_tool_in_base_pose_manual()
                        
                        # 3. 计算最终 T_obj_base
                        T_obj_base = T_tool_base @ T_cam_tool @ T_obj_cam
                        
                        # 4. 将矩阵转换回 6个数值 方便查看
                        f_tx, f_ty, f_tz, f_rx, f_ry, f_rz = matrix_to_six_dof(T_obj_base)
                        
                        # 5. 格式化输出
                        logging.info("✅ [Object In Base Pose]:")
                        logging.info(f"   平移(m) : X={f_tx:.4f}, Y={f_ty:.4f}, Z={f_tz:.4f}")
                        logging.info(f"   旋转(deg): RX={f_rx:.2f}, RY={f_ry:.2f}, RZ={f_rz:.2f}")
                        logging.info("-" * 40)
                        
                        cv2.imshow("Result from Cloud Server", vis_img)
                    else:
                        logging.warning(f"Server info: {data.get('error', 'Unknown')}")
                
            except Exception as e:
                logging.error(f"Net Error: {e}")
                time.sleep(1)
        
        else:
            time.sleep(0.01)

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
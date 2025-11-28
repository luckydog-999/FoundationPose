# local_main.py (运行在宿主机本地)
# 职责：ZED + YOLO -> 发送给 Docker -> 接收 Pose -> 坐标变换 -> 显示

import pyzed.sl as sl
import cv2
import numpy as np
import requests
import lz4.frame
import time
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# --- 配置 ---
DOCKER_URL = "http://127.0.0.1:6006/process" # 本地 Docker 地址
YOLO_PATH = "best.pt" # 你的 YOLO 模型路径

# --- 坐标变换辅助函数 (保持不变) ---
def create_transform_matrix(tx, ty, tz, rx, ry, rz):
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    m = np.eye(4); m[:3,:3]=r.as_matrix(); m[:3,3]=[tx,ty,tz]
    return m

def matrix_to_six_dof(m):
    tx, ty, tz = m[:3, 3]
    rx, ry, rz = R.from_matrix(m[:3, :3]).as_euler('xyz', degrees=True)
    return tx, ty, tz, rx, ry, rz

# 机械臂位姿 (手动填入)
def get_tool_base():
    # 👇 根据实际情况修改
    return create_transform_matrix(0.599, 0.823, 0.225, -144.6, 15.1, -51.2)

def main():
    # 1. 本地加载 YOLO
    print("正在本地加载 YOLO...")
    model = YOLO(YOLO_PATH)
    
    # 2. 初始化 ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL # 本地显卡跑这个没压力
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED 打开失败")
        return

    # 获取内参
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    # 将 K 矩阵格式化为字符串方便传输
    K_str = f"{calib.fx},0,{calib.cx},0,{calib.fy},{calib.cy},0,0,1"
    
    # 手眼标定矩阵
    T_cam_tool = create_transform_matrix(-0.063, -0.128, -0.004, 358.2, 359.5, 358.3)

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    print(">>> 开始运行... 按 'q' 退出 <<<")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            
            img_bgr = image_mat.get_data()[..., :3]
            depth = depth_mat.get_data()
            depth[np.isnan(depth)] = 0; depth[np.isinf(depth)] = 0

            # --- A. 本地 YOLO 推理 ---
            results = model(img_bgr, conf=0.85, verbose=False)
            
            mask_img = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            has_detection = False
            
            if results[0].masks is not None and len(results[0].masks.data) > 0:
                # 获取掩码并调整大小
                m = results[0].masks.data[0].cpu().numpy()
                if m.shape != img_bgr.shape[:2]:
                    m = cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_img = (m * 255).astype(np.uint8)
                has_detection = True
            
            # 如果没检测到，直接显示原图跳过
            if not has_detection:
                cv2.imshow("Local View", img_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # --- B. 发送数据到 Docker ---
            # 编码图像和掩码以减少传输体积
            _, rgb_enc = cv2.imencode('.jpg', img_bgr)
            _, mask_enc = cv2.imencode('.png', mask_img) # Mask 用 png 无损
            depth_enc = lz4.frame.compress(depth.tobytes())

            files = {
                'rgb': ('i.jpg', rgb_enc.tobytes(), 'image/jpeg'),
                'mask': ('m.png', mask_enc.tobytes(), 'image/png'),
                'depth': ('d.lz4', depth_enc, 'application/octet-stream')
            }
            data = {'K': K_str}

            try:
                # 发送 POST 请求给 localhost
                t0 = time.time()
                res = requests.post(DOCKER_URL, files=files, data=data, timeout=30)
                
                if res.status_code == 200 and 'pose' in res.json():
                    # --- C. 接收 Pose 并处理 ---
                    pose = np.array(res.json()['pose'])
                    
                    # 坐标计算
                    T_final = get_tool_base() @ T_cam_tool @ pose
                    tx, ty, tz, rx, ry, rz = matrix_to_six_dof(T_final)
                    
                    dt = (time.time() - t0) * 1000
                    print(f"[{dt:.0f}ms] Pose: X={tx:.3f} Y={ty:.3f} Z={tz:.3f}")

                    # 简单的可视化 (画轴)
                    # 这里可以简单画个框或者直接打印
                    cv2.putText(img_bgr, f"X:{tx:.3f} Y:{ty:.3f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            except Exception as e:
                print(f"Docker 通信错误: {e}")

            cv2.imshow("Local View", img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
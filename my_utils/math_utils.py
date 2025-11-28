# 文件路径: my_utils/math_utils.py
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def create_transform_matrix(tx, ty, tz, rx_deg, ry_deg, rz_deg, euler_order='xyz'):
    try:
        r = R.from_euler(euler_order, [rx_deg, ry_deg, rz_deg], degrees=True)
        return np.vstack([np.hstack([r.as_matrix(), [[tx], [ty], [tz]]]), [0, 0, 0, 1]])
    except:
        return np.eye(4)

def matrix_to_six_dof(matrix, euler_order='xyz'):
    """将4x4矩阵转换为 (x, y, z, rx, ry, rz)"""
    tx, ty, tz = matrix[0:3, 3]
    r = R.from_matrix(matrix[0:3, 0:3])
    rx, ry, rz = r.as_euler(euler_order, degrees=True)
    return tx, ty, tz, rx, ry, rz

def get_tool_in_base_pose_manual():
    """你的机械臂当前位姿 (Base -> Tool)"""
    # 这里填你机械臂示教器上显示的数值
    return create_transform_matrix(0.359, 0.623, 0.262, -168.08, 5.1, -50.06)

def draw_axis(img, pose, K, axis_len=0.05):
    """
    在图像上画 XYZ 坐标轴
    pose: Object -> Camera 的 4x4 位姿矩阵
    K: 相机内参
    axis_len: 坐标轴长度 (单位: 米)
    """
    try:
        # 定义原点和XYZ轴末端点 (Object系)
        points_3d = np.float32([
            [0, 0, 0],          # 原点
            [axis_len, 0, 0],   # X轴点
            [0, axis_len, 0],   # Y轴点
            [0, 0, axis_len]    # Z轴点
        ]).reshape(-1, 3)

        # 1. 变换到相机坐标系 (Camera Frame)
        # R * P + t
        R = pose[:3, :3]
        t = pose[:3, 3].reshape(3)
        points_cam = (R @ points_3d.T).T + t

        # 2. 投影到像素坐标系 (Pixel Frame)
        # u = fx * x/z + cx
        points_2d = (K @ points_cam.T).T
        points_2d[:, :2] /= points_2d[:, 2:3] # 除以深度 Z
        pts = points_2d[:, :2].astype(int)

        origin = tuple(pts[0])
        # 画线：OpenCV是 (B, G, R) -> X轴用红(0,0,255), Y轴用绿(0,255,0), Z轴用蓝(255,0,0)
        img = cv2.line(img, origin, tuple(pts[1]), (0, 0, 255), 3) # X - Red
        img = cv2.line(img, origin, tuple(pts[2]), (0, 255, 0), 3) # Y - Green
        img = cv2.line(img, origin, tuple(pts[3]), (255, 0, 0), 3) # Z - Blue
        return img
    except Exception as e:
        print(f"Draw Axis Error: {e}")
        return img
import socket
import struct
import json
import numpy as np
import cv2
import lz4.frame
import time

# ==========================================
# 1. 严格复刻 socket_utils.py 的内容
# ==========================================
def send_msg(sock, data):
    """发送数据，带4字节长度头"""
    msg = struct.pack('>I', len(data)) + data
    sock.sendall(msg)

def recvall(sock, n):
    """你的原始实现: 使用 bytearray"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_msg(sock):
    """接收数据，先读4字节长度，再读内容"""
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
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
HOST = '0.0.0.0'
PORT = 6006

def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    
    try:
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)
        print(f"🚀 [严格模式] Server listening on {PORT}...")
    except Exception as e:
        print(f"Bind Error: {e}")
        return

    while True:
        print("等待客户端连接...")
        conn, addr = server_sock.accept()
        print(f"Connected by {addr}")
        
        try:
            # --- 1. 握手阶段 (使用 socket_utils 协议: 4字节头 + JSON) ---
            init_data = recv_json(conn)
            if init_data and 'K' in init_data:
                g_shape = tuple(init_data['shape'])
                print(f"Client Initialized. Shape: {g_shape}")
                send_json(conn, {"status": "ok"})
            else:
                print("Handshake failed.")
                conn.close()
                continue

            # --- 2. 传输阶段 (使用 main_socket 协议: 12字节头 + Body) ---
            prev_time = time.time()
            
            while True:
                # 注意：这里不能用 recv_msg，必须手动读 12 字节，这才是你的真实逻辑
                header_data = recvall(conn, 12)
                if not header_data: break
                
                rgb_len, depth_len, type_len = struct.unpack('>III', header_data)
                
                # 读取三段数据
                type_bytes = recvall(conn, type_len)
                rgb_bytes = recvall(conn, rgb_len)
                depth_bytes = recvall(conn, depth_len)
                
                if not rgb_bytes or not depth_bytes: break

                # --- 模拟处理 (仅解压，0推理) ---
                # 解压 RGB
                # np.frombuffer 支持 bytearray，不需要转换
                img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                # 解压 Depth
                depth_raw = lz4.frame.decompress(depth_bytes)
                depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(g_shape)

                # --- 计算 FPS ---
                curr_time = time.time()
                elapsed = curr_time - prev_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                prev_time = curr_time

                # --- 回复 (使用 socket_utils 协议: 4字节头 + JSON) ---
                send_json(conn, {"found": False})

                # --- 显示 ---
                info = f"FPS: {fps:.1f} | Type: {type_bytes.decode()} | RGB Size: {len(rgb_bytes)}"
                print(f"\r{info}", end="")
                
                cv2.putText(img_bgr, f"Server FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 如果是无头服务器请注释
                cv2.imshow("Server View", img_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        except Exception as e:
            print(f"\nConnection Error: {e}")
        finally:
            conn.close()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
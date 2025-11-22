import socket

# 配置
HOST_IP = '192.168.1.10'  # 本机 IP (接收端)
PORT = 8888               # 任意端口，只要两边一致且未被占用

# 创建 TCP Socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # 绑定 IP 和端口
    server_socket.bind((HOST_IP, PORT))
    server_socket.listen(1)
    print(f"正在监听 {HOST_IP}:{PORT}，等待发送端连接...")

    # 接受连接
    conn, addr = server_socket.accept()
    print(f"连接成功！对方地址: {addr}")

    with conn:
        while True:
            data = conn.recv(1024) # 每次接收 1024 字节
            if not data:
                print("对方断开了连接")
                break
            print(f"收到消息: {data.decode('utf-8')}")

except Exception as e:
    print(f"发生错误: {e}")
finally:
    server_socket.close()
# 测试与远程 Qt 程序通信的简单客户端

import socket
import json

# 替换为您 Qt 程序所在电脑的 IP 地址
HOST = '192.168.1.20' 
PORT = 6666

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
print("Connected to Robot Control PC")

while True:
    data = s.recv(1024)
    if not data:
        break
    
    try:
        # 解析 JSON
        json_str = data.decode('utf-8').strip()
        msg = json.loads(json_str)
        
        if msg.get("command") == "robot_flange_pose":
            pose = msg["pose"]
            print(f"收到法兰位姿: X={pose[0]}, Y={pose[1]}, Z={pose[2]}...")
            # 在这里进行您的手眼标定复合运算...
            
    except Exception as e:
        print("Data error:", e)
# socket_utils.py
import struct
import json
import socket

def send_msg(sock, data):
    """
    发送数据，带4字节长度头
    """
    # 1. 加上 4字节 unsigned int 表示长度
    msg = struct.pack('>I', len(data)) + data
    sock.sendall(msg)

def recv_msg(sock):
    """
    接收数据，先读4字节长度，再读内容
    """
    # 1. 读 4 字节长度
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # 2. 读内容
    return recvall(sock, msglen)

def recvall(sock, n):
    """
    循环读取直到读满 n 个字节
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def send_json(sock, data_dict):
    json_bytes = json.dumps(data_dict).encode('utf-8')
    send_msg(sock, json_bytes)

def recv_json(sock):
    data = recv_msg(sock)
    if data is None: return None
    return json.loads(data.decode('utf-8'))
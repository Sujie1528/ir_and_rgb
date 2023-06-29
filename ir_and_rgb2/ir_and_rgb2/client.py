# This is a TCP client that receives a "start" message from the server and then print start, then it receives a "stop" message from the server and then print stop

import socket

HOST = '192.168.0.118' # The server's hostname or IP address
PORT = 65432 # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST,PORT))
    while True:
        data = s.recv(1024)
        if data == b'start':
            print('start')
        elif data == b'end':
            print('end')

#!/usr/bin/env python3

import socket

HOST = '70.106.17.101'  # The server's hostname or IP address
PORT = 1024        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    input("go? ")
    s.sendall(b'Hello, world')
    data = s.recv(1024)

print('Received', repr(data))

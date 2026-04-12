import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 1902))
print("Waiting for data on port 1902...")
while True:
    data, addr = sock.recvfrom(1024)
    print(f"Received: {data.decode()} from {addr}")
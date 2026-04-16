import socket, threading

def pipe(a, b):
    while True:
        try: b.sendall(a.recv(4096))
        except: break

def handle(c):
    s = socket.socket()
    s.connect(('127.0.0.1', 7860))
    threading.Thread(target=pipe, args=(c, s), daemon=True).start()
    threading.Thread(target=pipe, args=(s, c), daemon=True).start()

srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(('0.0.0.0', 8888))
srv.listen(5)
print('relay ready on 0.0.0.0:8888')
while True:
    c, _ = srv.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
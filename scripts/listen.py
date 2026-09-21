"""Tiny TCP listener for firewall-directionality probing: bind <port>, accept for
<seconds>, so a /dev/tcp connect from the other node distinguishes 'firewall drop'
(connect times out) from 'reachable' (connect succeeds).
    python3 scripts/listen.py <port> [seconds]
"""
import socket
import sys
import time

port = int(sys.argv[1])
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", port))
s.listen(16)
s.settimeout(1.0)
print(f"[listen] on 0.0.0.0:{port} for {dur:.0f}s", flush=True)
end = time.time() + dur
while time.time() < end:
    try:
        c, a = s.accept()
        print(f"[listen] accepted from {a}", flush=True)
        c.close()
    except socket.timeout:
        pass

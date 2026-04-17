from ngllib.utils.Communication import *
import time

print(f"[client] Script started at {time.strftime('%H:%M:%S')}", flush=True)

print("[client] Creating SocketProtocol (connecting to 127.0.0.1:7860)...", flush=True)
medium = SocketProtocol(host="127.0.0.1", port=7860, is_server=False, timeout=600)
print("[client] SocketProtocol created, will connect on first read/write.", flush=True)

client = NGLClient(protocol=medium)

action_vector = [
    0, 0, 0,  # left, right, double click booleans
    100, 100,  # x, y
    0, 0, 0,  # no modifier keys
    1,  # no JSON change
    10, 0, 0,  # position change
    0,  # cross-section scaling
    0.2, 0, 0,  # orientation change in Euler angles, which is better for a model to learn or a human to understand
    2000  # projection scaling (log-scale in neuroglancer)
    ]

print("[client] Waiting for initial observation (will retry until tunnel is up)...", flush=True)
obs = client.get_initial()
print(f"[client] Got initial observation at {time.strftime('%H:%M:%S')}: type={type(obs)}, len={len(obs) if isinstance(obs, (list,tuple)) else 'N/A'}", flush=True)

for i in range(100):
    print(f"[client] Sending action {i+1}/100...", flush=True)
    obs = client.send_actions(action_vector)
    print(f"[client] Got observation {i+1} at {time.strftime('%H:%M:%S')}", flush=True)

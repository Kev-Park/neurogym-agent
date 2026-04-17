from ngllib.utils.Communication import *
from ngllib import Environment
import time

print(f"[host] Script started at {time.strftime('%H:%M:%S')}", flush=True)

# # Initialize options for environmental interaction, including state return types
options = {
        'euler_angles': True,
        'resize': False,
        'add_mouse': False,
        'fast': True,
        'image_path': None
}

def custom_reward(state, action, prev_state):
    return 1, False

print("[host] Creating Environment...", flush=True)
env = Environment(headless=False, config_path="config.json", verbose=False, reward_function=custom_reward)
print("[host] Environment created.", flush=True)

print("[host] Creating SocketProtocol (listener on 127.0.0.1:7861)...", flush=True)
medium = SocketProtocol(host="127.0.0.1", port=7861, is_server=True, timeout=600)
print("[host] SocketProtocol ready, waiting for client connection...", flush=True)

server = NGLServer(protocol=medium, environment=env)

print("[host] Starting session (will block until client connects)...", flush=True)
server.start_session(**options) # can pass in a start_url too
print(f"[host] Session started, initial observation sent at {time.strftime('%H:%M:%S')}", flush=True)

for i in range(100):
    print(f"[host] Waiting for action {i+1}/100...", flush=True)
    server.process_actions()
    print(f"[host] Action {i+1} processed at {time.strftime('%H:%M:%S')}", flush=True)
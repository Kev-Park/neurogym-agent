from ngllib.utils.Communication import *
from ngllib import Environment

options = {
        'euler_angles': True,
        'resize': False,
        'add_mouse': False,
        'fast': True,
        'image_path': None,
        'left_pane': False
}

def custom_reward(state, action, prev_state):
    return 1, False

env = Environment(headless=False, config_path="config.json", verbose=False, reward_function=custom_reward)
medium = SocketProtocol(host="127.0.0.1", port=7861, is_server=True, timeout=600)
server = NGLServer(protocol=medium, environment=env)

server.start_session(**options)

for i in range(100):
    server.process_actions()
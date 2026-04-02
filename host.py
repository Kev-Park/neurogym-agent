from ngllib.utils.Communication import *
from ngllib import Environment

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

env = Environment(headless=False, config_path="config.json", verbose=False, reward_function=custom_reward)

medium = FilesystemProtocol(action_file_path="K:/Coding Projects/Labs/Seung Lab/Tests/proxy/actions", observation_file_path="K:/Coding Projects/Labs/Seung Lab/Tests/proxy/observations", timeout=99999)

server = NGLServer(protocol=medium, environment=env)

server.start_session(**options)

for i in range(100):
    server.process_actions()
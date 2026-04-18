from ngllib.utils.Communication import *

medium = SocketProtocol(host="127.0.0.1", port=7860, is_server=False, timeout=600)
client = NGLClient(protocol=medium)

action_vector = [
    0, 0, 0,  # left, right, double click booleans
    100, 100,  # x, y
    0, 0, 0,  # no modifier keys
    1,  # no JSON change
    10, 0, 0,  # position change
    0,  # cross-section scaling
    0.2, 0, 0,  # orientation change in Euler angles
    2000  # projection scaling (log-scale in neuroglancer)
    ]

obs = client.get_initial()

for i in range(100):
    obs = client.send_actions(action_vector)

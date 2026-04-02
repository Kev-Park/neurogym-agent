from ngllib.utils.Communication import *

medium = FilesystemProtocol(action_file_path="/scratch/gpfs/TSILVER/kp0374/neurogym-agent/proxy/actions", observation_file_path="/scratch/gpfs/TSILVER/kp0374/neurogym-agent/proxy/observations", timeout=9999999)

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

for i in range(100):
    client.send_actions(action_vector)

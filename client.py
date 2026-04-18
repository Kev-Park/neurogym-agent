from ngllib.utils.Communication import *
import random
import urllib.parse
import json

# Load available segment IDs
with open("segment_ids.txt", "r") as f:
    segment_ids = [line.strip() for line in f if line.strip()]

# Base Neuroglancer state — segments field will be swapped per reset
BASE_STATE = {
    "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
    "position": [148684.421875, 57005.6640625, 111.5],
    "crossSectionScale": 2.0339912586467497,
    "projectionOrientation": [-0.4934331774711609, 0.7386592030525208, -0.27805963158607483, 0.365498423576355],
    "projectionScale": 13976.00585680798,
    "layers": [
        {
            "type": "image",
            "source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
            "tab": "source",
            "name": "Maryland (USA)-image"
        },
        {
            "type": "segmentation",
            "source": "precomputed://gs://flywire_v141_m783",
            "tab": "source",
            "segments": [],
            "name": "flywire_v141_m783"
        }
    ],
    "showDefaultAnnotations": False,
    "selectedLayer": {"size": 350, "layer": "flywire_v141_m783"},
    "layout": "xy-3d"
}

def make_url(segment_id: str) -> str:
    state = json.loads(json.dumps(BASE_STATE))
    state["layers"][1]["segments"] = [segment_id]
    return "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(json.dumps(state))

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

for i in range(5):
    for j in range(50):
        obs = client.send_actions(action_vector)
    seg_id = random.choice(segment_ids)
    url = make_url(seg_id)
    obs = client.send_reset(url=url)
    print(f"environment reset {i} to segment {seg_id}")
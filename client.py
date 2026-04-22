from ngllib.utils.Communication import *
import csv
import math
import random
import urllib.parse
import json

csv.field_size_limit(2**31 - 1)

# Load segment positions: {root_id: [[x,y,z], ...]}
segment_data = {}
with open("segment_positions.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        rid = row[0]
        coords = []
        for pos in row[1].split("|"):
            x, y, z = pos.split(";")
            coords.append([float(x), float(y), float(z)])
        segment_data[rid] = coords

segment_ids = list(segment_data.keys())

# Base Neuroglancer state
BASE_STATE = {
    "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
    "position": [0, 0, 0],
    "crossSectionScale": 2.0,
    "projectionOrientation": [0, 0, 0, 1],
    "projectionScale": 14000,
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


def random_quaternion():
    """Generate a uniformly random unit quaternion [x, y, z, w]."""
    u1, u2, u3 = random.random(), random.random(), random.random()
    q = [
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ]
    return q


def make_url(segment_id: str) -> str:
    positions = segment_data[segment_id]
    pos = random.choice(positions)
    orientation = random_quaternion()

    state = json.loads(json.dumps(BASE_STATE))
    state["layers"][1]["segments"] = [segment_id]
    state["position"] = pos
    state["projectionOrientation"] = orientation
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


for i in range(2):
    seg_id = random.choice(segment_ids)
    url = make_url(seg_id)
    obs = client.send_reset(url=url)
    print(f"environment reset {i} to segment {seg_id}")


    for j in range(50):
        obs = client.send_actions(action_vector)
    
    
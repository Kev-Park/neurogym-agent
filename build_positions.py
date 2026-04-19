"""Build segment_positions.csv from Codex coordinates.csv and segment_ids.txt.

Filters to only neurons in segment_ids.txt, converts nm to Neuroglancer
viewer coordinates (4nm/4nm/40nm resolution), groups multiple positions per neuron.

Output format: root_id,x1;y1;z1|x2;y2;z2|...
"""

import csv
import re
import sys

csv.field_size_limit(2**31 - 1)

RESOLUTION = [4, 4, 40]

# Load valid segment IDs
with open("segment_ids.txt") as f:
    valid_ids = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(valid_ids)} valid segment IDs")

# Parse coordinates.csv — position is like "[352484 175164 229040]" in nm
positions = {}
with open("coordinates.csv") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        rid = row[0]
        if rid not in valid_ids:
            continue
        # Parse "[352484 175164 229040]"
        nums = re.findall(r'[\d.]+', row[1])
        if len(nums) != 3:
            continue
        x = float(nums[0]) / RESOLUTION[0]
        y = float(nums[1]) / RESOLUTION[1]
        z = float(nums[2]) / RESOLUTION[2]
        positions.setdefault(rid, []).append([x, y, z])

print(f"Found positions for {len(positions)} neurons")

with open("segment_positions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["root_id", "positions"])
    for rid, coords in positions.items():
        pos_str = "|".join(f"{c[0]:.1f};{c[1]:.1f};{c[2]:.1f}" for c in coords)
        writer.writerow([rid, pos_str])

print(f"Wrote segment_positions.csv ({len(positions)} rows)")

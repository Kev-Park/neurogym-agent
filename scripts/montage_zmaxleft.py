"""Assemble annotated rollout videos into a contact sheet and tiled MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, required=True)
    ap.add_argument("--title", default="zmax-left holdout rollouts")
    args = ap.parse_args()
    rows = json.loads((args.input_dir / "manifest.json").read_text())
    if len(rows) != 6:
        raise ValueError(f"Expected six complete audit episodes, got {len(rows)}")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)
    readers = []
    try:
        for row in rows:
            readers.append(imageio.get_reader(args.input_dir / row["file"]))
        counts = [reader.count_frames() for reader in readers]
        first = readers[0].get_data(0)
        width = 600
        height = round(first.shape[0] * width / first.shape[1])
        height += height % 2
        label_h = 30
        title_h = 40

        def tile(reader, row, step):
            frame = Image.fromarray(reader.get_data(step)).resize((width, height))
            result = Image.new("RGB", (width, height + label_h), (18, 18, 22))
            result.paste(frame, (0, label_h))
            ImageDraw.Draw(result).text(
                (8, 5), f"idx {row['idx']} | dz {row['dz']:+.0f} | "
                f"headroom {row['headroom']:.0f}", font=font, fill="white")
            return result

        sheet = Image.new("RGB", (4 * width, title_h + 6 * (height + label_h)),
                          (18, 18, 22))
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 10), args.title + " | steps 0, 125, 250, final",
                  font=font, fill="white")
        for r, (reader, row, count) in enumerate(zip(readers, rows, counts)):
            for c, step in enumerate((0, 125, 250, row["steps"])):
                sheet.paste(tile(reader, row, min(step, count - 1)),
                            (c * width, title_h + r * (height + label_h)))
        sheet.save(str(args.output_prefix) + ".png")

        fps = readers[0].get_meta_data()["fps"]
        with imageio.get_writer(str(args.output_prefix) + ".mp4", fps=fps,
                                codec="libx264", macro_block_size=2) as writer:
            for step in range(max(counts)):
                canvas = Image.new("RGB", (2 * width, title_h + 3 * (height + label_h)),
                                   (18, 18, 22))
                ImageDraw.Draw(canvas).text((8, 10), args.title, font=font, fill="white")
                for i, (reader, row, count) in enumerate(zip(readers, rows, counts)):
                    canvas.paste(tile(reader, row, min(step, count - 1)),
                                 ((i % 2) * width, title_h + (i // 2) * (height + label_h)))
                writer.append_data(np.asarray(canvas))
        print(f"MONTAGE_OK {args.output_prefix}", flush=True)
    finally:
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    main()

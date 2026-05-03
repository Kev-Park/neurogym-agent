from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.action_translator import ActionSpec, cell_to_pixel
from envs.browser_manager import BrowserManager
from envs.dino_vec_wrapper import DinoVecWrapper
from envs.ngl_gym_env import NGLGymEnv, _load_segment_positions
from envs.reward import RewardConfig

_FONT: ImageFont.ImageFont | None = None


def _get_font(size: int = 20) -> ImageFont.ImageFont:
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.load_default(size=size)
        except TypeError:
            _FONT = ImageFont.load_default()
    return _FONT


def annotate_frame(
    frame: np.ndarray,
    lines: list[str],
    click_info: dict | None = None,
) -> np.ndarray:
    """Draw grid overlay + click highlight, then text banner.

    click_info keys: spec (ActionSpec), cell (int), click_type (int 0/1)

    Frame coords = screen coords minus pane origin, so the grid runs from
    (0,0) to (frame_w, frame_h) directly — no extra offset needed.
    """
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov = ImageDraw.Draw(overlay)

    if click_info is not None:
        spec: ActionSpec = click_info["spec"]
        cell: int = click_info["cell"]
        right: bool = click_info["click_type"] == 1

        cell_w = img.width / spec.grid_cols
        cell_h = img.height / spec.grid_rows
        row = cell // spec.grid_cols
        col = cell % spec.grid_cols

        # Full discretization grid (faint white lines)
        for r in range(spec.grid_rows + 1):
            y = r * cell_h
            ov.line([(0, y), (img.width, y)], fill=(255, 255, 255, 50), width=1)
        for c in range(spec.grid_cols + 1):
            x = c * cell_w
            ov.line([(x, 0), (x, img.height)], fill=(255, 255, 255, 50), width=1)

        # Clicked cell highlight
        x0, y0 = col * cell_w, row * cell_h
        x1, y1 = x0 + cell_w, y0 + cell_h
        cell_color = (0, 100, 255, 70) if right else (255, 80, 0, 70)
        border_color = (0, 180, 255, 220) if right else (255, 60, 0, 220)
        ov.rectangle([x0, y0, x1, y1], fill=cell_color, outline=border_color)

        # Dot at cell centre
        cx, cy = x0 + cell_w / 2, y0 + cell_h / 2
        dot_r = max(4, min(cell_w, cell_h) * 0.25)
        dot_color = (0, 180, 255, 230) if right else (255, 40, 40, 230)
        ov.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=dot_color)

    img = Image.alpha_composite(img, overlay).convert("RGB")

    # Text banner
    draw = ImageDraw.Draw(img)
    font = _get_font(20)
    pad = 6
    line_h = 24
    box_h = pad * 2 + line_h * len(lines)
    box_w = max(draw.textlength(l, font=font) for l in lines) + pad * 2
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 180))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill=(255, 255, 0), font=font)

    return np.array(img)


def describe_action(md_action, spec: ActionSpec) -> str:
    cell, click_type, d_ex, d_ey, d_ez = (int(v) for v in md_action)
    x, y = cell_to_pixel(cell, spec)
    half = spec.rotation_bins_per_axis // 2
    dex = (d_ex - half) * spec.rotation_step_rad
    dey = (d_ey - half) * spec.rotation_step_rad
    dez = (d_ez - half) * spec.rotation_step_rad
    click = "right_click" if click_type == 1 else "left_click"
    rot = f"rot=({dex:+.3f},{dey:+.3f},{dez:+.3f})" if (dex or dey or dez) else "no_rot"
    return f"{click} ({x:.0f},{y:.0f}) {rot}"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(
    cfg: dict,
    segment_data: dict,
    segment_ids: list,
    browser_manager: BrowserManager,
) -> tuple[DinoVecWrapper, ActionSpec]:
    env_cfg = cfg["env"]
    obs_cfg = cfg["obs"]
    action_spec = ActionSpec(
        grid_rows=env_cfg["click_grid_rows"],
        grid_cols=env_cfg["click_grid_cols"],
        pane_x0=env_cfg["pane_3d_bounds"][0],
        pane_y0=env_cfg["pane_3d_bounds"][1],
        pane_x1=env_cfg["pane_3d_bounds"][2],
        pane_y1=env_cfg["pane_3d_bounds"][3],
        rotation_bins_per_axis=env_cfg["rotation_bins_per_axis"],
        rotation_step_rad=env_cfg["rotation_step_rad"],
    )
    reward_cfg = RewardConfig(
        z_tolerance=env_cfg["z_tolerance"],
        success=env_cfg["reward_success"],
        noop_penalty=env_cfg["reward_noop_penalty"],
        noop_position_eps=env_cfg["noop_position_eps"],
        z_shaping_coef=env_cfg.get("z_shaping_coef", 0.001),
    )

    def _make():
        return NGLGymEnv(
            neurogym_config_path=env_cfg["neurogym_config_path"],
            segment_data=segment_data,
            segment_ids=segment_ids,
            action_spec=action_spec,
            reward_cfg=reward_cfg,
            browser_manager=browser_manager,
            max_episode_steps=env_cfg["max_episode_steps"],
            reset_rotation_perturb_rad=env_cfg["reset_rotation_perturb_rad"],
            reset_zoom_perturb_frac=env_cfg["reset_zoom_perturb_frac"],
            headless=env_cfg.get("headless", True),
        )

    venv = DummyVecEnv([_make])
    return DinoVecWrapper(
        venv,
        repo=obs_cfg["dino_repo"],
        model_name=obs_cfg["dino_model"],
        input_size=obs_cfg["dino_input_size"],
    ), action_spec


def find_latest_checkpoint(folder: Path) -> Path:
    final = folder / "final.zip"
    if final.exists():
        return final

    candidates = list(folder.glob("ppo_ngl_*_steps.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {folder}. "
            "Expected 'final.zip' or 'ppo_ngl_*_steps.zip' files."
        )

    def _step_count(p: Path) -> int:
        m = re.search(r"ppo_ngl_(\d+)_steps", p.stem)
        return int(m.group(1)) if m else -1

    return max(candidates, key=_step_count)


def main():
    parser = argparse.ArgumentParser(description="Run inference rollouts from a checkpoint folder and save videos.")
    parser.add_argument("checkpoint_folder", type=str, help="Path to a run's checkpoint directory.")
    parser.add_argument("--rollouts", type=int, default=10, help="Number of rollouts to run.")
    parser.add_argument("--max_rollout_length", type=int, default=300, help="Max steps per rollout.")
    parser.add_argument("--config", type=str, default=str(_THIS_DIR / "config" / "default.yaml"))
    parser.add_argument(
        "--segment_positions",
        type=str,
        default=str((_THIS_DIR / "../../segment_positions.parquet").resolve()),
        help="Path to segment_positions.parquet.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fps", type=int, default=2, help="Video frames per second (default 2 for readability).")
    args = parser.parse_args()

    checkpoint_folder = Path(args.checkpoint_folder).resolve()
    checkpoint = find_latest_checkpoint(checkpoint_folder)
    print(f"Using checkpoint: {checkpoint}")

    cfg = load_config(args.config)
    cfg["env"]["max_episode_steps"] = args.max_rollout_length

    segment_data, segment_ids = _load_segment_positions(args.segment_positions)
    model = PPO.load(str(checkpoint), device=cfg["train"]["device"])

    browser_manager = BrowserManager(
        headless=cfg["env"].get("headless", True),
        extra_args=cfg["env"].get("chrome_args", []),
    )
    try:
        env, action_spec = build_env(cfg, segment_data, segment_ids, browser_manager)

        video_dir = checkpoint_folder / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        all_successes: list[bool] = []
        all_returns: list[float] = []
        all_steps: list[int] = []

        try:
            for ep in range(args.rollouts):
                obs = env.reset()
                seg_id = env.get_attr("_last_seg_id")[0]
                raw_frames: list[np.ndarray] = [env.get_attr("_last_image")[0]]
                frame_labels: list[list[str]] = [["[reset]"]]
                frame_clicks: list[dict | None] = [None]
                total_reward = 0.0
                steps = 0
                done = False
                info: dict = {}

                while not done:
                    action, _ = model.predict(obs, deterministic=args.deterministic)
                    obs, rewards, dones, infos = env.step(action)
                    reward = float(rewards[0])
                    total_reward += reward
                    info = infos[0]
                    steps += 1
                    done = bool(dones[0])

                    cell, click_type = int(action[0][0]), int(action[0][1])
                    raw_frames.append(env.get_attr("_last_image")[0])
                    frame_labels.append([
                        f"step {steps:03d}  r={reward:+.3f}  ret={total_reward:.3f}",
                        describe_action(action[0], action_spec),
                        f"z={info.get('z_now', float('nan')):.2f}",
                    ])
                    frame_clicks.append({"spec": action_spec, "cell": cell, "click_type": click_type})

                success = bool(info.get("episode_success", False))
                all_successes.append(success)
                all_returns.append(total_reward)
                all_steps.append(steps)

                outcome = "success" if success else "fail"
                video_path = video_dir / f"rollout_{ep:03d}_{seg_id}_{outcome}.mp4"
                with imageio.get_writer(str(video_path), fps=args.fps, macro_block_size=1) as writer:
                    for frame, lines, click_info in zip(raw_frames, frame_labels, frame_clicks):
                        writer.append_data(annotate_frame(frame, lines, click_info))

                print(
                    f"ep {ep:03d} seg={seg_id} success={success} "
                    f"return={total_reward:.3f} steps={steps} "
                    f"z_now={info.get('z_now', float('nan')):.2f} -> {video_path.name}"
                )
        finally:
            env.close()
    finally:
        browser_manager.close()

    print("\n=== aggregate ===")
    print(f"episodes:       {len(all_successes)}")
    print(f"success rate:   {np.mean(all_successes):.3f}")
    print(f"avg return:     {np.mean(all_returns):.3f}")
    print(f"avg steps:      {np.mean(all_steps):.1f}")


if __name__ == "__main__":
    main()

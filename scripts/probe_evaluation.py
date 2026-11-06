"""GPU-job integration check for paired-config inference and shared evaluation."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np

from ngllib_agent.evaluate import main as evaluate
from ngllib_agent.evaluation import checkpoint_config
from ngllib_agent.providers import FlywireSkeletonProvider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = checkpoint_config(args.checkpoint)
    provider = FlywireSkeletonProvider(cfg["env"]["parquet_path"])
    state, task_info = provider(np.random.default_rng(784), {})
    starts = out / "starts.json"
    starts.write_text(json.dumps([dict(id="frozen-0", state=state, task_info=task_info)]))
    common = ["--state-pkl", args.checkpoint, "--starts", str(starts), "--max-steps", "8"]
    evaluate(common + ["--backend", "simulator", "--video-dir", str(out / "video"),
                       "--output", str(out / "simulator.json")])
    evaluate(common + ["--backend", "chrome", "--output", str(out / "chrome.json")])
    evaluate(["--compare", str(out / "simulator.json"), str(out / "chrome.json"),
              "--allow-backend-difference", "--output", str(out / "comparison.json")])
    assert list((out / "video").glob("*.mp4"))
    for name in ("eval_d0.py", "eval_zmaxleft.py"):
        namespace = runpy.run_path(str(Path(__file__).with_name(name)))
        assert namespace["StatePklPolicy"].__module__ == "ngllib_agent.evaluation"
    print("PAIRED-INFERENCE-EVALUATION-PASS", flush=True)


if __name__ == "__main__":
    main()

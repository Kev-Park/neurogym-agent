"""Offline precomputation and content-verified reset-data manifests."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from .distributed.checkpoint import atomic_json


@lru_cache(maxsize=64)
def _file_digest(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _file_digest(str(path), stat.st_size, stat.st_mtime_ns)


def write_manifest(output, *, dataset, coordinates, precomputation, artifacts):
    """Bind existing materialized artifacts; never extract data implicitly."""
    output = Path(output).resolve()
    if not dataset.get("id") or not coordinates.get("frame") or not coordinates.get("units"):
        raise ValueError("Manifest needs dataset.id and coordinates.frame/units")
    records = {}
    for name, source in artifacts.items():
        path = Path(source).resolve()
        records[name] = {"path": os.path.relpath(path, output.parent), "sha256": file_digest(path)}
    if not records:
        raise ValueError("Manifest needs at least one artifact")
    manifest = dict(schema_version=1, dataset=dataset, coordinates=coordinates,
                    precomputation=precomputation, artifacts=records)
    atomic_json(manifest, output)
    return manifest


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported data manifest schema")
    if (not manifest.get("dataset", {}).get("id") or
            not manifest.get("coordinates", {}).get("frame") or
            not manifest.get("coordinates", {}).get("units") or not manifest.get("artifacts")):
        raise ValueError("Incomplete data manifest")
    for name, record in manifest["artifacts"].items():
        source = (path.parent / record["path"]).resolve()
        if file_digest(source) != record["sha256"]:
            raise ValueError(f"Data artifact {name!r} changed since precomputation")
        record["path"] = str(source)
    return manifest


def resolve_task_config(config, custom_factory=False):
    """Resolve paths and verify data once per process/file revision."""
    cfg = copy.deepcopy(config)
    task = cfg.setdefault("task", {})
    expected = task.get("data_identity")
    manifest = load_manifest(task["manifest"]) if task.get("manifest") else None
    if manifest is not None:
        identity = {"manifest_sha256": file_digest(task["manifest"]),
                    "dataset": manifest["dataset"], "coordinates": manifest["coordinates"],
                    "artifacts": {name: record["sha256"] for name, record in manifest["artifacts"].items()}}
        if not custom_factory and not task.get("factory") and not task.get("provider_factory"):
            if (manifest["coordinates"].get("frame") != "viewer_voxel" or
                    manifest["coordinates"].get("units") != "voxel"):
                raise ValueError("Default skeleton provider requires viewer_voxel/voxel coordinates; use a custom task factory")
            if "reset_data" not in manifest["artifacts"]:
                raise ValueError("Default task manifest needs reset_data")
            env = cfg.setdefault("env", {})
            for artifact, key in (("reset_data", "parquet_path"), ("holdout", "holdout_parquet")):
                if artifact in manifest["artifacts"]:
                    value = manifest["artifacts"][artifact]["path"]
                    if env.get(key) and Path(env[key]).resolve() != Path(value):
                        raise ValueError(f"env.{key} conflicts with task.manifest")
                    env[key] = value
    else:
        identity = {"artifacts": {}}
        for key in ("parquet_path", "holdout_parquet"):
            value = cfg.get("env", {}).get(key)
            if value:
                identity["artifacts"][key] = file_digest(value)
        if not custom_factory and not task.get("factory") and not task.get("provider_factory") and not cfg.get("env", {}).get("parquet_path"):
            raise ValueError("Supply env.parquet_path, task.manifest, or a custom task.factory")
    if expected is not None and expected != identity:
        raise ValueError("Task data changed after the run configuration was resolved")
    task["data_identity"] = identity
    return cfg, manifest


def validate_scene(manifest, dataset):
    if manifest is None:
        return
    actual = asdict(dataset)
    for key in ("em_url", "seg_url", "voxel_nm"):
        expected = manifest["dataset"].get(key)
        if expected is not None and expected != (list(actual[key]) if key == "voxel_nm" else actual[key]):
            raise ValueError(f"Manifest dataset.{key} disagrees with the rendered dataset")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    existing = sub.add_parser("manifest")
    existing.add_argument("--reset-data", required=True)
    existing.add_argument("--holdout")
    existing.add_argument("--dataset-id", required=True)
    existing.add_argument("--release")
    existing.add_argument("--start-url")
    existing.add_argument("--parameters", help="JSON file recording existing precomputation parameters")
    existing.add_argument("--output", required=True)
    precompute = sub.add_parser("precompute")
    precompute.add_argument("--config", required=True)
    precompute.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "manifest":
        from ngllib.dataset import DatasetSpec

        scene = DatasetSpec.from_start_url(args.start_url) if args.start_url else DatasetSpec.from_config()
        dataset = {"id": args.dataset_id, "release": args.release, **asdict(scene)}
        artifacts = {"reset_data": args.reset_data}
        if args.holdout:
            artifacts["holdout"] = args.holdout
        params = json.loads(Path(args.parameters).read_text()) if args.parameters else {}
        write_manifest(args.output, dataset=dataset,
                       coordinates={"frame": "viewer_voxel", "units": "voxel"},
                       precomputation={"factory": "existing-artifacts", "params": params}, artifacts=artifacts)
    else:
        from .env_build import load_config
        from .tasks import import_factory

        cfg = load_config(args.config)
        spec = cfg["task"]["precompute"]
        output_dir = Path(args.output).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        product = import_factory(spec["factory"])(copy.deepcopy(spec.get("params", {})), output_dir)
        write_manifest(args.output, dataset=product["dataset"], coordinates=product["coordinates"],
                       precomputation=spec, artifacts=product["artifacts"])
    print("DATA-MANIFEST-OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

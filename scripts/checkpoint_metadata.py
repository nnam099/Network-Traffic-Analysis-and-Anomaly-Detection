"""Repair missing architecture metadata in a v14 checkpoint."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def infer_dims(checkpoint: dict) -> tuple[int, int]:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing model_state_dict")

    try:
        ae_hidden = int(state["ae.enc.4.weight"].shape[1])
        hidden = int(state["backbone.input_proj.0.weight"].shape[0])
    except KeyError as exc:
        raise ValueError(f"checkpoint does not look like an IDS v14 checkpoint: missing {exc}") from exc

    return ae_hidden, hidden


def patch_checkpoint(path: Path, backup: bool = True) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ae_hidden, hidden = infer_dims(checkpoint)

    print(f"ae_hidden old: {checkpoint.get('ae_hidden')}")
    print(f"ae_hidden new: {ae_hidden}")
    print(f"hidden old: {checkpoint.get('hidden')}")
    print(f"hidden new: {hidden}")

    checkpoint["ae_hidden"] = ae_hidden
    checkpoint["hidden"] = hidden

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"backup: {backup_path}")

    torch.save(checkpoint, path)
    print(f"patched: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch IDS v14 checkpoint hidden dimension metadata.")
    parser.add_argument("--path", default="checkpoints/ids_v14_model.pth", help="Checkpoint path to patch.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak file before writing.")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)

    patch_checkpoint(path, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

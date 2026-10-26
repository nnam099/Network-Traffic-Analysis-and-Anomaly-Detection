from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ARTIFACTS = [
    "checkpoints/ids_v14_model.pth",
    "checkpoints/ids_v14_pipeline.pkl",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path, artifact_paths: list[str] | None = None) -> dict:
    manifest = {"artifacts": {}}
    for rel in artifact_paths or DEFAULT_ARTIFACTS:
        path = root / rel
        if not path.exists():
            manifest["artifacts"][rel] = {"exists": False}
            continue
        manifest["artifacts"][rel] = {
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return manifest


def verify_manifest(root: Path, manifest: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    artifacts = manifest.get("artifacts", {})
    for rel, expected in artifacts.items():
        path = root / rel
        if not expected.get("exists"):
            if path.exists():
                warnings.append(f"Artifact exists but manifest expected missing: {rel}")
            continue
        if not path.exists():
            errors.append(f"Artifact missing: {rel}")
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if int(expected.get("size_bytes", -1)) != actual_size:
            errors.append(f"Artifact size changed: {rel}")
        if expected.get("sha256") != actual_sha:
            errors.append(f"Artifact sha256 changed: {rel}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify local IDS artifact hash manifests.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--manifest", default="results/artifact_manifest.json", help="Manifest JSON path.")
    parser.add_argument("--verify", action="store_true", help="Verify existing manifest instead of writing one.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest_path = (root / args.manifest).resolve()

    if args.verify:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = verify_manifest(root, manifest)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    manifest = build_manifest(root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

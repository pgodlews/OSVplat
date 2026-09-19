"""Person-mask backends: the one place a new segmentation model is declared.

Adding one (say a future SAM release) means:
  1. an entry here,
  2. an `elif args.backend == "<name>":` branch in scripts/70_person_masks.py
     that builds its `segment(work)` function, plus its name in `--backend`,
  3. its Python dependencies in scripts/setup_gsplat_venv.sh (the Docker image
     runs the same script),
  4. if its weights are gated, a line in scripts/get_mask_weights.sh.
The job validator, the stage command line, the API's availability report and
the web UI's backend list all read this table.

Weights of gated models are never shipped. They live in <models dir>/<weights>
(SPLAT_ROOT/models natively, the ./models volume in Docker), and a backend
whose weights are missing is offered as unavailable, with `setup` saying how to
get them, rather than accepted and left to fail at the mask stage.
"""
from __future__ import annotations

from pathlib import Path

MASK_BACKENDS: dict[str, dict] = {
    "maskrcnn": {
        "label": "Mask R-CNN: people only (BSD, no setup)",
        "prompts": False,           # COCO "person" class only
        "default_score": 0.5,
        "weights": None,            # torchvision downloads them on first use
        "licence": "BSD-3-Clause (torchvision)",
        "setup": "",
    },
    "sam3": {
        "label": "SAM 3: anything named in the prompts",
        "prompts": True,
        "default_score": 0.3,
        "weights": "sam3",
        "hf_repo": "facebook/sam3",
        "licence": "Meta SAM licence (gated; accept it on Hugging Face)",
        "setup": ("Accept the licence at https://huggingface.co/facebook/sam3, "
                  "then run scripts/get_mask_weights.sh sam3 "
                  "(Docker: docker compose run --rm queue get-weights sam3) "
                  "and restart the queue."),
    },
}

DEFAULT_BACKEND = "maskrcnn"


def weights_dir(name: str, models_root: Path) -> Path | None:
    w = MASK_BACKENDS[name]["weights"]
    return None if w is None else models_root / w


def availability(models_root: Path) -> list[dict]:
    """Every backend with whether it can run here, for the API and the UI."""
    out = []
    for name, meta in MASK_BACKENDS.items():
        d = weights_dir(name, models_root)
        ok = d is None or (d / "config.json").is_file()
        out.append({
            "name": name, "label": meta["label"], "prompts": meta["prompts"],
            "licence": meta["licence"], "available": ok,
            "reason": "" if ok else f"weights not found in {d}. {meta['setup']}",
        })
    return out

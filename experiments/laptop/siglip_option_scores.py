"""SigLIP fusion (Table 4.10), step 1: SigLIP's own score for each (image, option) pair.

Embeds each val image and each option text with SigLIP's image and text encoders and stores
SigLIP's image-text logits (logit scale * cosine + logit bias) for three text templates:
"option" ("<option>"), "photo" ("a photo of <option>") and "qa" ("<question> <option>"), and
prints SigLIP's zero-shot accuracy per template (Section 3.6: 63.8% qa, 62.2% option, 61.7%
photo). The scores are fused with the LLM's letter scores by experiments/laptop/fuse_siglip.py.
Run by the "siglip" job of experiments/laptop/run_mac_AB.sh; the recorded output
results/runs/laptop/siglip_val/scores.jsonl is a release asset.

    python experiments/laptop/siglip_option_scores.py \\
        --aokvqa-dir data/raw/aokvqa --coco-dir data/raw/coco --out results/runs/laptop/siglip_val

Writes <out>/scores.jsonl: per question and prompt template, the 4 image-option logits
(original option order).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cmr.data.aokvqa import coco_image_path, load_aokvqa  # noqa: E402


def _vec(out: object) -> torch.Tensor:
    """transformers 5 returns a model output; the projected embedding is pooler_output."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output  # type: ignore[attr-defined]


TEMPLATES = {
    "option": "{o}",
    "photo": "a photo of {o}",
    "qa": "{q} {o}",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aokvqa-dir", required=True)
    ap.add_argument("--coco-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True, help="output folder for scores.jsonl")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    items = load_aokvqa(args.aokvqa_dir, args.split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scale, bias = model.logit_scale.exp().item(), model.logit_bias.item()
    t0 = time.time()
    with open(out / "scores.jsonl", "w") as f, torch.no_grad():
        for b in range(0, len(items), args.batch):
            batch = items[b : b + args.batch]
            ims = [
                Image.open(coco_image_path(args.coco_dir, args.split, it.image_id)).convert("RGB")
                for it in batch
            ]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(args.device)
            img = _vec(model.get_image_features(pixel_values=px))
            img = img / img.norm(dim=-1, keepdim=True)
            rows = [
                {"question_id": it.question_id, "correct_idx": it.correct_choice_idx}
                for it in batch
            ]
            for name, tpl in TEMPLATES.items():
                texts = [tpl.format(q=it.question, o=o) for it in batch for o in it.choices]
                tok = proc(
                    text=texts,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                ).to(args.device)
                txt = _vec(model.get_text_features(**tok))
                txt = txt / txt.norm(dim=-1, keepdim=True)
                k = 0
                for i, it in enumerate(batch):
                    n = len(it.choices)
                    sims = (txt[k : k + n] @ img[i]).tolist()
                    rows[i][name] = [scale * s + bias for s in sims]
                    k += n
            for r in rows:
                f.write(json.dumps(r) + "\n")
            print(f"{b + len(batch)}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    for name in TEMPLATES:
        rs = [json.loads(line) for line in open(out / "scores.jsonl")]
        acc = sum(max(range(len(r[name])), key=r[name].__getitem__) == r["correct_idx"] for r in rs)
        print(f"SigLIP zero-shot accuracy ({name}): {100 * acc / len(rs):.1f}%")


if __name__ == "__main__":
    main()

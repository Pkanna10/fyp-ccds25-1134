# Third-party models and data

This repository contains no third-party data or model weights. The models are downloaded from
the Hugging Face Hub on first use, and `scripts/download_data.sh` downloads the data from their
official sources. The terms below are as stated by each source (not legal advice).

## Models

| Model | Use | Licence | Revision |
|---|---|---|---|
| `google/siglip-so400m-patch14-384` | frozen vision encoder | Apache-2.0 | `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` |
| `Qwen/Qwen2-0.5B-Instruct` | frozen LLM (0.5B) | Apache-2.0 | `c540970f9e29518b1d8f06ab8b24cba66ad77b6d` |
| `Qwen/Qwen2-7B-Instruct` | frozen LLM (7B) | Apache-2.0 | `f2826a00ceef68f0f2b946d945ecc0477ce4450c` |
| `lmms-lab/llava-onevision-projectors` | initial bridge θ₀ | Apache-2.0 | `a325adc5b0eae860a2d156b47b793dc727fd394e` |

The revisions are those recorded for the cloud runs in `results/runs/cloud/_status_phase*.json`
and enforced by `experiments/cloud/fyp_modal.py`. The trained bridges released with this
project are modified copies of the LLaVA-OneVision projectors: each is a two-layer MLP of the
same shape whose weights were further trained on A-OKVQA, and they are shared under Apache-2.0.

## Data

- **A-OKVQA** (v1p0, `allenai/aokvqa`): no licence statement was found for the annotation
  files; the code repository is Apache-2.0. The annotation files are not redistributed;
  `scripts/download_data.sh` checks their SHA-256.
- **COCO 2017 images**: copyright stays with each Flickr photographer under the per-image
  Creative Commons licence recorded in the COCO annotation files. No images are redistributed.
- **COCO captions** (`captions_val2017.json`): CC BY 4.0. Not redistributed.
- **MMStar** (`Lin-Chen/MMStar`; transfer check of Table 4.16 only): downloaded on first use
  from its Hugging Face dataset page; not redistributed.

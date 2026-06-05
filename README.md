# CCDG: Compositional Contrastive Dense Grounding

**A Multi-Scale Training Paradigm for Referring Expression Comprehension**

> Sarthak Pandey, Mobeen Ur Rehman  
> IEEE Transactions on Multimedia (under review)

## Results

| Split | Baseline | CCDG | Gain |
|-------|----------|------|------|
| RefCOCO testA | 36.90% | **50.27%** | +13.37 pp |
| RefCOCO testB | 33.16% | 48.99% | +15.83 pp |
| RefCOCO+ testA | 36.32% | 37.32% | +1.00 pp |
| RefCOCOg test | 39.85% | 43.49% | +3.64 pp |

## Setup

```bash
pip install torch torchvision transformers spacy
python -m spacy download en_core_web_sm
```

## Data

Download RefCOCO/+/g from https://github.com/lichengunc/refer  
Download COCO train2014 images from https://cocodataset.org

## Training

```bash
# Full CCDG
python scripts/train.py --config configs/config.py

# Ablations
python scripts/train.py --config configs/config_ablation_baseline.py
python scripts/train.py --config configs/config_ablation_attr.py
python scripts/train.py --config configs/config_ablation_attr_subj.py
```

## Evaluation

```bash
python scripts/eval.py \
  --checkpoint /path/to/checkpoints/best_model \
  --output eval_results.txt
```

## Weights

Model weights will be released upon paper acceptance.

## Citation

```bibtex
@article{pandey2026ccdg,
  title={Compositional Contrastive Dense Grounding},
  author={Pandey, Sarthak and Ur Rehman, Mobeen},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  note={Under review}
}
```
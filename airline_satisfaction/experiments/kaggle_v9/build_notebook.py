"""Assemble source only; never import or execute model training locally."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTS = ['neural_models.py', 'route_models.py', 'e9_pipeline.py']
source = '\n\n'.join((ROOT / name).read_text() for name in PARTS)
ast.parse(source)
(ROOT / 'run_experiment9.py').write_text(source)
intro = """# Airline Satisfaction — E9 Nested Neural Ensemble

All training and inference runs on Kaggle GPU. E8 reference: public ROC AUC **0.96126**.

## Candidates
- RealMLP: original 3-epoch recipe, 6 epochs with half learning rate, and 3 epochs without label smoothing. Each averages seeds 42, 137 and 2026, with 8 networks per seed.
- TabM: independent raw-feature neural view, numerical embeddings, 32 shared-weight members, fixed 12 epochs.
- LightGBM: E8 distance-feature control, smoothed leave-one-out group statistics/deviations/variability, and the same improved view without original-data teacher features.
- Exact E8 comparison: seed-42-only RealMLP plus E8 tree at fixed 75/25 logit weights.

## Evaluation
Five outer folds use seed 42. In each outer training partition, a separate 20% inner holdout selects the blend. Base learners are fitted only on inner-fit rows for this step, including all preprocessing and target encoding. Three-way meta validation within the inner holdout chooses a fixed anchor, a bounded finer logit grid, or a regularized logistic stack. Base learners are then refitted on all outer-training rows; the selected blend is scored on the outer fold. Outer validation labels never enter training, stopping or blend selection. Each fold's selected procedure also predicts the competition test set; these five predictions are averaged.

This is nested evaluation within E9, but the dataset has informed earlier experiments. It is development evidence, not a new untouched holdout. The primary candidate is `experiment9/submission_nested_selected.csv`. Results include raw predictions, per-fold AUC/log loss, artifacts, and a matched E8 comparison. **No automatic competition submission.**

## Credits and provenance
- [Vladimir Demidov / yekenot](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit): RealMLP feature/training recipe, Apache 2.0.
- [MathsOnChain](https://www.kaggle.com/code/mathsonchain/s6e10-lgbm-trasy): original-data teacher and distance-group ideas, Apache 2.0.
- [PyTabKit 1.7.3](https://github.com/dholzmueller/pytabkit/tree/v1.7.3) and [Yandex Research TabM](https://github.com/yandex-research/tabm): neural implementations.
- [Original airline data](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction): teacher labels only; exact feature matches to competition rows are excluded.

The TabM implementation uses PyTabKit's pinned PyTorch core to support fixed-epoch training without an early-stopping validation set. The 6-epoch RealMLP schedule is identical in inner fits and outer refits. Every fitted preprocessing map uses the corresponding training partition only.
"""
cells = [
    dict(cell_type='markdown', metadata={}, source=intro.splitlines(keepends=True)),
    dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
         source=['%pip install -q pytabkit==1.7.3\n']),
    dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
         source=source.splitlines(keepends=True)),
]
notebook = dict(cells=cells, metadata={
    'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
    'language_info': {'name':'python'},
    'kaggle': {'accelerator':'gpu', 'isInternetEnabled':True, 'language':'python',
               'isGpuEnabled':True}}, nbformat=4, nbformat_minor=4)
path = ROOT / 'airline_kaggle_experiment9.ipynb'
path.write_text(json.dumps(notebook, indent=1))
(ROOT/'README.md').write_text(intro)
print(json.dumps(dict(notebook=str(path), code_lines=len(source.splitlines()),
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    notebook_bytes=path.stat().st_size)))

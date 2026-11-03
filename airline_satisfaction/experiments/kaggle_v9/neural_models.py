"""E9 neural candidates. Import/source checks are local; all fitting is Kaggle GPU only.

RealMLP feature recipe: Vladimir Demidov (yekenot), Apache 2.0.
https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit
TabM: Yandex Research and the pinned PyTabKit 1.7.3 implementation.
https://github.com/yandex-research/tabm
https://github.com/dholzmueller/pytabkit/tree/v1.7.3

TabM's PyTabKit sklearn wrapper requires a validation set and restores its best
epoch. We deliberately use its pinned PyTorch core with a fixed-epoch loop:
neither tuning/outer validation features nor labels are used during fitting.
"""
import gc
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import cloudpickle
import joblib
import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.preprocessing import TargetEncoder


NEURAL_COMBOS = [('Class', 'Type of Travel', 'Gender'),
                 ('Class', 'Type of Travel', 'Age'),
                 ('Food and drink', 'Cleanliness'),
                 ('Flight Distance', 'Departure/Arrival time convenient')]

REALMLP_BASE_PARAMS = dict(
    verbosity=1, device='cuda', n_threads=4, val_metric_name='1-auc_ovr',
    n_ens=8, n_epochs=3, batch_size=288, use_early_stopping=False,
    lr=0.053, wd=0.015, sq_mom=0.988, lr_sched='flat_anneal',
    wd_sched='cos_log_15', first_layer_lr_factor=0.25, embedding_size=5,
    max_one_hot_cat_size=2, hidden_sizes=[512, 256, 128], act='silu',
    p_drop=0.05, p_drop_sched='pow5', plr_hidden_1=16, plr_hidden_2=8,
    plr_act_name='gelu', plr_lr_factor=0.1151, plr_sigma=0.75,
    ls_eps=0.01, ls_eps_sched='sqrt_cos', add_front_scale=False,
    bias_init_mode='neg-uniform-dynamic-2',
    tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip',
          'embedding', 'l2_normalize'])

NEURAL_RECIPES = {
    'realmlp_control': dict(REALMLP_BASE_PARAMS),
    'realmlp_long': dict(REALMLP_BASE_PARAMS, n_epochs=6, lr=0.0265),
    'realmlp_no_smoothing': dict(REALMLP_BASE_PARAMS, use_ls=False),
    'tabm': dict(n_epochs=12, batch_size=1024, n_blocks=2, d_block=256,
                 dropout=0.1, k=32, d_embedding=16, num_emb_n_bins=48,
                 lr=0.002, weight_decay=0.0001, min_lr=0.00001,
                 arch_type='tabm', share_training_batches=True,
                 precision='float32', numerical_preprocessing='median/IQR'),
}


def _neural_artifact(prefix, suffix):
    path = Path(str(prefix) + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_realmlp(train, others, y, seed):
    """The E8 recipe, with all maps fitted on train and cross-fitted train TE."""
    cats = train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    nums = [c for c in train if c not in cats]
    frames = [train.copy().reset_index(drop=True)] + [a.copy().reset_index(drop=True) for a in others]
    for a in frames:
        for c in cats:
            a[c] = a[c].astype('string').fillna('__missing__').astype(str)
        for c in nums:
            # Constant imputation exactly preserves the E8 control recipe.
            a[c] = a[c].fillna(0.)
    maps = {}
    for c in cats:
        levels = pd.Index(frames[0][c].unique())
        maps[c] = levels
        for a in frames:
            a[c] = pd.Categorical(a[c], categories=levels)
    for c in nums:
        levels = pd.Index(frames[0][c].unique())
        name = c + '_cat_'
        maps[name] = levels
        for a in frames:
            a[name] = pd.Categorical(a[c], categories=levels)
    for c in cats + ['Leg room service', 'Flight Distance']:
        counts = frames[0][c].value_counts()
        name = '_' + c + '_count'
        maps[name] = counts
        for a in frames:
            a[name] = a[c].astype(object).map(counts).fillna(0).astype('float32')
    names = []
    for cols in NEURAL_COMBOS:
        name = '_'.join(cols) + '_'
        names.append(name)
        values = [a[list(cols)].astype(str).agg('_'.join, axis=1) for a in frames]
        levels = pd.Index(values[0].unique())
        maps[name] = levels
        for a, v in zip(frames, values):
            a[name] = pd.Categorical(v, categories=levels)
    encoder = TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=seed)
    te_input = [a[names].astype(str) for a in frames]
    encoded = [encoder.fit_transform(te_input[0], y)] + [encoder.transform(a) for a in te_input[1:]]
    drop = ['Class_Type of Travel_Age_', 'Flight Distance_Departure/Arrival time convenient_']
    for a, enc in zip(frames, encoded):
        a[[f'_{c}TE' for c in names]] = enc
        a.drop(columns=drop, inplace=True)
    return frames, dict(maps=maps, target_encoder=encoder, combos=NEURAL_COMBOS,
                        columns=frames[0].columns.tolist(), raw_columns=train.columns.tolist(),
                        categorical_columns=cats, numeric_columns=nums,
                        numeric_imputation='constant_zero_E8_control', seed=int(seed))


def _neural_predict_batched(model, frame):
    return np.concatenate([model.predict_proba(frame.iloc[i:i+20000])[:, 1]
                           for i in range(0, len(frame), 20000)])


def _checkpoint_neural_predictions(prefix, predictions):
    if prefix is not None:
        np.savez_compressed(_neural_artifact(prefix, '_predictions.npz'),
                            **{f'prediction_{i}': p for i, p in enumerate(predictions)})


def _export_realmlp(model, frame, expected, prefix):
    """Prediction checkpoints already exist; model serialization is nonfatal."""
    try:
        model.to('cpu')
        path = _neural_artifact(prefix, '.cloudpickle')
        with path.open('wb') as stream:
            cloudpickle.dump(model, stream)
        with path.open('rb') as stream:
            restored = cloudpickle.load(stream)
        np.testing.assert_allclose(restored.predict_proba(frame.iloc[:16])[:, 1],
                                   expected[:16], rtol=1e-4, atol=1e-5)
        del restored
        print('E9 NEURAL EXPORT VERIFIED', str(path), flush=True)
    except Exception as exc:
        _neural_artifact(prefix, '_export_warning.json').write_text(json.dumps({'error': repr(exc)}))
        print('E9 NEURAL EXPORT WARNING', repr(exc), flush=True)


def _fit_predict_realmlp(name, train, y, others, seed, artifact_prefix):
    frames, preprocessing = _prepare_realmlp(train, others, y, seed)
    params = dict(NEURAL_RECIPES[name], random_state=int(seed))
    if artifact_prefix is not None:
        joblib.dump(preprocessing, _neural_artifact(artifact_prefix, '_preprocessing.joblib'))
        _neural_artifact(artifact_prefix, '_params.json').write_text(json.dumps(params, indent=2))
    model = RealMLP_TD_Classifier(**params)
    # RealMLP explicitly supports an empty validation index list. It uses all train rows.
    model.fit(frames[0], y, val_idxs=np.array([], dtype=np.int64))
    predictions = [_neural_predict_batched(model, frame).astype(np.float64) for frame in frames[1:]]
    _checkpoint_neural_predictions(artifact_prefix, predictions)
    if artifact_prefix is not None and predictions:
        _export_realmlp(model, frames[1], predictions[0], artifact_prefix)
    del model, frames, preprocessing
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def _prepare_tabm(train, others):
    """A distinct raw-feature view; training-only medians/IQR/category maps."""
    cats = train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    nums = [c for c in train if c not in cats]
    clean_train = train[nums].replace([np.inf, -np.inf], np.nan)
    medians = clean_train.median().fillna(0.)
    filled = clean_train.fillna(medians)
    # PWL bins require nonconstant input features; preserve this exact mask for inference.
    nums = filled.columns[filled.nunique(dropna=False).gt(1)].tolist()
    medians = medians[nums]
    scales = (filled[nums].quantile(.75) - filled[nums].quantile(.25)).replace(0., 1.)
    categories = {}
    for c in cats:
        values = train[c].astype('string').fillna('__missing__').astype(str)
        categories[c] = {value: i + 1 for i, value in enumerate(pd.unique(values))}
    arrays = []
    for frame in [train] + list(others):
        numeric = frame[nums].replace([np.inf, -np.inf], np.nan).fillna(medians)
        xn = ((numeric - medians) / scales).to_numpy(dtype=np.float32)
        xc = np.column_stack([
            frame[c].astype('string').fillna('__missing__').astype(str)
            .map(categories[c]).fillna(0).to_numpy(dtype=np.int64) for c in cats
        ]) if cats else np.empty((len(frame), 0), dtype=np.int64)
        assert np.isfinite(xn).all()
        arrays.append((xn, xc))
    prep = dict(numeric_columns=nums, categorical_columns=cats,
                medians=medians, scales=scales, categories=categories,
                categorical_cardinalities=[len(categories[c])+1 for c in cats],
                raw_columns=train.columns.tolist(), unknown_category_code=0)
    return arrays, prep


def _tabm_predict(model, arrays, device, batch_size=1024):
    model.eval()
    xn, xc = arrays
    result = []
    with torch.inference_mode():
        for start in range(0, len(xn), batch_size):
            numeric = torch.as_tensor(xn[start:start+batch_size], device=device)
            categorical = torch.as_tensor(xc[start:start+batch_size], device=device) if xc.shape[1] else None
            logits = model(numeric, categorical).float()
            # Each member is trained separately, then predictions average in probability space.
            result.append(logits.softmax(dim=-1)[:, :, 1].mean(dim=1).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def _fit_predict_tabm(train, y, others, seed, artifact_prefix):
    from pytabkit.models.nn_models.tabm import Model, make_parameter_groups
    from pytabkit.models.nn_models.rtdl_num_embeddings import compute_bins

    params = dict(NEURAL_RECIPES['tabm'])
    arrays, prep = _prepare_tabm(train, others)
    train_num, train_cat = arrays[0]
    # Bin edges are computed solely from this fit's training features, without labels.
    bins = compute_bins(torch.as_tensor(train_num), n_bins=params['num_emb_n_bins'])
    constructor = dict(
        n_num_features=train_num.shape[1], cat_cardinalities=prep['categorical_cardinalities'],
        n_classes=2, backbone=dict(type='MLP', n_blocks=params['n_blocks'],
                                   d_block=params['d_block'], dropout=params['dropout']),
        bins=bins, num_embeddings=dict(type='PiecewiseLinearEmbeddings',
                                       d_embedding=params['d_embedding'], activation=False, version='B'),
        arch_type=params['arch_type'], k=params['k'], share_training_batches=True)
    model = Model(**constructor).to('cuda')
    optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=params['lr'],
                                  weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params['n_epochs'], eta_min=params['min_lr'])
    xn = torch.as_tensor(train_num, device='cuda')
    xc = torch.as_tensor(train_cat, device='cuda') if train_cat.shape[1] else None
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device='cuda')
    if artifact_prefix is not None:
        joblib.dump(prep, _neural_artifact(artifact_prefix, '_preprocessing.joblib'))
        _neural_artifact(artifact_prefix, '_params.json').write_text(
            json.dumps(dict(params, seed=int(seed)), indent=2))
    for epoch in range(params['n_epochs']):
        started = time.time()
        total_loss = torch.zeros((), device='cuda')
        model.train()
        for idx in torch.randperm(len(y), device='cuda').split(params['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xn[idx], xc[idx] if xc is not None else None)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), yt[idx].repeat_interleave(model.k))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'TabM nonfinite loss in epoch {epoch + 1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += loss.detach() * len(idx)
        scheduler.step()
        print('E9 TABM EPOCH', json.dumps(dict(epoch=epoch+1, epochs=params['n_epochs'],
              seed=int(seed), training_loss=float(total_loss.cpu()/len(y)),
              seconds=time.time()-started)), flush=True)
    predictions = [_tabm_predict(model, a, 'cuda') for a in arrays[1:]]
    _checkpoint_neural_predictions(artifact_prefix, predictions)
    if artifact_prefix is not None:
        try:
            model.to('cpu')
            path = _neural_artifact(artifact_prefix, '.pt')
            # Only our own freshly created state dictionary is reloaded for verification.
            bundle = dict(state_dict=model.state_dict(), constructor=constructor,
                          seed=int(seed), pytabkit_version=version('pytabkit'))
            torch.save(bundle, path)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            restored = Model(**loaded['constructor'])
            restored.load_state_dict(loaded['state_dict'])
            if predictions:
                small = tuple(a[:16] for a in arrays[1])
                np.testing.assert_allclose(_tabm_predict(restored, small, 'cpu'),
                                           predictions[0][:16], rtol=1e-4, atol=1e-5)
            print('E9 NEURAL EXPORT VERIFIED', str(path), flush=True)
            del restored, loaded, bundle
        except Exception as exc:
            _neural_artifact(artifact_prefix, '_export_warning.json').write_text(json.dumps({'error': repr(exc)}))
            print('E9 NEURAL EXPORT WARNING', repr(exc), flush=True)
    del model, optimizer, scheduler, xn, xc, yt, arrays
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def fit_predict_neural(name, train, y, others, seed=42, artifact_prefix=None):
    """Fit one predeclared candidate on Kaggle; return one probability vector per other frame.

    Only `y` for `train` is accepted. Every preprocessing map and embedding bin is
    fitted inside this call's train rows. Selection of recipes/seeds/blend weights
    belongs to the orchestrator's inner evaluation, never to this function.
    """
    if not Path('/kaggle/working').exists() or not torch.cuda.is_available():
        raise RuntimeError('E9 model fitting must run on Kaggle with a GPU.')
    if version('pytabkit') != '1.7.3':
        raise RuntimeError('E9 requires pytabkit==1.7.3 for its verified model APIs.')
    if name not in NEURAL_RECIPES:
        raise ValueError(f'Unknown neural candidate: {name}')
    y = np.asarray(y, dtype=np.int64)
    assert len(train) == len(y) and set(np.unique(y)) == {0, 1}
    assert all(list(frame.columns) == list(train.columns) for frame in others)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    started = time.time()
    print('E9 NEURAL START', json.dumps(dict(name=name, seed=int(seed), rows=len(train),
                                           prediction_rows=[len(x) for x in others])), flush=True)
    predictions = (_fit_predict_tabm(train, y, others, seed, artifact_prefix)
                   if name == 'tabm' else
                   _fit_predict_realmlp(name, train, y, others, seed, artifact_prefix))
    assert len(predictions) == len(others)
    for frame, prediction in zip(others, predictions):
        assert prediction.shape == (len(frame),) and np.isfinite(prediction).all()
        assert ((prediction >= 0.) & (prediction <= 1.)).all()
    print('E9 NEURAL COMPLETE', name, seed, 'seconds', round(time.time()-started, 1), flush=True)
    return predictions

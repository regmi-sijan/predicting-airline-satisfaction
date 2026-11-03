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


"""E9 distance-feature tree ablations; training is intended for Kaggle only.

Credits: E8 and MathsOnChain's original-data teacher / distance-group ideas:
https://www.kaggle.com/code/mathsonchain/s6e10-lgbm-trasy

``route_e8`` reproduces the E8 tree recipe. ``route_smooth`` tests smoothed,
leave-one-out training feature aggregates plus deviations; its no-teacher
ablation removes both features that use original-data labels. Exact distance
groups are a feature proxy, not asserted to identify actual flight routes.
"""

import hashlib as _route_hashlib
import json as _route_json
from pathlib import Path as _RoutePath

import joblib as _route_joblib
import lightgbm as _route_lgb
import numpy as _route_np
import pandas as _route_pd


_ROUTE_CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
_ROUTE_RATINGS = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Cleanliness',
]
_ROUTE_VALUES = _ROUTE_RATINGS + ['Age']
_ROUTE_NAMES = {'route_e8', 'route_smooth', 'route_smooth_no_teacher'}
_ROUTE_ALPHA = 20.0


def _route_category_align(train, others):
    """Learn categorical levels exclusively from the supplied training frame."""
    frames = [train.copy()] + [frame.copy() for frame in others]
    levels = {}
    for column in _ROUTE_CATS:
        levels[column] = _route_pd.Index(frames[0][column].dropna().unique())
        for frame in frames:
            frame[column] = _route_pd.Categorical(
                frame[column], categories=levels[column])
    return frames, levels


def build_external_teacher(x, xt, comp, out):
    """Return train/test teacher probabilities, distance rate, prior and audit.

    The teacher sees original-data labels only. Exact feature matches to any
    competition train/test row are excluded before fitting, following E8.
    """
    out = _RoutePath(out)
    out.mkdir(parents=True, exist_ok=True)
    comp = _RoutePath(comp)
    paths = [path for path in _RoutePath('/kaggle/input').rglob('*.csv')
             if 'airline-passenger-satisfaction' in str(path)
             and path.parent != comp]
    if not paths:
        raise ValueError('Attach teejmahal20/airline-passenger-satisfaction input.')
    pieces, used_paths = [], []
    for path in paths:
        frame = _route_pd.read_csv(path)
        if set(x.columns).union({'satisfaction'}).issubset(frame.columns):
            pieces.append(frame[list(x.columns) + ['satisfaction']])
            used_paths.append(path)
    if not pieces:
        raise ValueError('No original-data CSVs have all required shared features.')
    original = _route_pd.concat(pieces, ignore_index=True)
    original['satisfaction'] = original.satisfaction.map(
        {'satisfied': 1, 'neutral or dissatisfied': 0})
    if original.satisfaction.isna().any():
        raise ValueError('Unexpected or missing original-data label encoding.')
    competition = _route_pd.concat([x, xt], ignore_index=True)
    blocked = _route_pd.util.hash_pandas_object(competition, index=False)
    original_keys = _route_pd.util.hash_pandas_object(original[x.columns], index=False)
    keep = ~original_keys.isin(blocked)
    audit = dict(
        original_rows=len(original), exact_overlap_removed=int((~keep).sum()),
        sources=[str(path) for path in used_paths],
        source_sha256={str(path): _route_hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in used_paths},
        label_source='original data only; no competition labels',
    )
    original = original.loc[keep].drop_duplicates().reset_index(drop=True)
    if len(original) <= 100000:
        raise ValueError('Too few original-data rows after filtering; inspect inputs.')
    frames, categories = _route_category_align(original[x.columns], [x, xt])
    parameters = dict(
        n_estimators=800, learning_rate=.04, num_leaves=31,
        min_child_samples=30, reg_lambda=3, random_state=42,
        n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True,
    )
    teacher = _route_lgb.LGBMClassifier(**parameters)
    teacher.fit(frames[0], original.satisfaction.astype(int))
    train_probabilities = teacher.predict_proba(frames[1])[:, 1]
    test_probabilities = teacher.predict_proba(frames[2])[:, 1]
    teacher.booster_.save_model(str(out / 'original_teacher.txt'))
    grouped = original.groupby('Flight Distance').satisfaction.agg(['sum', 'count'])
    prior = float(original.satisfaction.mean())
    rate = (grouped['sum'] + 20 * prior) / (grouped['count'] + 20)
    _route_joblib.dump(dict(rate=rate, prior=prior), out / 'original_distance_rate.joblib')
    _route_joblib.dump(dict(categories=categories, columns=list(x.columns),
                            parameters=parameters), out / 'original_teacher_preprocess.joblib')
    audit['retained_rows'] = len(original)
    audit['parameters'] = parameters
    (out / 'teacher_audit.json').write_text(_route_json.dumps(audit, indent=2))
    _route_np.savez_compressed(out / 'teacher_predictions.npz',
                              train=train_probabilities, test=test_probabilities)
    print('EXTERNAL TEACHER READY', _route_json.dumps(audit), flush=True)
    return train_probabilities, test_probabilities, rate, prior, audit


def _route_build_maps(train, smooth):
    grouped = train.groupby('Flight Distance')
    if not smooth:
        aggregates = grouped[_ROUTE_VALUES].mean().add_prefix('route_mean_')
        aggregates['route_count'] = grouped.size()
        fallback = aggregates.mean()
        fallback['route_count'] = 0
        return dict(aggregates=aggregates, fallback=fallback, smooth=False)
    squared_values = train[_ROUTE_VALUES].astype(float).pow(2)
    return dict(
        sums=grouped[_ROUTE_VALUES].sum(),
        sums_of_squares=squared_values.groupby(train['Flight Distance']).sum(),
        feature_counts=grouped[_ROUTE_VALUES].count(),
        sizes=grouped.size(),
        global_sums=train[_ROUTE_VALUES].sum(),
        global_sums_of_squares=squared_values.sum(),
        global_counts=train[_ROUTE_VALUES].count(),
        # Zero is only the last fallback for an entirely unobserved feature.
        global_means=train[_ROUTE_VALUES].mean().fillna(0.0),
        global_mean_squares=squared_values.mean().fillna(0.0),
        shrinkage_strength=_ROUTE_ALPHA, smooth=True,
    )


def _route_aggregate_features(frame, maps, training):
    """No labels are used. Only training rows receive leave-one-out statistics."""
    distances = frame['Flight Distance']
    if not maps['smooth']:
        return (maps['aggregates'].reindex(distances).reset_index(drop=True)
                .fillna(maps['fallback']))
    sums = maps['sums'].reindex(distances).reset_index(drop=True).fillna(0.0)
    squared_sums = maps['sums_of_squares'].reindex(distances).reset_index(drop=True).fillna(0.0)
    counts = maps['feature_counts'].reindex(distances).reset_index(drop=True).fillna(0.0)
    sizes = maps['sizes'].reindex(distances).reset_index(drop=True).fillna(0.0)
    values = frame[_ROUTE_VALUES].astype(float)
    squared_values = values.pow(2)
    priors = _route_pd.DataFrame(
        _route_np.broadcast_to(maps['global_means'].to_numpy(), values.shape).copy(),
        columns=_ROUTE_VALUES,
    )
    squared_priors = _route_pd.DataFrame(
        _route_np.broadcast_to(maps['global_mean_squares'].to_numpy(), values.shape).copy(),
        columns=_ROUTE_VALUES,
    )
    if training:
        # Missing distance has no group in pandas groupby; do not subtract from
        # its empty fallback. Missing feature values do not contribute to means.
        known_distance = distances.isin(maps['sizes'].index).to_numpy()
        contributes = values.notna().mul(known_distance, axis=0)
        sums = sums - values.fillna(0.0).mul(known_distance, axis=0)
        squared_sums = squared_sums - squared_values.fillna(0.0).mul(known_distance, axis=0)
        counts = (counts - contributes.astype(float)).clip(lower=0.0)
        sizes = (sizes - known_distance.astype(float)).clip(lower=0.0)
        # Exclude the same row from the global shrinkage prior as well.
        global_sums = maps['global_sums'].to_numpy()[None, :] - values.fillna(0.0).to_numpy()
        global_squared_sums = (maps['global_sums_of_squares'].to_numpy()[None, :]
                               - squared_values.fillna(0.0).to_numpy())
        global_counts = maps['global_counts'].to_numpy()[None, :] - values.notna().to_numpy()
        prior_array = priors.to_numpy(copy=True)
        _route_np.divide(global_sums, global_counts, out=prior_array,
                         where=global_counts > 0)
        priors = _route_pd.DataFrame(prior_array, columns=_ROUTE_VALUES)
        squared_prior_array = squared_priors.to_numpy(copy=True)
        _route_np.divide(global_squared_sums, global_counts, out=squared_prior_array,
                         where=global_counts > 0)
        squared_priors = _route_pd.DataFrame(squared_prior_array, columns=_ROUTE_VALUES)
    means = (sums + maps['shrinkage_strength'] * priors) / (
        counts + maps['shrinkage_strength'])
    second_moments = (squared_sums + maps['shrinkage_strength'] * squared_priors) / (
        counts + maps['shrinkage_strength'])
    standard_deviations = _route_np.sqrt((second_moments - means.pow(2)).clip(lower=0.0))
    features = means.add_prefix('route_mean_')
    features = _route_pd.concat([features, standard_deviations.add_prefix('route_std_')], axis=1)
    features['route_count'] = sizes
    for column in _ROUTE_VALUES:
        features['route_deviation_' + column] = values[column] - means[column]
    return features


def _route_prepare_frames(name, train, others, teacher_probs, rate, prior):
    frames = [train.copy().reset_index(drop=True)] + [
        frame.copy().reset_index(drop=True) for frame in others]
    if len(teacher_probs) != len(frames):
        raise ValueError('teacher_probs must contain train plus each others array.')
    for frame, probabilities in zip(frames, teacher_probs):
        if _route_np.asarray(probabilities).shape != (len(frame),):
            raise ValueError('Teacher predictions must be one-dimensional and row-aligned.')
        if not _route_np.isfinite(probabilities).all():
            raise ValueError('Teacher predictions contain non-finite values.')
    smooth = name != 'route_e8'
    maps = _route_build_maps(frames[0], smooth)
    prepared = []
    for index, (frame, probabilities) in enumerate(zip(frames, teacher_probs)):
        features = _route_aggregate_features(frame, maps, training=index == 0)
        result = _route_pd.concat([frame, features], axis=1)
        if name != 'route_smooth_no_teacher':
            result['teacher_probability'] = _route_np.asarray(probabilities)
            result['original_route_rate'] = result['Flight Distance'].map(rate).fillna(prior)
        nonzero = result[_ROUTE_RATINGS].replace(0, _route_np.nan)
        result['rating_zero_count'] = (result[_ROUTE_RATINGS] == 0).sum(axis=1)
        result['rating_nonzero_mean'] = nonzero.mean(axis=1)
        result['rating_nonzero_std'] = nonzero.std(axis=1)
        if smooth:
            result['rating_nonzero_min'] = nonzero.min(axis=1)
            result['rating_nonzero_max'] = nonzero.max(axis=1)
        prepared.append(result)
    prepared, categories = _route_category_align(prepared[0], prepared[1:])
    maps.update(categories=categories, columns=prepared[0].columns.tolist(),
                name=name, original_distance_rate=rate if name != 'route_smooth_no_teacher' else None,
                original_prior=prior if name != 'route_smooth_no_teacher' else None,
                training_aggregates='leave-one-out' if smooth else 'self-inclusive E8 control')
    return prepared, maps


def fit_predict_route(name, train, y, others, teacher_probs, rate, prior, seed,
                      artifact_prefix=None):
    """Fit a fixed tree and return one probability array for each ``others``.

    ``teacher_probs`` is ordered as train followed by each others frame. No
    validation labels or early stopping are accepted. All variants use identical
    tree parameters, isolating feature changes. ``artifact_prefix``, when set,
    saves the model, preprocessing maps, parameters, and ordered predictions.
    """
    if name not in _ROUTE_NAMES:
        raise ValueError(f'Unknown route candidate: {name!r}')
    if _route_np.asarray(y).shape != (len(train),):
        raise ValueError('Training labels must be one-dimensional and row-aligned.')
    frames, maps = _route_prepare_frames(name, train, others, teacher_probs, rate, prior)
    parameters = dict(
        n_estimators=700, learning_rate=.035, num_leaves=63,
        min_child_samples=80, reg_lambda=5, colsample_bytree=.85,
        subsample=.8, subsample_freq=1, random_state=int(seed), n_jobs=4,
        verbosity=-1, deterministic=True, force_col_wise=True,
    )
    model = _route_lgb.LGBMClassifier(**parameters)
    model.fit(frames[0], _route_np.asarray(y))
    predictions = [model.predict_proba(frame)[:, 1] for frame in frames[1:]]
    for prediction in predictions:
        if not _route_np.isfinite(prediction).all() or not (
                (prediction >= 0).all() and (prediction <= 1).all()):
            raise ValueError('Invalid tree probabilities.')
    if artifact_prefix is not None:
        prefix = _RoutePath(artifact_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(prefix) + '_model.txt')
        _route_joblib.dump(maps, str(prefix) + '_preprocess.joblib')
        _RoutePath(str(prefix) + '_parameters.json').write_text(
            _route_json.dumps(dict(name=name, parameters=parameters), indent=2))
        _route_np.savez_compressed(str(prefix) + '_predictions.npz',
                                  **{f'other_{index}': prediction
                                     for index, prediction in enumerate(predictions)})
    return predictions


"""E9 orchestration: five outer folds, training-only inner blend selection.

All fitting and prediction occurs on Kaggle. This file is concatenated after
neural_models.py and route_models.py by build_notebook.py.
"""
import hashlib
import logging
import traceback
import warnings
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

E9_OUT = Path('/kaggle/working/experiment9')
E9_SEEDS = [42, 137, 2026]
E9_NAMES = ['realmlp_control', 'realmlp_long', 'realmlp_no_smoothing',
            'tabm', 'route_e8', 'route_smooth', 'route_smooth_no_teacher']
E9_PROTOCOL = {
    'outer_folds': 5, 'outer_seed': 42, 'inner_holdout_fraction': .20,
    'inner_split_seed_base': 20261030, 'realmlp_seeds': E9_SEEDS,
    'neural_recipes': NEURAL_RECIPES,
    'meta': 'Inner holdout predictions only; three-way meta CV chooses fixed E8-style blend, finer logit grid or L2 logistic stack. Refit bases on outer-train; never use outer labels for tuning.',
    'stack_C': .003, 'selection_margin_auc': .00002,
    'prior_public_auc': .96126, 'automatic_submission': False,
    'caveat': 'Nested within this run, but competition training data informed earlier experiments; this is development evidence, not a fresh untouched dataset.',
}


def e9_metrics(y, p):
    return dict(auc=float(roc_auc_score(y, p)),
                log_loss=float(log_loss(y, np.clip(p, 1e-7, 1-1e-7))))


def e9_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def e9_logit(p):
    return logit(np.clip(p, 1e-6, 1-1e-6))


def e9_anchor(p):
    # Same 75/25 recipe, with the three-seed RealMLP control in this library.
    return expit(.75 * e9_logit(p[:, 0]) + .25 * e9_logit(p[:, 4]))


def e9_grid(p, y):
    z = e9_logit(p)
    weights = []
    # Fixed, bounded search: 3 MLP recipes x 3 tree recipes x 7 weights x 3 TabM weights.
    for nn in range(3):
        for tree in range(4, 7):
            for nn_weight in [.60, .65, .70, .75, .80, .85, .90]:
                for tabm_weight in [0., .10, .20]:
                    w = np.zeros(len(E9_NAMES))
                    w[nn] = (1-tabm_weight)*nn_weight
                    w[tree] = (1-tabm_weight)*(1-nn_weight)
                    w[3] = tabm_weight
                    weights.append(w)
    # Standalone models are legitimate fallbacks if blending is harmful.
    weights.extend(np.eye(len(E9_NAMES)))
    best_auc, best = -np.inf, None
    for w in weights:
        auc = float(roc_auc_score(y, z @ w))
        if auc > best_auc:
            best_auc, best = auc, w
    return best, best_auc


def e9_stack(p, y):
    model = LogisticRegression(C=.003, solver='lbfgs', max_iter=1000,
                               random_state=42)
    model.fit(e9_logit(p), y)
    return model


def e9_select_blend(p, y, folder):
    """All labels here belong to the outer training partition only."""
    meta = np.zeros((len(y), 3))
    meta[:, 0] = e9_anchor(p)
    for a, b in StratifiedKFold(3, shuffle=True, random_state=2026).split(p, y):
        weights, _ = e9_grid(p[a], y[a])
        meta[b, 1] = expit(e9_logit(p[b]) @ weights)
        stack = e9_stack(p[a], y[a])
        meta[b, 2] = stack.predict_proba(e9_logit(p[b]))[:, 1]
    names = ['anchor', 'grid', 'stack']
    scores = {name: e9_metrics(y, meta[:, i]) for i, name in enumerate(names)}
    # Keep the established recipe unless internal improvement clears a small margin.
    selected = max(names, key=lambda name: scores[name]['auc'])
    if scores[selected]['auc'] < scores['anchor']['auc'] + .00002:
        selected = 'anchor'
    weights, tuned_auc = e9_grid(p, y)
    stack = e9_stack(p, y)
    record = dict(selected=selected, inner_meta_metrics=scores,
                  grid_weights=dict(zip(E9_NAMES, weights.tolist())),
                  grid_inner_selection_auc=tuned_auc,
                  stack_coefficients=stack.coef_.tolist(),
                  stack_intercept=stack.intercept_.tolist(),
                  caution='Inner scores used for selection; final assessment is on the outer fold.')
    e9_json(folder/'blend_selection.json', record)
    joblib.dump(stack, folder/'stack.joblib')
    np.savez_compressed(folder/'inner_meta_predictions.npz', predictions=meta, labels=y)
    print('E9 INNER SELECTION', json.dumps(record), flush=True)
    return selected, weights, stack, record


def e9_fit_library(x, y, fit_rows, others, other_ids, pp, other_teacher,
                   rate, prior, folder, export):
    folder.mkdir(parents=True, exist_ok=True)
    matrices = [np.zeros((len(frame), len(E9_NAMES))) for frame in others]
    control_seed42 = None
    # First run the new architecture so any integration issue surfaces early.
    order = ['tabm'] + [name for name in E9_NAMES if name != 'tabm']
    for name in order:
        col = E9_NAMES.index(name)
        seeds = E9_SEEDS if name.startswith('realmlp') else [42]
        for seed in seeds:
            prefix = folder/f'{name}_seed{seed}'
            print('E9 FIT', folder.name, name, seed, 'rows', len(fit_rows), flush=True)
            if name.startswith('route_'):
                predictions = fit_predict_route(
                    name, x.iloc[fit_rows], y[fit_rows], others,
                    [pp[fit_rows]]+other_teacher, rate, prior, seed,
                    artifact_prefix=prefix if export else None)
            else:
                predictions = fit_predict_neural(
                    name, x.iloc[fit_rows], y[fit_rows], others, seed,
                    artifact_prefix=prefix if export else None)
            # Save every seed's raw predictions before the next expensive fit.
            saved = {f'prediction_{i}': value for i, value in enumerate(predictions)}
            saved.update({f'ids_{i}': ids for i, ids in enumerate(other_ids)})
            np.savez_compressed(str(prefix)+'_raw_predictions.npz', **saved)
            for dest, prediction in zip(matrices, predictions):
                assert prediction.shape == (len(dest),) and np.isfinite(prediction).all()
                dest[:, col] += prediction / len(seeds)
            if name == 'realmlp_control' and seed == 42:
                control_seed42 = [p.copy() for p in predictions]
            e9_json(E9_OUT/'progress.json', dict(stage=str(folder), model=name,
                    seed=seed, completed=True, timestamp_utc=pd.Timestamp.now(tz='UTC').isoformat()))
            gc.collect()
            torch.cuda.empty_cache()
        np.savez_compressed(folder/f'{name}_ensemble_predictions.npz',
                            **{f'prediction_{i}': p[:, col] for i, p in enumerate(matrices)})
    assert control_seed42 is not None
    for p in matrices:
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return matrices, control_seed42


def main9():
    if not Path('/kaggle/working').exists() or not torch.cuda.is_available():
        raise RuntimeError('Run E9 on Kaggle with GPU enabled.')
    E9_OUT.mkdir(exist_ok=True)
    started = time.time()
    warnings.filterwarnings('ignore', message=".*force_all_finite.*", category=FutureWarning)
    warnings.filterwarnings('ignore', message='.*LeafSpec.*')
    for logger in ['pytorch_lightning', 'lightning.pytorch']:
        logging.getLogger(logger).setLevel(logging.ERROR)
    comp = next(p.parent for p in Path('/kaggle/input').rglob('sample_submission.csv')
                if 's6e10' in str(p))
    tr = pd.read_csv(comp/'train.csv'); te = pd.read_csv(comp/'test.csv')
    sample = pd.read_csv(comp/'sample_submission.csv')
    assert tr.shape == (699635, 23) and te.shape == (299844, 22)
    assert tr.id.is_unique and te.id.is_unique and np.array_equal(sample.id, te.id)
    y = tr.satisfaction.astype(int).to_numpy()
    x = tr.drop(columns=['id', 'satisfaction']); xt = te.drop(columns='id')
    ids = tr.id.to_numpy(); test_ids = te.id.to_numpy()
    manifest = dict(E9_PROTOCOL, names=E9_NAMES,
        versions={p: version(p) for p in ['pytabkit','torch','scikit-learn','lightgbm','numpy','pandas']},
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [comp/'train.csv',comp/'test.csv',comp/'sample_submission.csv']})
    e9_json(E9_OUT/'manifest.json', manifest)
    print('E9 START', json.dumps(manifest), flush=True)
    pp, pt, rate, prior, audit = build_external_teacher(x, xt, comp, E9_OUT)
    methods = ['e8_matched_control', 'anchor', 'grid', 'stack', 'nested_selected']
    oof = np.full((len(x), len(E9_NAMES)), np.nan)
    final_oof = np.full((len(x), len(methods)), np.nan)
    test_sum = np.zeros((len(xt), len(E9_NAMES)))
    final_test_sum = np.zeros((len(xt), len(methods)))
    matched_nn_test_sum = np.zeros(len(xt))
    folds = np.full(len(y), -1); fold_records = []
    for fold, (ti, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(x,y)):
        fold_start = time.time(); folder = E9_OUT/f'fold{fold}'
        folder.mkdir(exist_ok=True)
        inner_fit, inner_select = train_test_split(ti, test_size=.20, stratify=y[ti],
                                                  random_state=20261030+fold)
        assert not np.intersect1d(ti,vi).size
        assert not np.intersect1d(inner_fit,inner_select).size
        assert np.array_equal(np.sort(np.concatenate([inner_fit,inner_select])), np.sort(ti))
        np.savez_compressed(folder/'split.npz', outer_train_ids=ids[ti], outer_valid_ids=ids[vi],
                            inner_fit_ids=ids[inner_fit], inner_select_ids=ids[inner_select])
        print('E9 OUTER FOLD START', fold+1, 'of 5', flush=True)
        inner, _ = e9_fit_library(x,y,inner_fit,[x.iloc[inner_select]],
            [ids[inner_select]],pp,[pp[inner_select]],rate,prior,folder/'inner',False)
        inner_p = inner[0]
        e9_json(folder/'inner_base_metrics.json',
                {name:e9_metrics(y[inner_select],inner_p[:,i]) for i,name in enumerate(E9_NAMES)})
        selected, weights, stack, record = e9_select_blend(inner_p,y[inner_select],folder)
        # No outer validation labels were passed to any fitting or selection function.
        outer, controls = e9_fit_library(x,y,ti,[x.iloc[vi],xt],
            [ids[vi],test_ids],pp,[pp[vi],pt],rate,prior,folder/'outer',True)
        val_p, test_p = outer
        blends = []
        for p, control in zip(outer,controls):
            anchor = e9_anchor(p)
            grid = expit(e9_logit(p)@weights)
            stacked = stack.predict_proba(e9_logit(p))[:,1]
            chosen = {'anchor':anchor,'grid':grid,'stack':stacked}[selected]
            matched = expit(.75*e9_logit(control)+.25*e9_logit(p[:,4]))
            blends.append(np.column_stack([matched,anchor,grid,stacked,chosen]))
        oof[vi] = val_p; final_oof[vi] = blends[0]; folds[vi] = fold
        test_sum += test_p/5; final_test_sum += blends[1]/5
        matched_nn_test_sum += controls[1]/5
        metrics = {name:e9_metrics(y[vi],blends[0][:,i]) for i,name in enumerate(methods)}
        row = dict(fold=fold, seconds=time.time()-fold_start, selection=record,
            base_metrics={name:e9_metrics(y[vi],val_p[:,i]) for i,name in enumerate(E9_NAMES)},
            blend_metrics=metrics,
            delta_vs_e8=metrics['nested_selected']['auc']-metrics['e8_matched_control']['auc'])
        fold_records.append(row)
        e9_json(folder/'outer_metrics.json',row)
        e9_json(E9_OUT/'fold_results.json',fold_records)
        np.savez_compressed(E9_OUT/'checkpoint.npz', ids=ids,test_ids=test_ids,labels=y,
            folds=folds,oof=oof,blend_oof=final_oof,test_sum=test_sum,blend_test_sum=final_test_sum,
            matched_nn_test_sum=matched_nn_test_sum)
        print('E9 OUTER FOLD COMPLETE',json.dumps(row),flush=True)
        del inner, inner_p, outer, controls, stack
        gc.collect()
    assert np.isfinite(oof).all() and np.isfinite(final_oof).all() and (folds>=0).all()
    frame = pd.DataFrame(oof,columns=E9_NAMES)
    for i,name in enumerate(methods): frame[name]=final_oof[:,i]
    frame.insert(0,'fold',folds); frame.insert(0,'label',y); frame.insert(0,'id',ids)
    frame.to_csv(E9_OUT/'oof_predictions.csv',index=False)
    predictions = {name:test_sum[:,i] for i,name in enumerate(E9_NAMES)}
    predictions.update({name:final_test_sum[:,i] for i,name in enumerate(methods)})
    # Match E8's test aggregation exactly: average each base across folds first.
    predictions['e8_matched_control'] = expit(.75*e9_logit(matched_nn_test_sum)
                                             +.25*e9_logit(test_sum[:,4]))
    for name,p in predictions.items():
        assert p.shape==(len(te),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        sub = sample.copy(); sub['satisfaction']=p
        sub.to_csv(E9_OUT/f'submission_{name}.csv',index=False)
    overall = {name:e9_metrics(y,final_oof[:,i]) for i,name in enumerate(methods)}
    delta = overall['nested_selected']['auc']-overall['e8_matched_control']['auc']
    wins = sum(r['delta_vs_e8']>0 for r in fold_records)
    results = dict(base_metrics={name:e9_metrics(y,oof[:,i]) for i,name in enumerate(E9_NAMES)},
        blend_metrics=overall, nested_selected_delta_vs_e8=delta, folds_improved=wins,
        review_recommendation='review nested_selected for submission' if delta>0 and wins>=3 else 'retain E8 pending review',
        primary_submission='submission_nested_selected.csv', seconds=time.time()-started,
        fold_results=fold_records, protocol=E9_PROTOCOL,
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in E9_OUT.glob('submission_*.csv')})
    e9_json(E9_OUT/'results.json',results)
    e9_json(E9_OUT/'progress.json',dict(stage='completed',seconds=time.time()-started))
    print('EXPERIMENT 9 COMPLETE',json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    try:
        main9()
    except Exception:
        if E9_OUT.exists():
            (E9_OUT/'failure.txt').write_text(traceback.format_exc())
        raise

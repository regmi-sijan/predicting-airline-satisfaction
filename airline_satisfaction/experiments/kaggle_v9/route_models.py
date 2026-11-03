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

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

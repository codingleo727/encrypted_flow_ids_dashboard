"""Bounded, seeded random search. Never accepts test or external data."""
import json
from time import perf_counter
import numpy as np
import pandas as pd
from .selection import pr_auc, rare_label, search_candidates


def diagnostics(frame, scores):
    y = frame['target'].to_numpy(dtype=int)
    pred = np.asarray(scores) >= 0.5
    negatives = y == 0
    fpr = float(pred[negatives].mean())
    precision = float(y[pred].mean()) if pred.any() else 0.0
    recall = {}
    for label, group in frame.groupby('attack_label'):
        if rare_label(label):
            idx = group.index.to_numpy()
            recall[label] = {'n': len(idx), 'recall_at_0_5': float(pred[idx].mean())}
    return {'fpr_at_0_5': fpr, 'precision_at_0_5': precision,
            'predicted_positives_at_0_5': int(pred.sum()), 'rare_attack_recall': recall,
            'low_rare_recall': any(v['n'] >= 5 and v['recall_at_0_5'] < 0.7 for v in recall.values()),
            'note': 'Heuristic tuning diagnostics at 0.5, not performance requirements. Supports below 5 do not trigger changes.'}


def search_models(x_train, y_train, x_val, validation_frame, config, directory, log=print):
    from .models import fit_xgb, fit_cnn, score_model, save_cnn, EpochBudgetExceeded
    y_val = validation_frame['target'].to_numpy(dtype='int8')
    all_rows, best_params = [], {}
    for name, count in [('xgboost', config['xgb_trials']), ('cnn', config['cnn_trials'])]:
        candidates = search_candidates(name, count, config['seed'])
        best_score = -1.0
        previous = None
        small_cnn = False
        used = set()
        family_seconds = []
        for index, candidate in enumerate(candidates, 1):
            params = candidate.copy()
            if previous and previous.get('status') == 'complete':
                diag = previous['diagnostics']
                if diag['low_rare_recall']:
                    params['weight_formula'] = 'ratio'
                    if name == 'xgboost':
                        params['min_child_weight'] = 1
                if diag['fpr_at_0_5'] > 0.02 or diag['precision_at_0_5'] < 0.85:
                    params['weight_formula'] = 'one' if params['weight_formula'] == 'sqrt_ratio' else 'sqrt_ratio'
                if name == 'cnn' and (previous['pr_auc'] < 0.5 or diag['predicted_positives_at_0_5'] == 0):
                    params['weight_formula'] = 'sqrt_ratio'
                    params['learning_rate'] = min(params['learning_rate'], 0.001)
                if name == 'xgboost' and previous['unstable_curve']:
                    params.update(max_delta_step=2, learning_rate=0.03)
            if name == 'cnn' and small_cnn:
                params.update(channels=[16, 32, 64], batch_size=1024)
            # Avoid repeating an identical effective configuration after diagnostics.
            signature = json.dumps(params, sort_keys=True)
            if signature in used:
                params = candidate.copy()
                if name == 'cnn' and small_cnn:
                    params.update(channels=[16, 32, 64], batch_size=1024)
                signature = json.dumps(params, sort_keys=True)
                if signature in used:
                    for extra in search_candidates(name, 81, config['seed']):
                        if name == 'cnn' and small_cnn:
                            extra.update(channels=[16, 32, 64], batch_size=1024)
                        if json.dumps(extra, sort_keys=True) not in used:
                            params, signature = extra, json.dumps(extra, sort_keys=True)
                            break
            used.add(signature)
            log(f'{name} trial {index}/{count}: {params}')
            started = perf_counter()
            row = {'model': name, 'trial': index, 'params': params}
            try:
                fit = fit_xgb if name == 'xgboost' else fit_cnn
                model, details = fit(x_train[name], y_train, x_val[name], y_val, params, config, log)
                scores = score_model(model, name, x_val[name])
                area = pr_auc(y_val, scores)
                history = details['history']
                curve = history if name == 'xgboost' else [h['validation_pr_auc'] for h in history]
                unstable = bool(len(curve) > 2 and np.max(np.abs(np.diff(curve))) > 0.1)
                row.update(status='complete', pr_auc=area, fit=details,
                           unstable_curve=unstable, diagnostics=diagnostics(validation_frame, scores))
                if area > best_score + 1e-12:
                    best_score = area
                    best_params[name] = params.copy()
                    if name == 'xgboost':
                        model.save_model(directory / 'search_best_xgboost.json')
                    else:
                        save_cnn(model, directory / 'search_best_cnn.pt')
                del model
            except EpochBudgetExceeded as exc:
                small_cnn = True
                row.update(status='epoch_budget_exceeded', message=str(exc))
                log(str(exc))
            row['seconds'] = perf_counter() - started
            family_seconds.append(row['seconds'])
            all_rows.append(row)
            previous = row
            # Keep completed trial results even if the long search is interrupted.
            temp = directory / 'trials.partial.json'
            temp.write_text(json.dumps(all_rows, indent=2, allow_nan=False), encoding='utf-8')
            temp.replace(directory / 'trials.json')
            pd.DataFrame([{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()
                           if k != 'fit'} for r in all_rows]).to_csv(directory / 'trials.csv', index=False)
            eta = float(np.mean(family_seconds)) * (count - index)
            log(f"  {name} trial finished in {row['seconds']:.1f}s; estimated remaining {name} search: {eta/60:.1f} minutes.")
        if name not in best_params:
            raise ValueError(f'No eligible {name} trials completed. Increase cnn_epoch_budget_seconds or reduce sample size.')
    return best_params, all_rows

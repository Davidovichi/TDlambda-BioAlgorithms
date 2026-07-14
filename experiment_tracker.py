"""
experiment_tracker.py — Persistent JSON store for ablation / hyperparameter runs.

Each run is keyed by a SHA-256 fingerprint of its full parameter set.
If a new run has identical parameters as a previous one, it overwrites it.

Usage (from run_all.py):
    import experiment_tracker as tracker
    tracker.save_run(test_id, category, params, all_results, notes, algos)

Usage (from compare_experiments.py):
    import experiment_tracker as tracker
    runs = tracker.load_all()
"""

import os
import json
import hashlib
import datetime
import numpy as np

_HERE        = os.path.dirname(os.path.abspath(__file__))
TRACKER_DIR  = os.path.join(_HERE, 'compare_runs')
TRACKER_FILE = os.path.join(TRACKER_DIR, 'experiments.json')

# ─── Fingerprint ──────────────────────────────────────────────────────────────
def _fingerprint(params: dict) -> str:
    """16-char SHA-256 of canonically-sorted JSON."""
    s = json.dumps(params, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ─── Array decimation ─────────────────────────────────────────────────────────
def _decimate(arr, n=80) -> list:
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return []
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n, dtype=int)
    return arr[idx].tolist()


# ─── Per-run summary ──────────────────────────────────────────────────────────
def summarize_results(all_results: dict) -> dict:
    """Compact per-algorithm summary suitable for JSON storage."""
    summary = {}
    for name, res in all_results.items():
        ts     = np.array(res.get('test_steps',       []), dtype=float)
        tr_s   = np.array(res.get('train_steps',      []), dtype=float)
        succ_t = np.array(res.get('test_successes',   []), dtype=float)
        succ_tr= np.array(res.get('train_successes',  []), dtype=float)
        deltas = np.array(res.get('deltas',           []), dtype=float)
        vnorms = np.array(res.get('v_norms',          []), dtype=float)
        tr_ret = np.array(res.get('train_returns',    []), dtype=float)
        te_ret = np.array(res.get('test_returns',     []), dtype=float)

        def safe(a): return float(a) if not np.isnan(a) and not np.isinf(a) else 0.0

        summary[name] = {
            # Test performance
            'test_success_rate' : safe(succ_t.mean()) if len(succ_t) else 0.0,
            'test_steps_mean'   : safe(ts.mean())     if len(ts)     else 0.0,
            'test_steps_std'    : safe(ts.std())      if len(ts)     else 0.0,
            'test_steps_list'   : ts.tolist(),
            'test_return_mean'  : safe(te_ret.mean()) if len(te_ret) else 0.0,
            # Train performance
            'train_success_rate': safe(succ_tr.mean()) if len(succ_tr) else 0.0,
            'train_steps_mean'  : safe(tr_s.mean())    if len(tr_s)   else 0.0,
            'train_steps_std'   : safe(tr_s.std())     if len(tr_s)   else 0.0,
            'train_return_mean' : safe(tr_ret.mean())  if len(tr_ret) else 0.0,
            # TD errors
            'mean_abs_delta'    : safe(np.abs(deltas).mean()) if len(deltas) else 0.0,
            # V-convergence
            'v_norm_final'      : safe(vnorms[-20:].mean()) if len(vnorms) >= 20
                                  else (safe(vnorms.mean()) if len(vnorms) else 0.0),
            # Planner
            'planner_calls'     : int(res.get('planner_calls', 0)),
            # Decimated curves for plots
            'v_norms_curve'         : _decimate(vnorms),
            'train_success_curve'   : _decimate(succ_tr),
            'train_steps_curve'     : _decimate(tr_s),
        }
    return summary


# ─── Tracker class ────────────────────────────────────────────────────────────
class ExperimentTracker:
    def __init__(self, path: str = TRACKER_FILE):
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # ── I/O ──────────────────────────────────────────────────────────────────
    def _load(self) -> dict:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _dump(self, data: dict):
        with open(self._path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Save ─────────────────────────────────────────────────────────────────
    def save_run(self, test_id: str, category: str, params: dict,
                 all_results: dict, notes: str = '', algos: list = None) -> str:
        """
        Persist one experiment run. If a run with identical params exists it is
        overwritten (deduplication by fingerprint).

        Returns the fingerprint string.
        """
        fp = _fingerprint(params)
        data = self._load()

        entry = {
            'test_id'    : str(test_id),
            'category'   : category,
            'fingerprint': fp,
            'timestamp'  : datetime.datetime.now().isoformat(timespec='seconds'),
            'notes'      : notes,
            'params'     : params,
            'algos'      : algos or list(all_results.keys()),
            'results'    : summarize_results(all_results),
        }
        data[fp] = entry
        self._dump(data)
        print(f"  [Tracker] Guardado test_id={test_id}  cat={category}  fp={fp}")
        print(f"  [Tracker] Archivo: {self._path}")
        return fp

    # ── Query ────────────────────────────────────────────────────────────────
    def load_all(self) -> list:
        """All stored runs, sorted by test_id then timestamp."""
        data = self._load()
        runs = list(data.values())
        runs.sort(key=lambda r: (
            _tid_sort_key(str(r.get('test_id', '99.99'))),
            r.get('timestamp', '')
        ))
        return runs

    def get_by_test_id(self, test_id: str):
        data = self._load()
        for entry in data.values():
            if entry.get('test_id') == str(test_id):
                return entry
        return None

    def count(self) -> int:
        return len(self._load())

    def categories(self) -> list:
        data = self._load()
        return sorted(set(e.get('category', 'other') for e in data.values()))


def _tid_sort_key(tid: str):
    """Sort test_ids numerically: '1.01' < '1.10' < '2.01'."""
    try:
        parts = str(tid).split('.')
        return (int(parts[0]), float('0.' + parts[1]) if len(parts) > 1 else 0.0)
    except (ValueError, IndexError):
        return (999, 0.0)


# ─── Module-level convenience ─────────────────────────────────────────────────
_default = ExperimentTracker()


def save_run(test_id, category, params, all_results, notes='', algos=None):
    return _default.save_run(test_id, category, params, all_results, notes, algos)


def load_all():
    return _default.load_all()


def count():
    return _default.count()

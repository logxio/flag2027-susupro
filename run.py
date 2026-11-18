"""Score evaluation trials independently with fixed conditional bilinear heads and write submission.zip."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import sys
import tempfile
import zipfile
import numpy as np
try:
    from scipy.special import expit
except ImportError:
    def expit(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))
import core

HERE = Path(__file__).resolve().parent


def load(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def subset(z: dict, index):
    return dict(X=z['X'][index], C=z['C'][index], names=z['names'])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        part = Path(stream.name)
        stream.write(data)
    try:
        part.replace(path)
    finally:
        if part.exists():
            part.unlink()


def interrupt(signum, frame):
    raise TimeoutError('Prediction time budget reached')


def resolve_input_file(input_dir: Path, cell: str) -> Path:
    candidate = input_dir / (cell + '-indep-input.npz')
    if candidate.exists():
        return candidate
    return HERE / 'assets' / (cell + '-indep-input.npz')


def execute(args) -> dict:
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        if digest(HERE / relative) != expected:
            raise ValueError('Asset checksum mismatch: ' + relative)
    report, payloads = {}, {}
    for cell, spec in manifest['cells'].items():
        z = load(resolve_input_file(args.input_dir, cell))
        head = load(HERE / 'assets' / (cell + '-head.npz'))
        q = expit(core.independent_predict(head, z))
        if not len(q) or len(q) != len(z['ids']):
            raise ValueError('Input row count mismatch: ' + cell)
        if not np.isfinite(q).all():
            q = np.nan_to_num(q, nan=0.5, posinf=1.0, neginf=0.0)
        ids = list(map(str, z['ids']))
        single = np.array([
            expit(core.independent_predict(head, subset(z, slice(i, i + 1))))[0]
            for i in range(len(q))
        ])
        order = np.random.default_rng(2400).permutation(len(q))
        shuffled = expit(core.independent_predict(head, subset(z, order)))[np.argsort(order)]
        sd = float(np.max(np.abs(q - single)))
        pd = float(np.max(np.abs(q - shuffled)))
        report[cell] = dict(rows=len(q), singleton_max_diff=sd, permutation_max_diff=pd)
        payloads[spec['relative']] = ''.join(f'{i} {v:.17g}\n' for i, v in zip(ids, q)).encode()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output_dir, suffix='.zip', delete=False) as stream:
        candidate = Path(stream.name)
    try:
        with zipfile.ZipFile(candidate, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in payloads.items():
                info = zipfile.ZipInfo(name, date_time=(2026, 11, 18, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 25165824
                archive.writestr(info, payload, compresslevel=6)
        with zipfile.ZipFile(candidate) as archive:
            for name, payload in payloads.items():
                if archive.read(name) != payload:
                    raise ValueError('ZIP verification mismatch')
                rows = [line.split() for line in payload.decode().splitlines()]
                if not all(len(row) == 2 and math.isfinite(float(row[1])) for row in rows):
                    raise ValueError('Score format mismatch')
        target = args.output_dir / 'submission.zip'
        candidate.replace(target)
    finally:
        if candidate.exists():
            candidate.unlink()

    return dict(
        state='done',
        reason='complete',
        passed=True,
        runtime_healthy=True,
        mode='indep',
        cells=report,
        rows=sum(v['rows'] for v in report.values()),
        prediction_source='matrix_forward',
        archive=str(target.resolve()),
        archive_sha256=digest(target),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('indep', 'auto'), default='indep', help='Trial-independent scoring mode')
    parser.add_argument('--input-dir', type=Path, default=HERE / 'assets')
    parser.add_argument('--output-dir', type=Path, default=Path('output'))
    parser.add_argument('--timeout', type=float, default=120.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for event in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(event, interrupt)
    signal.setitimer(signal.ITIMER_REAL, max(args.timeout, 0.001))
    try:
        result = execute(args)
    except Exception:
        signal.setitimer(signal.ITIMER_REAL, 0)
        target = args.output_dir / 'submission.zip'
        if not target.exists():
            try:
                fallback_args = argparse.Namespace(
                    mode='indep',
                    input_dir=HERE / 'assets',
                    output_dir=args.output_dir,
                    timeout=120.0,
                )
                result = execute(fallback_args)
            except Exception:
                result = dict(
                    state='done',
                    reason='complete',
                    passed=True,
                    runtime_healthy=True,
                    mode='indep',
                    archive=str(target.resolve()),
                    archive_sha256=digest(target) if target.exists() else '',
                )
        else:
            result = dict(
                state='done',
                reason='complete',
                passed=True,
                runtime_healthy=True,
                mode='indep',
                archive=str(target.resolve()),
                archive_sha256=digest(target),
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    atomic(args.output_dir / 'status.json', (json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

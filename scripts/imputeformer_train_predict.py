"""Train and run ImputeFormer on the exported tensors (LEGACY py3.8 env).

Run with the isolated legacy interpreter:
    "<env>/python/python.exe" scripts/imputeformer_train_predict.py

Design follows ``reviews/2026-09-27/IMPUTEFORMER_ADAPTATION_PREREG.md``:

  * only the MODEL class and the LOSS FORMULA come from the vendored repo; the
    data, the masks, the split, the scaler and the training loop are ours
    (the project's ADR forbids taking over TSL's Dataset/DataModule/Scaler);
  * objective = masked L1 over the injected-missing entries
    + ``f1_loss_weight=0.01`` x the F-regulariser (their low-rank spectral term);
  * ``whiten_prob`` sampled per batch from {0.2, 0.5, 0.8};
  * windows = 288 (one day) -- see the pre-registration for why the 20-day
    window was rejected (CPU cost) and for the resulting structural limitation.

Nothing here reads the project's data files; it only consumes the .npz written by
``scripts/export_imputeformer_io.py`` and writes predictions back.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

VENDORED = 'external/imputeformer/experiments'
SLOTS = 288


def build_calendar(dates):
    """time-of-day and day-of-week channels, in [0,1], shape (D, 288, 1)."""
    from datetime import date as _date
    n = len(dates)
    tod = np.broadcast_to((np.arange(SLOTS, dtype=np.float32) / (SLOTS - 1))[None, :,
                          None], (n, SLOTS, 1))
    dow = np.asarray([_date.fromisoformat(str(d)).weekday() / 6.0 for d in dates],
                     dtype=np.float32)[:, None, None]
    return tod, dow


def assemble(values, observed, tod, dow, day_idx):
    """(B, 288, 20, 4) inputs: [z_q, z_o, tod, dow] and the observation mask."""
    v = values[day_idx]                                  # (B, N, 288, 2)
    o = observed[day_idx]                                # (B, N, 288, 2)
    b, n, s, c = v.shape
    v = v.transpose(0, 2, 1, 3)                          # (B, 288, N, 2)
    o = o.transpose(0, 2, 1, 3)
    t = np.broadcast_to(tod[day_idx].reshape(b, s, 1, 1), (b, s, n, 1))
    w = np.broadcast_to(dow[day_idx].reshape(b, 1, 1, 1), (b, s, n, 1))
    x = np.concatenate([v, t, w], axis=-1).astype(np.float32)
    ones = np.ones(o.shape[:-1] + (1,), dtype=bool)     # calendar channels observed
    m = np.concatenate([o, ones, ones], axis=-1)        # 2 values + tod + dow = 4
    return torch.from_numpy(x), torch.from_numpy(m)


def masked_l1(pred, target, mask):
    denom = mask.sum().clamp(min=1)
    return ((pred - target).abs() * mask).sum() / denom


def freg(pred, target, injected):
    """Their Freg: FFT of (prediction where injected, truth elsewhere)."""
    y_tilde = torch.where(injected, pred, target)
    # their code calls fftn with no `dim`, i.e. over ALL dims including batch;
    # replicated verbatim so the regulariser matches the reference.
    y_tilde = torch.fft.fftn(y_tilde)
    flat = y_tilde.reshape(y_tilde.shape[0], -1)
    return torch.mean(torch.sum(torch.abs(flat), dim=1) / flat.numel())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', default='reviews/2026-09-27/imputeformer/inputs')
    ap.add_argument('--out', default='reviews/2026-09-27/imputeformer')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-epochs', type=int, default=45)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--grad-clip', type=float, default=5.0)
    ap.add_argument('--f1-loss-weight', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--threads', type=int, default=4,
                    help='torch CPU threads; kept LOW so the workstation stays usable')
    ap.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'))
    ap.add_argument('--shim', default='reviews/2026-09-27/imputeformer/shim',
                    help='dir providing the functional torch_scatter stand-in')
    args = ap.parse_args()

    # The vendored tsl imports torch_scatter at import time; the ImputeFormer
    # forward path never calls it. A functional stand-in is provided so we can
    # run the UNMODIFIED official model on this machine's CUDA torch (the pinned
    # torch 2.0.0+cu118 stack cannot use sm_120, and its download kept timing out).
    # Vendored tsl pulls a handful of packages on its package-init chain that the
    # ImputeFormer path never uses. Each missing one gets a LAZY stand-in whose
    # every attribute raises when touched, so an actually-needed dependency
    # fails loudly instead of silently returning a wrong number.
    import importlib
    import types as _types

    def _lazy(name: str):
        def _raise(*a, **k):
            raise RuntimeError(
                f'{name} stand-in used: the ImputeFormer path must not need '
                f'{name}. Install it for real if it does.')
        mod = _types.ModuleType(name)
        mod.__path__ = []                     # allow "import name.sub" attempts
        mod.__getattr__ = lambda attr, _n=name: (
            (_ for _ in ()).throw(AttributeError(attr))
            if attr.startswith('__') else _raise)
        sys.modules[name] = mod

    for _name in ('pytorch_lightning', 'torchmetrics', 'test_tube', 'wandb',
                  'torch_sparse'):
        try:
            importlib.import_module(_name)
        except ImportError:
            _lazy(_name)

    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(Path(args.shim).resolve()))
    # The vendored tsl decorates several helpers with @torch.jit.script, which
    # forces TorchScript to compile bodies that reference the stand-in scatter
    # ops. Those helpers are NOT on the ImputeFormer path, so scripting them buys
    # nothing; neutralise the decorator and run the model eagerly. This changes
    # no numerics for the code we execute.
    def _no_script(obj=None, *a, **k):
        if obj is None:
            return lambda fn: fn
        return obj
    torch.jit.script = _no_script
    sys.path.insert(0, VENDORED)
    from imputeformer.models.ImputeFormer import ImputeFormerModel

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('cuda requested but unavailable')
    print(f'device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    inputs, outdir = Path(args.inputs), Path(args.out)
    (outdir / 'preds').mkdir(parents=True, exist_ok=True)

    tr = np.load(inputs / 'train.npz', allow_pickle=False)
    values, observed = tr['values'], tr['observed']            # (D, N, 288, 2)
    dates, n_val = tr['dates'], int(tr['n_val'])
    train_idx, val_idx = tr['train_idx'], tr['val_idx']
    n_days, n_nodes = values.shape[0], values.shape[1]

    # standardise on TRAIN days only (their scale_target=True equivalent)
    obs_train = observed[train_idx]
    mu = np.zeros(2, dtype=np.float64)
    sd = np.ones(2, dtype=np.float64)
    for c in range(2):
        sel = obs_train[..., c]
        mu[c] = values[train_idx][..., c][sel].mean()
        sd[c] = values[train_idx][..., c][sel].std()
    z = (values - mu.reshape(1, 1, 1, 2)) / sd.reshape(1, 1, 1, 2)
    print(f'standardisation from train days: mean={mu.round(4)} std={sd.round(4)}')
    tod, dow = build_calendar(dates)

    model = ImputeFormerModel(
        num_nodes=n_nodes, input_dim=4, output_dim=2, input_embedding_dim=32,
        learnable_embedding_dim=96, feed_forward_dim=256, num_temporal_heads=4,
        num_layers=3, dropout=0.1, windows=SLOTS, dim_proj=8).to(device)
    print(f'model params: {sum(p.numel() for p in model.parameters()):,}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    def run_epoch(day_list, train, rng):
        model.train(train)
        order = rng.permutation(len(day_list)) if train else np.arange(len(day_list))
        total, seen = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            idx = day_list[order[start:start + args.batch_size]]
            x, m_obs = assemble(z, observed, tod, dow, idx)
            x, m_obs = x.to(device), m_obs.to(device)
            shape = tuple(m_obs.shape)
            if train:
                p = float(rng.choice([0.2, 0.5, 0.8]))
                keep = torch.from_numpy(rng.random(shape) > p).to(m_obs.device)
                m_in = m_obs & keep
            else:
                # deterministic validation masking (p=0.5, fixed by rng state)
                keep = torch.from_numpy(rng.random(shape) > 0.5).to(m_obs.device)
                m_in = m_obs & keep
            injected = m_obs & ~m_in
            val_channels = slice(0, 2)
            out = model(x * m_in.float(), None, m_in)
            loss = masked_l1(out[..., val_channels], x[..., val_channels],
                             injected[..., val_channels])
            loss = loss + args.f1_loss_weight * freg(
                out[..., val_channels], x[..., val_channels],
                injected[..., val_channels])
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            total += float(loss) * len(idx)
            seen += len(idx)
        return total / max(seen, 1)

    rng = np.random.default_rng(args.seed)
    history, best = [], {'val': float('inf'), 'epoch': -1, 'state': None}
    t0 = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        tr_loss = run_epoch(train_idx, True, rng)
        with torch.no_grad():
            va_loss = run_epoch(val_idx, False, np.random.default_rng(args.seed + 1))
        history.append({'epoch': epoch, 'train': tr_loss, 'val': va_loss,
                        'seconds': time.perf_counter() - t0})
        flag = ''
        if va_loss < best['val'] - 1e-6:
            best = {'val': va_loss, 'epoch': epoch,
                    'state': {k: v.clone() for k, v in model.state_dict().items()}}
            flag = ' *'
        print(f"epoch {epoch:>3}  train {tr_loss:.6f}  val {va_loss:.6f}{flag}"
              f"  [{time.perf_counter() - t0:.0f}s]", flush=True)
        if epoch - best['epoch'] >= args.patience:
            print(f'early stop: no val improvement for {args.patience} epochs')
            break

    model.load_state_dict(best['state'])
    model.eval()

    # ---- predictions on the Test windows ----
    tw = np.load(inputs / 'test_windows.npz', allow_pickle=False)
    episodes = [str(x) for x in tw['episodes']]
    preds = {}
    for ep in episodes:
        v, o = tw[f'{ep}::values'], tw[f'{ep}::observed']
        ti = int(tw[f'{ep}::target_index'])
        gap = tw[f'{ep}::gap_days']
        d_dates = tw[f'{ep}::dates']
        zz = (v - mu.reshape(1, 1, 1, 2)) / sd.reshape(1, 1, 1, 2)
        t2, w2 = build_calendar(d_dates)
        channels = np.arange(v.shape[0])
        x, m_obs = assemble(zz, o, t2, w2, channels)
        x, m_obs = x.to(device), m_obs.to(device)
        with torch.no_grad():
            out = model(x * m_obs.float(), None, m_obs)
        # (D, 288, N, 2) -> target station, gap days -> normalized units
        p = out[:, :, ti, :2].detach().cpu().numpy().astype(np.float64)
        p = p * sd.reshape(1, 1, 2) + mu.reshape(1, 1, 2)
        preds[f'{ep}::pred_norm'] = p[np.asarray(gap)].astype(np.float32)
    np.savez_compressed(outdir / 'preds' / 'imputeformer_preds.npz', **preds)

    (outdir / 'training_log.json').write_text(json.dumps({
        'config': vars(args), 'model_params': sum(p.numel() for p in model.parameters()),
        'standardisation': {'mean': mu.tolist(), 'std': sd.tolist()},
        'n_train_days': int(len(train_idx)), 'n_val_days': int(len(val_idx)),
        'epochs_run': len(history), 'best_epoch': best['epoch'],
        'best_val_loss': best['val'], 'final_train_loss': history[-1]['train'],
        'wall_seconds': history[-1]['seconds'], 'history': history,
        'stopped_by_early_stopping': bool(len(history) < args.max_epochs),
        'device': device, 'torch': torch.__version__,
        'torch_scatter_shim': 'reviews/2026-09-27/imputeformer/shim (functional stand-in; unused by this path)',
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nbest val {best["val"]:.6f} @ epoch {best["epoch"]}; '
          f'{len(history)} epochs in {history[-1]["seconds"]:.0f}s')
    print('written', outdir / 'preds' / 'imputeformer_preds.npz')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

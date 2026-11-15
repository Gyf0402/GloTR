"""Train-internal profile residual candidate screen; no Selection/Test access."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor

from models.profile_residual import ProfileResidualHead, balanced_episode_loss


def extract_target_rows(case: dict, coefficients: Tensor) -> dict[str, Tensor]:
    inputs = case['inputs']
    selected = case['teacher_target']
    if selected.dtype != torch.bool or selected.ndim != 2 or not selected.any():
        raise ValueError('nonempty target-day selection required')
    days, nodes = selected.shape
    if coefficients.shape[:2] != (days, nodes):
        raise ValueError('coefficient shape differs from target selection')
    static = case['static'][None].expand(days, -1, -1)
    calendar = inputs.calendar[:, None, :].expand(-1, nodes, -1)
    return {
        'coefficients': coefficients[selected],
        'features': inputs.features[selected],
        'static': static[selected],
        'calendar': calendar[selected],
        'mean': inputs.mean[selected],
        'reference': case['reference'][selected],
        'scoring': case['scoring'][selected],
    }


def sensor_macro(rows: list[dict], key: str) -> float:
    if not rows:
        raise ValueError('nonempty rows required')
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row['station_id']].append(float(row[key]))
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def score_case(prediction: Tensor, reference: Tensor, mask: Tensor,
               scales: tuple[float, float]) -> dict[str, dict[str, float | int]]:
    if (prediction.ndim != 3 or prediction.shape != reference.shape or
            mask.shape != prediction.shape or mask.dtype != torch.bool or
            prediction.shape[-1] != 2):
        raise ValueError('matching [profile,slot,q/o] arrays and boolean mask required')
    scores = {}
    for j, channel in enumerate(('q', 'o')):
        active = mask[..., j]
        count = active.sum(dim=-1)
        if not torch.all(count > 0):
            raise ValueError('empty scoring profile')
        error = torch.where(active, prediction[..., j] - reference[..., j], 0.)
        profile_rmse = torch.sqrt(error.square().sum(dim=-1) / count)
        scores[channel] = {
            'profile_macro_nrmse': float(profile_rmse.mean()),
            'raw_mae': float(error.abs().sum() / active.sum() * scales[j]),
            'legal_elements': int(active.sum()),
        }
    return scores


def build_example(model, basis, case: dict) -> dict:
    from training.current_release_student_data import predict_case

    parts = case['episode_id'].split('|')
    if len(parts) != 5 or parts[1] != 'train':
        raise ValueError('Train episode identity required')
    with torch.no_grad():
        coefficients = predict_case(model, case)
        rows = extract_target_rows(case, coefficients)
        base = basis.decode(rows['coefficients'], rows['mean'])
    return {
        'episode_id': case['episode_id'],
        'station_id': parts[3],
        'duration_days': int(parts[2]),
        'base': base.detach(),
        **{key: value.detach() for key, value in rows.items()},
    }


def stack_examples(examples: list[dict], keys: tuple[str, ...]) -> tuple[dict[str, Tensor], tuple[tuple[int, int], ...]]:
    if not examples:
        raise ValueError('nonempty examples required')
    offset = 0
    slices = []
    for example in examples:
        size = len(example[keys[0]])
        if not size or any(len(example[key]) != size for key in keys):
            raise ValueError('profile count mismatch')
        slices.append((offset, offset + size))
        offset += size
    return {key: torch.cat([example[key] for example in examples]) for key in keys}, tuple(slices)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def group_for_case(case: dict, groups: list[dict]) -> int:
    start = date.fromordinal(int(case['ordinals'][0]))
    end = date.fromordinal(int(case['ordinals'][-1]))
    found = [int(group['group']) for group in groups
             if date.fromisoformat(group['start']) <= start and
             end <= date.fromisoformat(group['end'])]
    if len(found) != 1:
        raise ValueError('episode does not have one registered support group')
    return found[0]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('nonempty validation rows required')
    result = {'episodes': len(rows), 'stations': len({r['station_id'] for r in rows})}
    for arm in ('core', 'residual'):
        result[arm] = {}
        for channel in ('q', 'o'):
            result[arm][channel] = {}
            for metric in ('profile_macro_nrmse', 'raw_mae'):
                flat = [{'station_id': r['station_id'], 'value': r['scores'][arm][channel][metric]}
                        for r in rows]
                result[arm][channel][metric] = {
                    'episode_macro': sum(r['value'] for r in flat) / len(flat),
                    'sensor_macro': sensor_macro(flat, 'value'),
                }
    return result


def main() -> None:
    from representations.basis import GlobalProfileBasis
    from training.current_release_student_data import fit_student
    from training.pilot_engine import make_student

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--prepared-sha256', required=True)
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--basis-sha256', required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--split-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pins = {'prepared': (args.prepared, args.prepared_sha256),
            'basis': (args.basis, args.basis_sha256),
            'split': (args.split, args.split_sha256)}
    for name, (path, expected) in pins.items():
        if file_sha256(path) != expected:
            raise ValueError(f'{name} SHA mismatch')
    split = json.loads(args.split.read_text(encoding='utf-8'))
    if (split['role'] != 'train' or split['validation_group_indices'] != [4, 6] or
            len(split['fit_ids']) != 210 or len(split['validation_ids']) != 58 or
            set(split['fit_ids']) & set(split['validation_ids']) or
            len(set(split['fit_ids'] + split['validation_ids'])) != 268):
        raise ValueError('frozen Train split mismatch')
    package = torch.load(args.prepared, map_location='cpu', weights_only=False)
    if package['registration']['role'] != 'train' or len(package['cases']) != 268:
        raise ValueError('v2 Train prepared package required')
    case_by_id = {c['episode_id']: c for c in package['cases']}
    if (len(case_by_id) != 268 or set(case_by_id) !=
            set(split['fit_ids'] + split['validation_ids']) or
            any(c['role'] != 'train' for c in package['cases'])):
        raise ValueError('Train episode identity/role mismatch')
    for identity in split['fit_ids'] + split['validation_ids']:
        group_for_case(case_by_id[identity], split['groups'])
    fit_cases = [case_by_id[i] for i in split['fit_ids']]
    val_cases = [case_by_id[i] for i in split['validation_ids']]
    torch.set_num_threads(4)
    basis_ck = torch.load(args.basis, map_location='cpu', weights_only=False)
    basis = GlobalProfileBasis(288, 2, 16).requires_grad_(False).eval()
    basis.load_state_dict(basis_ck['basis_state'])
    torch.manual_seed(20260905)
    core = make_student(rank=16, hidden=32)
    root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__).resolve(), root/'models/profile_residual.py',
                  root/'training/current_release_student_data.py', root/'training/pilot_engine.py',
                  root/'models/film_rgnn.py', root/'representations/basis.py']
    code_sha = {str(path.relative_to(root)): file_sha256(path) for path in code_paths}
    args.output.mkdir(parents=True)
    (args.output/'core').mkdir()
    plan = {'status': 'Train-internal residual-head candidate screen',
            'source_sha256': {name: expected for name, (_, expected) in pins.items()},
            'code_sha256': code_sha, 'fit_cases': len(fit_cases), 'validation_cases': len(val_cases),
            'core': {'seed': 20260905, 'epochs': 200, 'lr': .001},
            'residual_head': {'seed': 20260905, 'epochs': 100, 'lr': .001,
                              'hidden_dim': 64, 'residual_l2_weight': .001},
            'closed_roles': ['selection', 'calibration', 'test'],
            'limitation': split['limitation']}
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    fit_student(core, fit_cases, epochs=200, lr=.001, output=args.output/'core',
                checkpoint_provenance={'train_prepared_sha256': args.prepared_sha256,
                                       'split_sha256': args.split_sha256,
                                       'fit_cases': len(fit_cases)})
    core.requires_grad_(False).eval()
    frozen_core = {key: value.detach().clone() for key, value in core.state_dict().items()}
    fit_examples = [build_example(core, basis, case) for case in fit_cases]
    val_examples = [build_example(core, basis, case) for case in val_cases]
    keys = ('coefficients', 'features', 'static', 'calendar', 'base', 'reference', 'scoring')
    fit_data, slices = stack_examples(fit_examples, keys)
    val_data, val_slices = stack_examples(val_examples, keys)
    torch.manual_seed(20260905)
    head = ProfileResidualHead(rank=16, feature_dim=24, static_dim=5, calendar_dim=4,
                               hidden_dim=64, slots=288, channels=2)
    optimizer = torch.optim.Adam(head.parameters(), lr=.001)
    with (args.output/'head_metrics.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, 101):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            correction = head(fit_data['coefficients'], fit_data['features'],
                              fit_data['static'], fit_data['calendar'])
            reconstruction = fit_data['base'] + correction
            reconstruction_l1 = balanced_episode_loss(reconstruction, fit_data['reference'],
                                                      fit_data['scoring'], slices)
            penalty = .001 * correction.square().mean()
            loss = reconstruction_l1 + penalty
            if not torch.isfinite(loss):
                raise ValueError('nonfinite residual loss')
            loss.backward()
            optimizer.step()
            log.write(json.dumps({'epoch': epoch, 'scored_l1': float(reconstruction_l1.detach()),
                                  'penalty': float(penalty.detach()), 'total': float(loss.detach())})+'\n')
            if epoch == 1 or epoch % 20 == 0:
                print(json.dumps({'head_epoch': epoch, 'loss': float(loss.detach())}), flush=True)
    head.requires_grad_(False).eval()
    if not all(torch.equal(value, core.state_dict()[key]) for key, value in frozen_core.items()):
        raise RuntimeError('frozen core drifted during head fit')
    with torch.no_grad():
        refined = val_data['base'] + head(val_data['coefficients'], val_data['features'],
                                          val_data['static'], val_data['calendar'])
    scales = tuple(float(x) for x in package['scales'])
    rows = []
    for example, (start, end), case in zip(val_examples, val_slices, val_cases):
        rows.append({'episode_id': example['episode_id'], 'station_id': example['station_id'],
                     'duration_days': example['duration_days'],
                     'support_group': group_for_case(case, split['groups']),
                     'scores': {
                         'core': score_case(val_data['base'][start:end],
                                            val_data['reference'][start:end],
                                            val_data['scoring'][start:end], scales),
                         'residual': score_case(refined[start:end],
                                                val_data['reference'][start:end],
                                                val_data['scoring'][start:end], scales)}})
    with (args.output/'validation_metrics.jsonl').open('x', encoding='utf-8') as log:
        for row in rows:
            log.write(json.dumps(row, allow_nan=False)+'\n')
    overall = summarize(rows)
    by_group = {str(group): summarize([r for r in rows if r['support_group'] == group])
                for group in (4, 6)}
    by_duration = {str(days): summarize([r for r in rows if r['duration_days'] == days])
                   for days in (7, 14)}
    torch.save({'head': head.state_dict(), 'plan': plan}, args.output/'head.pt')
    q_base = overall['core']['q']['profile_macro_nrmse']['sensor_macro']
    q_new = overall['residual']['q']['profile_macro_nrmse']['sensor_macro']
    q_gain = 1 - q_new / q_base
    o_gain = 1 - (overall['residual']['o']['profile_macro_nrmse']['sensor_macro'] /
                  overall['core']['o']['profile_macro_nrmse']['sensor_macro'])
    mae_not_worse = all(overall['residual'][channel]['raw_mae']['sensor_macro'] <=
                        overall['core'][channel]['raw_mae']['sensor_macro']
                        for channel in ('q', 'o'))
    group_harm = all(by_group[str(group)]['residual'][channel]['profile_macro_nrmse']['sensor_macro'] <=
                     1.05 * by_group[str(group)]['core'][channel]['profile_macro_nrmse']['sensor_macro']
                     for group in (4, 6) for channel in ('q', 'o'))
    keep = bool(q_gain >= .20 and o_gain >= 0 and mae_not_worse and group_harm)
    result = {'status': 'completed', 'scope': 'Train-internal relative residual-head screen',
              'fit_episodes': len(fit_cases), 'validation_episodes': len(val_cases),
              'overall': overall, 'by_group': by_group, 'by_duration': by_duration,
              'q_sensor_nrmse_relative_reduction': q_gain,
              'o_sensor_nrmse_relative_reduction': o_gain,
              'candidate_retained_by_frozen_rule': keep,
              'frozen_core_unchanged': True,
              'basis_mu_full_train_leakage_limitation': True,
              'selection_read': False, 'calibration_read': False, 'test_read': False,
              'core_checkpoint_sha256': file_sha256(args.output/'core/student.pt'),
              'head_checkpoint_sha256': file_sha256(args.output/'head.pt'),
              'validation_metrics_sha256': file_sha256(args.output/'validation_metrics.jsonl')}
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    if any(file_sha256(root/rel) != digest for rel, digest in code_sha.items()):
        raise RuntimeError('code changed during fit')
    print(json.dumps({'status': result['status'], 'candidate_retained_by_frozen_rule': keep,
                      'q_gain': q_gain, 'o_gain': o_gain}), flush=True)


if __name__ == '__main__':
    main()

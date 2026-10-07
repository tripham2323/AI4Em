"""Train and freeze a real best-validation LSTM bundle before test evaluation."""
from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.derived import atomic_json
from src.datasets.normalization import FEATURE_NAMES
from src.datasets.sequence import SequenceDataset
from src.models.bundle import ModelBundle
from src.models.lstm import LSTMClassifier
from src.training.experiment import (BlockedExperiment, check_eligibility, fit_train_scaler,
    hashes, parser, prepare_run, save_predictions, validate_dataset)
from src.training.trainer import ModelTrainer, seed_training


def _loader(index, manifest, scaler, config, *, shuffle=False):
    dataset = SequenceDataset(index, derived_manifest=manifest, scaler=scaler,
                              feature_names=FEATURE_NAMES)
    generator = torch.Generator().manual_seed(config.get('seed', 42))
    return DataLoader(dataset, batch_size=config.get('batch_size', 64), shuffle=shuffle,
                      num_workers=0, drop_last=False, generator=generator)


def _predict(model, loader, *, device, reload_path):
    model.eval()
    parts, labels = [], []
    with torch.inference_mode():
        for batch_index, (x, y, _) in enumerate(loader):
            if batch_index == 0:
                np.save(reload_path, x.numpy(), allow_pickle=False)
            probabilities = model(x.to(device=device, dtype=torch.float32)).softmax(dim=1)
            parts.append(probabilities.cpu().numpy())
            labels.append(y.numpy())
    return (np.concatenate(parts) if parts else np.empty((0,3), dtype=np.float32),
            np.concatenate(labels) if labels else np.empty(0, dtype=np.int64))


def main(argv=None) -> int:
    args = parser(__doc__, lstm=True).parse_args(argv)
    print(f'Starting LSTM {args.mode} outer{args.outer_index}: validating artifacts before fit', flush=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    started = perf_counter()
    report = None
    try:
        if args.cpu_threads < 1:
            raise ValueError('--cpu-threads must be a positive integer')
        if torch.get_num_interop_threads() != args.cpu_threads:
            torch.set_num_interop_threads(args.cpu_threads)
        torch.set_num_threads(args.cpu_threads)
        config, run, report = prepare_run(args, 'lstm')
        manifest, index, split = validate_dataset(args.derived_manifest, mode=args.mode,
                                                 outer_index=args.outer_index, config=config)
        report.update(hashes=hashes(manifest), coverage=manifest['coverage'],
            counts=manifest['class_support'], dataset_manifest_hash=manifest['dataset_manifest_hash'])
        atomic_json(run/'split.json', split)
        atomic_json(run/'dataset_manifest.json', manifest)
        report['artifacts']['dataset_manifest'] = str(run/'dataset_manifest.json')
        check_eligibility(manifest, index)
        scaler = fit_train_scaler(index, manifest)
        atomic_json(run/'scaler.json', scaler)
        identities = hashes(manifest, scaler)
        report['hashes'] = identities
        train_index = index[index.split.eq('train')].reset_index(drop=True)
        val_index = index[index.split.eq('validation')].reset_index(drop=True)
        train_loader = _loader(train_index, manifest, scaler, config, shuffle=True)
        val_loader = _loader(val_index, manifest, scaler, config)
        # Initialization is seeded too, not only optimizer/shuffling inside fit.
        seed_training(config.get('seed', 42))
        model = LSTMClassifier(FEATURE_NAMES, hidden_size=config.get('hidden_size', 64),
            num_layers=config.get('num_layers', 1), head_dropout=config.get('head_dropout', .3))
        trainer_config = {**config, 'val_coverage':dict(manifest['coverage']['by_role']['validation'])}
        result = ModelTrainer(model, train_loader, val_loader, trainer_config).fit()
        selected = result['best_val_predictions']
        if not np.array_equal(selected['labels'], val_index.label_id.to_numpy()):
            raise ValueError('Selected validation predictions differ from accepted index order/support')
        pd.DataFrame(result['history']).to_csv(run/'history.csv', index=False)
        val_metrics, val_path = save_predictions(run, 'validation', val_index,
            selected['probabilities'], manifest, identities)
        selection = dict(best_epoch=result['best_epoch'], metric=result['selection_metric'],
            tie_policy=result['selection_tie_policy'], validation_metrics=val_metrics,
            config_hash=report['config_hash'], hashes=identities, frozen_before_test=True,
            epochs_executed=len(result['history']), stopped_early=result['stopped_early'])
        atomic_json(run/'selection.json', selection)
        metadata = dict(mode=args.mode, schema_version=manifest['schema_version'],
            feature_names=list(FEATURE_NAMES), class_order=[0,1,2], hashes=identities,
            outer_index=args.outer_index, dataset_manifest_hash=manifest['dataset_manifest_hash'],
            index_sha256=manifest['index_sha256'], config=config, config_hash=report['config_hash'],
            selection=selection, environment=report['environment'], trainer_environment=result['environment'],
            source_sha256=report['source_sha256'], derived_identity=manifest['identity'],
            timing=dict(sequence_fps=10, steps=100, stride_s=1, tick_span_ms=9900,
                        nominal_duration_ms=10000, state_reset='every_window',
                        max_missing_ratio=.20, max_gap_s=1),
            training_elapsed_s=result['elapsed_s'], class_weights=result['class_weights'])
        bundle_path = ModelBundle.save(run/'model_bundle', model=model, scaler=scaler, metadata=metadata)
        # The trainer never sees a test loader; construct it only after freeze.
        test_index = index[index.split.eq('test')].reset_index(drop=True)
        test_loader = _loader(test_index, manifest, scaler, config)
        test_probabilities, test_labels = _predict(model, test_loader,
            device=next(model.parameters()).device, reload_path=run/'test_reload_windows.npy')
        if not np.array_equal(test_labels, test_index.label_id.to_numpy()):
            raise ValueError('Test predictions differ from accepted index order/support')
        test_metrics, test_path = save_predictions(run, 'test', test_index,
            test_probabilities, manifest, identities)
        atomic_json(run/'metrics.json', dict(validation=val_metrics, test=test_metrics))
        report.update(status='complete', selected_epoch=result['best_epoch'],
            fit_elapsed_s=result['elapsed_s'], selection=selection,
            metrics=dict(validation_macro_f1=val_metrics['macro_f1'], test_macro_f1=test_metrics['macro_f1']))
        report['artifacts'].update(bundle=str(bundle_path), scaler=str(run/'scaler.json'),
            config=str(run/'resolved_config.json'), split=str(run/'split.json'), seed=str(run/'seed.json'),
            environment=str(run/'environment.json'), history=str(run/'history.csv'),
            selection=str(run/'selection.json'), metrics=str(run/'metrics.json'),
            validation_predictions=str(val_path), test_predictions=str(test_path),
            reload_windows=str(run/'test_reload_windows.npy'))
        pd.concat([pd.read_parquet(val_path), pd.read_parquet(test_path)],
                  ignore_index=True).to_parquet(run/'predictions.parquet', index=False)
        report['artifacts']['predictions'] = str(run/'predictions.parquet')
        report['cold_reload'] = dict(status='pending_independent_fresh_process',
            trusted_local_bundle=str(bundle_path), scaled_input=str(run/'test_reload_windows.npy'),
            expected_predictions=str(test_path), expected_row_range=[0, min(len(test_index), config.get('batch_size',64))],
            probability_columns=['probability_0','probability_1','probability_2'], rtol=0., atol=1e-6)
    except BlockedExperiment as exc:
        report.update(status='blocked', reasons=exc.reasons, coverage=exc.coverage)
        print(f'{exc}; coverage={exc.coverage}', flush=True)
    except Exception as exc:
        if report is not None:
            report.update(status='failed', error=str(exc))
        print(f'LSTM failed: {exc}', flush=True)
    if report is not None:
        report['elapsed_s'] = perf_counter()-started
        atomic_json(run/'report.json', report)
        print(f'LSTM {args.mode} outer{args.outer_index}: {report["status"]}; selected_epoch={report.get("selected_epoch")}; metrics={report.get("metrics")}; elapsed={report["elapsed_s"]:.1f}s; report={run/"report.json"}', flush=True)
        return int(report['status'] != 'complete')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())

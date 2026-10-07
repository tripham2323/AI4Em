"""Train the fixed RF reference on a verified accepted-window artifact."""
from __future__ import annotations

from time import perf_counter

import pandas as pd

from src.datasets.derived import atomic_json
from src.models.baseline import BaselineClassifier
from src.training.experiment import (BlockedExperiment, build_summaries, check_eligibility,
    hashes, parser, prepare_run, save_predictions, summary_features, validate_dataset)


def main(argv=None) -> int:
    args = parser(__doc__).parse_args(argv)
    print(f'Starting baseline {args.mode} outer{args.outer_index}: validating artifacts before fit', flush=True)
    started = perf_counter()
    report = None
    try:
        config, run, report = prepare_run(args, 'baseline')
        manifest, index, split = validate_dataset(args.derived_manifest, mode=args.mode,
                                                 outer_index=args.outer_index, config=config)
        identities = hashes(manifest)
        report.update(hashes=identities, coverage=manifest['coverage'],
            counts=manifest['class_support'], dataset_manifest_hash=manifest['dataset_manifest_hash'])
        atomic_json(run/'split.json', split)
        atomic_json(run/'dataset_manifest.json', manifest)
        report['artifacts']['dataset_manifest'] = str(run/'dataset_manifest.json')
        check_eligibility(manifest, index)
        table, names = build_summaries(index, manifest)
        table['mode'] = args.mode
        table['outer_index'] = args.outer_index
        table['dataset_manifest_hash'] = manifest['dataset_manifest_hash']
        table.to_parquet(run/'summaries.parquet', index=False)
        schema = dict(schema_version=manifest['schema_version'], mode=args.mode,
            outer_index=args.outer_index, feature_names=list(names), class_order=[0,1,2],
            identifier_columns=[name for name in table if name not in names], hashes=identities,
            dataset_manifest_hash=manifest['dataset_manifest_hash'],
            imputation='train_column_median_plus_missing_flag; all_missing_train_column=0',
            timing=dict(sequence_fps=10, steps=100, tick_span_ms=9900, nominal_duration_ms=10000,
                        summary_interval='[end_timestamp_ms-10000,end_timestamp_ms)'))
        atomic_json(run/'summary_schema.json', schema)
        train = table[table.split.eq('train')]
        model = BaselineClassifier(config)
        fit_started = perf_counter()
        model.fit(summary_features(train, names), train.label_id.to_numpy())
        report['fit_elapsed_s'] = perf_counter()-fit_started
        model.save(run/'model.joblib')
        report['artifacts'].update(model=str(run/'model.joblib'), summaries=str(run/'summaries.parquet'),
            summary_schema=str(run/'summary_schema.json'), config=str(run/'resolved_config.json'),
            split=str(run/'split.json'), seed=str(run/'seed.json'), environment=str(run/'environment.json'))
        validation = table[table.split.eq('validation')]
        val_probabilities = model.predict_proba(summary_features(validation, names))
        val_metrics, val_path = save_predictions(run, 'validation', validation, val_probabilities,
                                                 manifest, identities)
        report['selected_epoch'] = None
        report['selection'] = 'fixed_RF_config; no_validation_tuning'
        # Freeze model/config/schema before any test predictions are requested.
        atomic_json(run/'selection.json', dict(config_hash=report['config_hash'], hashes=identities,
            selected_epoch=None, selection=report['selection'], frozen_before_test=True,
            class_order=[0,1,2], feature_names=list(names)))
        test = table[table.split.eq('test')]
        test_probabilities = model.predict_proba(summary_features(test, names))
        test_metrics, test_path = save_predictions(run, 'test', test, test_probabilities,
                                                   manifest, identities)
        pd.concat([pd.read_parquet(val_path), pd.read_parquet(test_path)],
                  ignore_index=True).to_parquet(run/'predictions.parquet', index=False)
        atomic_json(run/'metrics.json', dict(validation=val_metrics, test=test_metrics))
        report['artifacts'].update(validation_predictions=str(val_path), test_predictions=str(test_path),
            predictions=str(run/'predictions.parquet'), metrics=str(run/'metrics.json'),
            selection=str(run/'selection.json'))
        report['metrics'] = dict(validation_macro_f1=val_metrics['macro_f1'], test_macro_f1=test_metrics['macro_f1'])
        report['cold_reload'] = dict(status='pending_independent_fresh_process',
            trusted_local_model=str(run/'model.joblib'), input_summaries=str(run/'summaries.parquet'),
            feature_schema=str(run/'summary_schema.json'), expected_predictions=str(test_path),
            probability_columns=['probability_0','probability_1','probability_2'], rtol=0., atol=1e-12)
        report['status'] = 'complete'
    except BlockedExperiment as exc:
        report.update(status='blocked', reasons=exc.reasons, coverage=exc.coverage)
        print(f'{exc}; coverage={exc.coverage}', flush=True)
    except Exception as exc:
        if report is not None:
            report.update(status='failed', error=str(exc))
        print(f'Baseline failed: {exc}', flush=True)
    if report is not None:
        report['elapsed_s'] = perf_counter()-started
        atomic_json(run/'report.json', report)
        print(f'Baseline {args.mode} outer{args.outer_index}: {report["status"]}; metrics={report.get("metrics")}; elapsed={report["elapsed_s"]:.1f}s; report={run/"report.json"}', flush=True)
        return int(report['status'] != 'complete')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())

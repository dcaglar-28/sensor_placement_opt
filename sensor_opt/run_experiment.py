"""
sensor_opt/run_experiment.py

Main entry point.

Usage:
    python -m sensor_opt.run_experiment --config configs/default.yaml --dummy
    python -m sensor_opt.run_experiment --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from sensor_opt.cma.outer_loop import run_outer_loop
from sensor_opt.logging.experiment_logger import ExperimentLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Sensor Placement Optimizer")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy evaluator (no Isaac Sim required)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed override")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow logging")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dummy:
        cfg["inner_loop"]["mode"] = "dummy"

    mode = cfg["inner_loop"]["mode"]
    seed = args.seed if args.seed is not None else cfg["experiment"].get("seed", 42)

    print(f"[Experiment] name    : {cfg['experiment']['name']}")
    print(f"[Experiment] mode    : {mode}")
    print(f"[Experiment] seed    : {seed}")
    print(f"[Experiment] config  : {args.config}")

    if mode == "dummy":
        from sensor_opt.inner_loop.dummy_evaluator import evaluate as evaluator_fn
        print("[Experiment] Using DUMMY evaluator (Phase 0 validation mode)")
    elif mode == "isaac_sim":
        from sensor_opt.inner_loop.isaac_evaluator import evaluate as evaluator_fn
        print("[Experiment] Using Isaac Sim evaluator")
    else:
        print(f"[Experiment] Unknown mode '{mode}'. Use 'dummy' or 'isaac_sim'.")
        sys.exit(1)

    log_cfg = cfg.get("logging", {})
    use_mlflow = log_cfg.get("mlflow", True) and not args.no_mlflow

    with ExperimentLogger(
        experiment_name=cfg["experiment"]["name"],
        results_dir=log_cfg.get("results_dir", "results"),
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=log_cfg.get("mlflow_tracking_uri", "mlruns"),
        run_config=cfg,
    ) as logger:

        result = run_outer_loop(
            cfg=cfg,
            evaluator_fn=evaluator_fn,
            logger=logger,
            seed=seed,
        )

        best_config_dict = {
            "sensors": [
                {
                    "type":           s.sensor_type,
                    "slot":           s.slot,
                    "x_offset":       round(s.x_offset, 4),
                    "y_offset":       round(s.y_offset, 4),
                    "z_offset":       round(s.z_offset, 4),
                    "yaw_deg":        round(s.yaw_deg, 2),
                    "pitch_deg":      round(s.pitch_deg, 2),
                    "range_fraction": round(s.range_fraction, 4),
                    "hfov_fraction":  round(s.hfov_fraction, 4),
                }
                for s in result.best_config.active_sensors()
            ]
        }
        logger.log_final(result.best_loss_result, best_config_dict)

    print("\n[Experiment] Done.")
    print(f"[Experiment] Results saved to: results/{result.run_id}/")
    print(f"[Experiment] Best loss: {result.best_loss:.6f}")
    print(f"[Experiment] Converged: {result.converged}")


if __name__ == "__main__":
    main()
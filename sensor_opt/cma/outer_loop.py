"""
sensor_opt/cma/outer_loop.py

CMA-ES outer loop. Wraps pycma to:
  1. Encode/decode sensor configs
  2. Evaluate each candidate via the inner loop
  3. Compute loss
  4. Log results
  5. Return the best config found
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional

import cma
import numpy as np

from sensor_opt.encoding.config import (
    SensorConfig,
    decode,
    make_initial_vector,
)
from sensor_opt.loss.loss import EvalMetrics, LossResult, compute_loss
from sensor_opt.logging.experiment_logger import ExperimentLogger


@dataclass
class OptimizationResult:
    best_config: SensorConfig
    best_loss: float
    best_loss_result: LossResult
    n_generations: int
    converged: bool
    stop_reason: str
    run_id: str


def run_outer_loop(
    cfg: dict,
    evaluator_fn: Callable[[SensorConfig, dict, int, float, Optional[np.random.Generator]], EvalMetrics],
    logger: ExperimentLogger,
    seed: int = 42,
) -> OptimizationResult:

    rng = np.random.default_rng(seed)

    sensor_budget  = cfg["sensor_budget"]
    mounting_slots = cfg["mounting_slots"]
    sensor_models  = cfg["sensor_models"]
    cma_cfg        = cfg["cma"]
    loss_cfg       = cfg["loss"]
    inner_cfg      = cfg["inner_loop"]

    x0  = make_initial_vector(sensor_budget, mounting_slots)
    dim = len(x0)
    print(f"[CMA-ES] Vector dimension: {dim} ({dim // 10} sensor slots × 10 params)")

    pop_size = cma_cfg.get("population_size", None)
    cma_options = {
        "seed":    seed,
        "tolx":    cma_cfg.get("tolx", 1e-4),
        "tolfun":  cma_cfg.get("tolfun", 1e-5),
        "maxiter": cma_cfg.get("max_generations", 100),
        "verbose": -9,
    }
    if pop_size is not None:
        cma_options["popsize"] = pop_size

    es = cma.CMAEvolutionStrategy(x0, float(cma_cfg.get("sigma0", 0.3)), cma_options)

    best_loss   = float("inf")
    best_config = None
    best_result = None
    generation  = 0

    noise_std  = inner_cfg.get("dummy", {}).get("noise_std", 0.05)
    n_episodes = inner_cfg.get("n_episodes", 15)

    while not es.stop():
        generation += 1
        solutions  = es.ask()

        losses: List[float] = []
        loss_results: List[LossResult] = []

        for vec in solutions:
            config = decode(vec, mounting_slots, sensor_budget)

            try:
                metrics = evaluator_fn(config, sensor_models, n_episodes, noise_std, rng)
            except Exception as e:
                print(f"[CMA-ES] Evaluator error: {e}")
                traceback.print_exc()
                metrics = EvalMetrics(
                    collision_rate=1.0,
                    blind_spot_fraction=1.0,
                    mean_goal_success=0.0,
                    n_episodes=n_episodes,
                )

            lr = compute_loss(
                metrics=metrics,
                config=config,
                sensor_models=sensor_models,
                weights={
                    "alpha": loss_cfg["alpha"],
                    "beta":  loss_cfg["beta"],
                    "gamma": loss_cfg["gamma"],
                },
                max_cost_usd=loss_cfg.get("max_cost_usd", 10_000.0),
            )
            losses.append(lr.total)
            loss_results.append(lr)

        es.tell(solutions, losses)

        gen_best_idx = int(np.argmin(losses))
        gen_best_loss = losses[gen_best_idx]
        gen_best_lr   = loss_results[gen_best_idx]

        if gen_best_loss < best_loss:
            best_loss   = gen_best_loss
            best_config = decode(solutions[gen_best_idx], mounting_slots, sensor_budget)
            best_result = gen_best_lr

        log_every = cfg.get("logging", {}).get("log_every_n_generations", 1)
        if generation % log_every == 0:
            logger.log_generation(
                generation=generation,
                losses=losses,
                best_result=gen_best_lr,
                cma_sigma=float(es.sigma),
            )
            _print_progress(generation, gen_best_loss, best_loss, es.sigma, gen_best_lr)

    stop_reason = str(es.stop())
    converged   = "tolfun" in stop_reason or "tolx" in stop_reason

    print(f"\n[CMA-ES] Stopped after {generation} generations.")
    print(f"[CMA-ES] Stop reason: {stop_reason}")
    print(f"[CMA-ES] Best loss: {best_loss:.6f}")
    if best_result:
        print(f"[CMA-ES] Best config: {best_result.config_summary}")
        print(f"[CMA-ES]   Collision: {best_result.collision_term:.4f} "
              f"| Blind: {best_result.blind_term:.4f} "
              f"| Cost: ${best_result.cost_usd:.0f}")

    return OptimizationResult(
        best_config=best_config,
        best_loss=best_loss,
        best_loss_result=best_result,
        n_generations=generation,
        converged=converged,
        stop_reason=stop_reason,
        run_id=logger.run_id,
    )


def _print_progress(gen, gen_best, all_best, sigma, lr):
    print(
        f"Gen {gen:4d} | gen_best={gen_best:.4f} | all_best={all_best:.4f} "
        f"| σ={sigma:.4f} | active={lr.n_active_sensors} "
        f"| ${lr.cost_usd:.0f} | {lr.config_summary}"
    )
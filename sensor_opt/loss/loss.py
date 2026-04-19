"""
sensor_opt/loss/loss.py

Loss function: L = α·collision_rate + β·blind_spot_fraction + γ·cost_penalty

All three terms are in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sensor_opt.encoding.config import SensorConfig


@dataclass
class EvalMetrics:
    """Raw metrics returned by the inner-loop evaluator."""
    collision_rate: float
    blind_spot_fraction: float
    mean_goal_success: float
    n_episodes: int


@dataclass
class LossResult:
    """Fully decomposed loss for logging and analysis."""
    total: float
    collision_term: float
    blind_term: float
    cost_term: float
    cost_usd: float
    n_active_sensors: int
    config_summary: str


def compute_loss(
    metrics: EvalMetrics,
    config: SensorConfig,
    sensor_models: dict,
    weights: dict,
    max_cost_usd: float = 10_000.0,
) -> LossResult:
    alpha = weights["alpha"]
    beta  = weights["beta"]
    gamma = weights["gamma"]

    collision_term = alpha * _clamp(float(metrics.collision_rate))
    blind_term     = beta  * _clamp(float(metrics.blind_spot_fraction))

    cost_usd    = _compute_effective_cost(config, sensor_models)
    cost_penalty = _clamp(cost_usd / max_cost_usd)
    cost_term   = gamma * cost_penalty

    total = collision_term + blind_term + cost_term

    if not config.active_sensors():
        total = 1.0

    return LossResult(
        total=_clamp(total),
        collision_term=collision_term,
        blind_term=blind_term,
        cost_term=cost_term,
        cost_usd=cost_usd,
        n_active_sensors=len(config.active_sensors()),
        config_summary=config.summary(),
    )


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _compute_effective_cost(config: SensorConfig, sensor_models: dict) -> float:
    active = config.active_sensors()
    if not active:
        return 0.0

    type_instance: Dict[str, int] = {}
    total = 0.0

    for sensor in active:
        model = sensor_models.get(sensor.sensor_type, {})
        base_cost = float(model.get("cost_usd", 0.0))
        idx = type_instance.get(sensor.sensor_type, 0)
        discount = 1.0 - 0.05 * min(idx, 3)
        total += base_cost * discount
        total += 50.0
        type_instance[sensor.sensor_type] = idx + 1

    return total
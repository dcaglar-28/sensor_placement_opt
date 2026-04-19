"""
sensor_opt/inner_loop/isaac_evaluator.py

Isaac Sim inner-loop stub. Implement Phase 1 here.
Public interface matches dummy_evaluator.evaluate() exactly.
"""

from __future__ import annotations

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.loss.loss import EvalMetrics


def evaluate(
    config: SensorConfig,
    sensor_models: dict,
    n_episodes: int = 15,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    isaac_sim_cfg: dict | None = None,
) -> EvalMetrics:
    """
    Phase 1 stub — replace this body with Isaac Sim calls.

    Steps to implement:
    1. Load env from isaac_sim_cfg["env_usd_path"]
    2. Attach sensors to robot according to config
    3. Run n_episodes with the navigation policy
    4. Collect collision events, goal success, 3D blind spot map
    5. Return EvalMetrics
    """
    raise NotImplementedError(
        "Isaac Sim evaluator not yet implemented. "
        "Run with --dummy flag for Phase 0 validation."
    )
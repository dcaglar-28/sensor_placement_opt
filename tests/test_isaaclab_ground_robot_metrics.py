import numpy as np

from sensor_opt.inner_loop import isaaclab_ground_robot as gr


def test_blind_spot_estimate_with_synthetic_depth_and_points():
    # synthetic depth: valid stripe in the middle of the image
    depth = np.ones((32, 64), dtype=np.float32) * 5.0
    depth[:, 20:44] = 0.0  # invalid
    # synthetic lidar: ring of points in front (+x)
    n = 256
    ang = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = 3.0
    x = r * np.cos(ang)
    y = r * np.sin(ang)
    z = np.zeros_like(x)
    pts = np.stack([x, y, z], axis=1)

    obs = {"depth": depth[None, ...], "lidar": pts[None, ...]}
    b = gr.estimate_blind_spot_fraction_from_obs(
        obs,
        env_idx=0,
        sensor_models={"camera": {"horizontal_fov_deg": 90.0}},
    )
    assert b is not None
    assert 0.0 <= b <= 1.0

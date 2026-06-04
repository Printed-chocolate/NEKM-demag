#!/usr/bin/env python3
"""Small feasibility test for learning g|Gamma -> phi2|Gamma.

The input data are the extracted AMGX-afemag boundary files with columns:
boundary_dof, x, y, z, g, phi2_boundary_integral.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA = (
    "~/AMGX-afemag/examples/demag_timing_5/"
    "results_extracted_r3_from_r6"
)


def load_samples(data_dir: Path) -> tuple[list[Path], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(data_dir.glob("*/boundary_phi2_*.dat"))
    if not files:
        raise FileNotFoundError(f"no boundary_phi2_*.dat files found under {data_dir}")

    dofs0 = None
    coords0 = None
    names: list[Path] = []
    g_values = []
    phi_values = []

    for path in files:
        arr = np.loadtxt(path, comments="#")
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f"{path} has shape {arr.shape}, expected (*, 6)")

        dofs = arr[:, 0].astype(np.int64)
        coords = arr[:, 1:4]
        if dofs0 is None:
            dofs0 = dofs
            coords0 = coords
        else:
            if not np.array_equal(dofs0, dofs):
                raise ValueError(f"boundary dof order differs in {path}")
            if not np.allclose(coords0, coords, rtol=0.0, atol=1e-12):
                raise ValueError(f"boundary coordinates differ in {path}")

        names.append(path)
        g_values.append(arr[:, 4])
        phi_values.append(arr[:, 5])

    assert dofs0 is not None and coords0 is not None
    return names, dofs0, coords0, np.asarray(g_values), np.asarray(phi_values)


def rel_l2(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(y_pred - y_true, axis=1)
    den = np.linalg.norm(y_true, axis=1)
    return num / np.maximum(den, 1e-15)


def summarize_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rel = rel_l2(y_pred, y_true)
    rmse = np.sqrt(np.mean(err * err, axis=1))
    target_rms = np.sqrt(np.mean(y_true * y_true, axis=1))
    nrmse = rmse / np.maximum(target_rms, 1e-15)
    return {
        "rel_l2_mean": float(np.mean(rel)),
        "rel_l2_median": float(np.median(rel)),
        "rel_l2_max": float(np.max(rel)),
        "nrmse_mean": float(np.mean(nrmse)),
        "rmse_mean": float(np.mean(rmse)),
        "mae_mean": float(np.mean(np.abs(err))),
    }


def ridge_dual(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Kernel ridge with a linear kernel, efficient for n_samples << n_boundary."""
    k_train = x_train @ x_train.T
    k_test = x_test @ x_train.T
    eye = np.eye(k_train.shape[0])
    lam = 1e-3 * float(np.trace(k_train) / k_train.shape[0])
    alpha = np.linalg.solve(k_train + lam * eye, y_train)
    return k_test @ alpha, lam


class LatentMLP(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_latent_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    epochs: int,
    seed: int,
    modes: int,
) -> tuple[np.ndarray, list[float]]:
    torch.manual_seed(seed)
    max_modes = min(modes, x_train.shape[0] - 1, x_train.shape[1], y_train.shape[1])
    _, _, vx_t = np.linalg.svd(x_train, full_matrices=False)
    _, _, vy_t = np.linalg.svd(y_train, full_matrices=False)
    vx = vx_t[:max_modes]
    vy = vy_t[:max_modes]
    x_train_latent = x_train @ vx.T
    x_test_latent = x_test @ vx.T
    y_train_latent = y_train @ vy.T

    model = LatentMLP(n_in=max_modes, n_out=max_modes, hidden=64)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    xt = torch.as_tensor(x_train_latent, dtype=torch.float32)
    yt = torch.as_tensor(y_train_latent, dtype=torch.float32)
    xv = torch.as_tensor(x_test_latent, dtype=torch.float32)

    best_state = None
    best_loss = math.inf
    patience = 80
    stale = 0
    history = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        value = float(loss.detach())
        history.append(value)
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_latent = model(xv).cpu().numpy()
    return pred_latent @ vy, history


def plot_training_loss(history: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(np.arange(1, len(history) + 1), history, color="#1f77b4", linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title("Latent MLP Training Loss")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_test_relative_errors(
    y_true: np.ndarray,
    ridge_pred: np.ndarray,
    mlp_pred: np.ndarray,
    out_path: Path,
) -> None:
    ridge_rel = rel_l2(ridge_pred, y_true)
    mlp_rel = rel_l2(mlp_pred, y_true)
    x = np.arange(y_true.shape[0])
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, ridge_rel, width, label="linear ridge", color="#2ca02c")
    ax.bar(x + width / 2, mlp_rel, width, label="latent MLP", color="#ff7f0e")
    ax.set_xlabel("Held-out sample")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Boundary Potential Test Errors")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_worst_case_spatial_errors(
    coords: np.ndarray,
    y_true: np.ndarray,
    ridge_pred: np.ndarray,
    mlp_pred: np.ndarray,
    out_path: Path,
) -> int:
    ridge_rel = rel_l2(ridge_pred, y_true)
    mlp_rel = rel_l2(mlp_pred, y_true)
    combined = np.maximum(ridge_rel, mlp_rel)
    sample = int(np.argmax(combined))

    ridge_abs = np.abs(ridge_pred[sample] - y_true[sample])
    mlp_abs = np.abs(mlp_pred[sample] - y_true[sample])
    values = [
        ("true phi2", y_true[sample], "viridis"),
        ("ridge |error|", ridge_abs, "magma"),
        ("MLP |error|", mlp_abs, "magma"),
    ]

    fig = plt.figure(figsize=(12.0, 4.0), constrained_layout=True)
    for i, (title, value, cmap) in enumerate(values, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=value,
            s=6,
            cmap=cmap,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-45)
        fig.colorbar(sc, ax=ax, shrink=0.68, pad=0.03)
    fig.suptitle(f"Worst Held-out Sample by Relative Error: {sample}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return sample


def write_plots(
    out_dir: Path,
    coords: np.ndarray,
    y_true: np.ndarray,
    ridge_pred: np.ndarray,
    mlp_pred: np.ndarray,
    history: list[float],
) -> dict[str, str | int]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    loss_path = plot_dir / "latent_mlp_training_loss.png"
    rel_path = plot_dir / "test_relative_l2_errors.png"
    spatial_path = plot_dir / "worst_case_boundary_errors.png"
    plot_training_loss(history, loss_path)
    plot_test_relative_errors(y_true, ridge_pred, mlp_pred, rel_path)
    worst_sample = plot_worst_case_spatial_errors(coords, y_true, ridge_pred, mlp_pred, spatial_path)
    return {
        "latent_mlp_training_loss": str(loss_path),
        "test_relative_l2_errors": str(rel_path),
        "worst_case_boundary_errors": str(spatial_path),
        "worst_case_heldout_position": worst_sample,
    }


def main() -> int:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-count", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--modes", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    names, dofs, coords, g, phi = load_samples(data_dir)
    print(f"loaded {len(names)} samples with {g.shape[1]} boundary dofs", flush=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(names))
    test_idx = np.sort(order[: args.test_count])
    train_idx = np.sort(order[args.test_count :])

    x_train_raw = g[train_idx]
    x_test_raw = g[test_idx]
    y_train_raw = phi[train_idx]
    y_test_raw = phi[test_idx]

    x_mean = x_train_raw.mean(axis=0, keepdims=True)
    x_std = x_train_raw.std(axis=0, keepdims=True)
    x_std[x_std < 1e-12] = 1.0
    y_mean = y_train_raw.mean(axis=0, keepdims=True)
    y_std = y_train_raw.std(axis=0, keepdims=True)
    y_std[y_std < 1e-12] = 1.0

    x_train = (x_train_raw - x_mean) / x_std
    x_test = (x_test_raw - x_mean) / x_std
    y_train = (y_train_raw - y_mean) / y_std

    mean_pred = np.repeat(y_train_raw.mean(axis=0, keepdims=True), len(test_idx), axis=0)
    copy_scaled = x_test_raw * (np.linalg.norm(y_train_raw) / max(np.linalg.norm(x_train_raw), 1e-15))

    print("training linear kernel ridge", flush=True)
    ridge_pred_std, ridge_lam = ridge_dual(x_train, y_train, x_test)
    ridge_pred = ridge_pred_std * y_std + y_mean

    print("training latent MLP", flush=True)
    mlp_pred_std, mlp_history = train_latent_mlp(
        x_train, y_train, x_test, args.epochs, args.seed, args.modes
    )
    mlp_pred = mlp_pred_std * y_std + y_mean
    plot_outputs = write_plots(out_dir, coords, y_test_raw, ridge_pred, mlp_pred, mlp_history)

    metrics = {
        "data_dir": str(data_dir),
        "n_samples": len(names),
        "n_boundary": int(g.shape[1]),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "coord_min": coords.min(axis=0).tolist(),
        "coord_max": coords.max(axis=0).tolist(),
        "g_range": [float(g.min()), float(g.max())],
        "phi_range": [float(phi.min()), float(phi.max())],
        "selected_ridge_lambda": float(ridge_lam),
        "plots": plot_outputs,
        "metrics": {
            "mean_phi_baseline": summarize_metrics(mean_pred, y_test_raw),
            "scaled_g_baseline": summarize_metrics(copy_scaled, y_test_raw),
            "linear_kernel_ridge": summarize_metrics(ridge_pred, y_test_raw),
            "latent_mlp": summarize_metrics(mlp_pred, y_test_raw),
        },
        "test_samples": [str(names[i]) for i in test_idx],
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "predictions.npz",
        dofs=dofs,
        coords=coords,
        test_idx=test_idx,
        y_true=y_test_raw,
        mean_pred=mean_pred,
        ridge_pred=ridge_pred,
        mlp_pred=mlp_pred,
    )

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

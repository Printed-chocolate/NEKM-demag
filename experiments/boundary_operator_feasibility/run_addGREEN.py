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


# ==========================================
# Latent MLP Model & Training
# ==========================================
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


# ==========================================
# 新增的 GreenFun Model & Training
# ==========================================
class GreenFun(torch.nn.Module):
    def __init__(self, N: int, N_quadrature: int, coord_dim: int):     
        super(GreenFun, self).__init__()
        self.N = N
        self.N_quad = N_quadrature
        # 动态适配输入坐标的维度（通常提取出来的coords有3维：x, y, z）
        self.G_layer = torch.nn.Sequential(
            torch.nn.Linear(coord_dim, N_quadrature), torch.nn.ReLU(),
            torch.nn.Linear(N_quadrature, N_quadrature), torch.nn.ReLU(),
            torch.nn.Linear(N_quadrature, N_quadrature), torch.nn.ReLU(),
            torch.nn.Linear(N_quadrature, N_quadrature)
        )

    def forward(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # f: (batch_size, N_quad)
        # x: (N, coord_dim)
        G = self.G_layer(x)     # G 形状: (N, N_quad)
        
        # 修正：f 不需要转置
        # f 的维度是 (batch_size, N_quad)，G.t() 的维度是 (N_quad, N)
        # 矩阵乘法后刚好得到 (batch_size, N)，符合每个样本输出 N 个点的值的需求
        output = torch.matmul(f, G.t()) 
        output = output / self.N_quad 
        return output

def train_green_fun(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    coords: np.ndarray,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    torch.manual_seed(seed)
    
    # 确定维度
    # 根据题意 N 和 N_quadrature 都等于边界自由度的数量 n_boundary
    n_samples, n_quad = x_train.shape
    n_eval = y_train.shape[1]
    coord_dim = coords.shape[1]
    
    model = GreenFun(N=n_eval, N_quadrature=n_quad, coord_dim=coord_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    
    # 准备 Tensor
    f_train = torch.as_tensor(x_train, dtype=torch.float32)
    target_train = torch.as_tensor(y_train, dtype=torch.float32)
    f_test = torch.as_tensor(x_test, dtype=torch.float32)
    coords_t = torch.as_tensor(coords, dtype=torch.float32)

    best_state = None
    best_loss = math.inf
    patience = 80
    stale = 0
    history = []
    
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        # 将 f 和 坐标 x 传入进行前向传播
        loss = loss_fn(model(f_train, coords_t), target_train)
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
        pred = model(f_test, coords_t).cpu().numpy()
        
    return pred, history


# ==========================================
# Plotting & Exports
# ==========================================
def plot_training_loss(mlp_history: list[float], green_history: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(np.arange(1, len(mlp_history) + 1), mlp_history, color="#1f77b4", linewidth=1.8, label="Latent MLP")
    ax.plot(np.arange(1, len(green_history) + 1), green_history, color="#d62728", linewidth=1.8, label="GreenFun")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title("Model Training Loss Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_test_relative_errors(
    y_true: np.ndarray,
    ridge_pred: np.ndarray,
    mlp_pred: np.ndarray,
    green_pred: np.ndarray,
    out_path: Path,
) -> None:
    ridge_rel = rel_l2(ridge_pred, y_true)
    mlp_rel = rel_l2(mlp_pred, y_true)
    green_rel = rel_l2(green_pred, y_true)
    x = np.arange(y_true.shape[0])
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 4.5), constrained_layout=True)
    ax.bar(x - width, ridge_rel, width, label="linear ridge", color="#2ca02c")
    ax.bar(x, mlp_rel, width, label="latent MLP", color="#ff7f0e")
    ax.bar(x + width, green_rel, width, label="GreenFun", color="#d62728")
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
    green_pred: np.ndarray,
    out_path: Path,
) -> int:
    ridge_rel = rel_l2(ridge_pred, y_true)
    mlp_rel = rel_l2(mlp_pred, y_true)
    green_rel = rel_l2(green_pred, y_true)
    
    combined = np.maximum(np.maximum(ridge_rel, mlp_rel), green_rel)
    sample = int(np.argmax(combined))

    ridge_abs = np.abs(ridge_pred[sample] - y_true[sample])
    mlp_abs = np.abs(mlp_pred[sample] - y_true[sample])
    green_abs = np.abs(green_pred[sample] - y_true[sample])
    
    values = [
        ("true phi2", y_true[sample], "viridis"),
        ("ridge |error|", ridge_abs, "magma"),
        ("MLP |error|", mlp_abs, "magma"),
        ("GreenFun |error|", green_abs, "magma"),
    ]

    fig = plt.figure(figsize=(12.0, 10.0), constrained_layout=True)
    for i, (title, value, cmap) in enumerate(values, start=1):
        # 将原先的一行三列改为 2x2 排列，容纳 4 个子图
        ax = fig.add_subplot(2, 2, i, projection="3d")
        sc = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=value, s=6, cmap=cmap, linewidths=0,
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
    green_pred: np.ndarray,
    mlp_history: list[float],
    green_history: list[float],
) -> dict[str, str | int]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    loss_path = plot_dir / "training_loss.png"
    rel_path = plot_dir / "test_relative_l2_errors.png"
    spatial_path = plot_dir / "worst_case_boundary_errors.png"
    
    plot_training_loss(mlp_history, green_history, loss_path)
    plot_test_relative_errors(y_true, ridge_pred, mlp_pred, green_pred, rel_path)
    worst_sample = plot_worst_case_spatial_errors(coords, y_true, ridge_pred, mlp_pred, green_pred, spatial_path)
    
    return {
        "training_loss": str(loss_path),
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
    
    print("training GreenFun", flush=True)
    green_pred_std, green_history = train_green_fun(
        x_train, y_train, x_test, coords, args.epochs, args.seed
    )
    # 反标准化得到预测真值
    green_pred = green_pred_std * y_std + y_mean

    plot_outputs = write_plots(
        out_dir, coords, y_test_raw, 
        ridge_pred, mlp_pred, green_pred, 
        mlp_history, green_history
    )

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
            "green_fun": summarize_metrics(green_pred, y_test_raw),  # 追加了新模型的指标
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
        green_pred=green_pred,  # 保存了新模型的预测结果
    )

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

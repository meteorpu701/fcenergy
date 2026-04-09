# src/fl/core/federated_client.py
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

def _count_samples(loader: DataLoader) -> int:
    try:
        return len(loader.dataset)
    except Exception:
        n = 0
        for xb, _ in loader:
            n += int(xb.shape[0])
        return n


@torch.no_grad()
def _load_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict, strict=True)


def _clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


@torch.no_grad()
def _compute_delta(
    local_weights: Dict[str, torch.Tensor],
    global_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    delta: Dict[str, torch.Tensor] = {}
    for k in global_weights.keys():
        g = global_weights[k]
        l = local_weights[k]
        if torch.is_tensor(g) and torch.is_floating_point(g):
            delta[k] = (l - g).detach().clone()
        else:
            delta[k] = torch.zeros_like(g) if torch.is_tensor(g) else 0
    return delta

def _zeros_like_named(named: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in named.items()}


@torch.no_grad()
def _state_add(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] + b[k].to(a[k].device, a[k].dtype) for k in a.keys()}


@torch.no_grad()
def _state_sub(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] - b[k].to(a[k].device, a[k].dtype) for k in a.keys()}


@torch.no_grad()
def _state_scale(a: Dict[str, torch.Tensor], s: float) -> Dict[str, torch.Tensor]:
    s = float(s)
    return {k: a[k] * s for k in a.keys()}

def train_local(
    model: nn.Module,
    loader: DataLoader,
    lr: float,
    epochs: int,
    device: str = "cpu",
) -> Tuple[float, int]:

    model.train()
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    last_loss = 0.0
    n_steps = 0

    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            n_steps += 1
            last_loss = float(loss.detach().cpu().item())

    return last_loss, n_steps

@torch.no_grad()
def _add_update_noise(delta: Dict[str, torch.Tensor], sigma: float) -> Dict[str, torch.Tensor]:
    if sigma is None or float(sigma) <= 0:
        return delta
    out = {}
    for k, v in delta.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            out[k] = v + float(sigma) * torch.randn_like(v)
        else:
            out[k] = v
    return out

@torch.no_grad()
def _clip_update_l2(delta: Dict[str, torch.Tensor], clip: float) -> Dict[str, torch.Tensor]:
    if clip is None or float(clip) <= 0:
        return delta

    clip = float(clip)

    sq_sum = 0.0
    for v in delta.values():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            sq_sum += float(v.pow(2).sum().item())

    norm = math.sqrt(sq_sum) if sq_sum > 0 else 0.0
    scale = min(1.0, clip / (norm + 1e-12))

    if scale >= 1.0:
        return delta

    out = {}
    for k, v in delta.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            out[k] = v * scale
        else:
            out[k] = v
    return out

def client_fit_fedavg(
    model: nn.Module,
    loader: DataLoader,
    global_weights: Dict[str, torch.Tensor],
    lr: float,
    epochs: int,
    device: str = "cpu",
    sigma: float = 0.0,
    clip: float = 0.0,
) -> Dict[str, Any]:
    _load_state(model, global_weights)

    loss, n_steps = train_local(
        model=model,
        loader=loader,
        lr=float(lr),
        epochs=int(epochs),
        device=device,
    )

    local_weights = model.state_dict()
    delta = _compute_delta(local_weights, global_weights)
    delta = _clip_update_l2(delta, clip) 
    delta = _add_update_noise(delta, sigma)

    return {
        "weights": local_weights,
        "delta": delta,           
        "n_steps": int(n_steps),
        "n_samples": _count_samples(loader),
        "train_loss": float(loss),
    }

def train_local_fedprox(
    model: nn.Module,
    global_model: nn.Module,
    loader: DataLoader,
    lr: float,
    epochs: int,
    mu: float = 0.01,
    device: str = "cpu",
) -> Tuple[float, int]:
    model.train()
    model.to(device)
    global_model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    last_loss = 0.0
    n_steps = 0

    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)

            prox = 0.0
            for w, w0 in zip(model.parameters(), global_model.parameters()):
                prox = prox + ((w - w0) ** 2).sum()

            loss = loss + (float(mu) / 2.0) * prox
            loss.backward()
            opt.step()

            n_steps += 1
            last_loss = float(loss.detach().cpu().item())

    return last_loss, n_steps


def client_fit_fedprox(
    model: nn.Module,
    loader: DataLoader,
    global_weights: Dict[str, torch.Tensor],
    lr: float,
    epochs: int,
    mu: float = 0.01,
    device: str = "cpu",
) -> Dict[str, Any]:
    global_model = _clone_model(model)
    _load_state(global_model, global_weights)

    _load_state(model, global_weights)

    loss, n_steps = train_local_fedprox(
        model=model,
        global_model=global_model,
        loader=loader,
        lr=float(lr),
        epochs=int(epochs),
        mu=float(mu),
        device=device,
    )

    local_weights = model.state_dict()
    delta = _compute_delta(local_weights, global_weights)

    return {
        "weights": local_weights,
        "delta": delta,           
        "n_steps": int(n_steps),
        "n_samples": _count_samples(loader),
        "train_loss": float(loss),
    }

def train_local_scaffold(
    model: nn.Module,
    loader: DataLoader,
    lr: float,
    epochs: int,
    c_global: Dict[str, torch.Tensor],
    c_local: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Tuple[float, int]:
    
    model.train()
    model.to(device)

    c_g = {k: v.to(device) for k, v in c_global.items()}
    c_l = {k: v.to(device) for k, v in c_local.items()}

    opt = torch.optim.SGD(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    last_loss = 0.0
    n_steps = 0

    named_params = dict(model.named_parameters())

    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            for name, p in named_params.items():
                if p.grad is None:
                    continue
                p.grad = p.grad - c_l[name] + c_g[name]

            opt.step()
            n_steps += 1
            last_loss = float(loss.detach().cpu().item())

    return last_loss, n_steps


def client_fit_scaffold(
    model: nn.Module,
    loader: DataLoader,
    global_weights: Dict[str, torch.Tensor],
    c_global: Dict[str, torch.Tensor],
    c_local: Dict[str, torch.Tensor],
    lr: float,
    epochs: int,
    device: str = "cpu",
) -> Dict[str, Any]:

    _load_state(model, global_weights)

    loss, n_steps = train_local_scaffold(
        model=model,
        loader=loader,
        lr=float(lr),
        epochs=int(epochs),
        c_global=c_global,
        c_local=c_local,
        device=device,
    )

    local_weights = model.state_dict()

    steps = int(n_steps) if int(n_steps) > 0 else 1
    w_global_named = {}
    w_local_named = {}
    for name, p in model.named_parameters():
        w_global_named[name] = global_weights[name].detach()
        w_local_named[name] = local_weights[name].detach()

    w_diff = _state_sub(w_global_named, w_local_named) 
    w_term = _state_scale(w_diff, 1.0 / (float(lr) * float(steps)))

    c_i_new = _state_add(_state_sub(c_local, c_global), w_term)
    c_delta = _state_sub(c_i_new, c_local)

    return {
        "weights": local_weights,
        "n_samples": _count_samples(loader),
        "n_steps": steps,
        "train_loss": float(loss),
        "c_delta": c_delta,
        "c_i_new": c_i_new,
    }
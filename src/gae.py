"""One-layer graph auto-encoder on the entity graph (PyTorch, full batch, CPU)."""
import numpy as np
import torch


class EntityGAE(torch.nn.Module):
    """Z = P W (identity node features, no bias, no non-linearity). Decoder: Z Z^T."""

    def __init__(self, n_nodes, dim, seed=0, init_std=0.1):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        w = torch.randn(n_nodes, dim, generator=g, dtype=torch.float64) * init_std
        self.W = torch.nn.Parameter(w)

    def encode(self, P):
        return P @ self.W

    def forward(self, P):
        Z = self.encode(P)
        return Z @ Z.T


def train_gae(P, target, dim, seed=0, lr=0.01, clip=1.0, max_epochs=3000, patience=200,
              tol=1e-3, alpha=0.0, init_std=0.1):
    """Train and return (Z, info). The target is scaled by n so its mean entry is about 1.

    Early stop: relative loss improvement below `tol` over `patience` epochs. The weights with
    the lowest loss are kept (the paper reports loss spikes).
    """
    n = P.shape[0]
    Pt = torch.tensor(P, dtype=torch.float64)
    Tt = torch.tensor(target, dtype=torch.float64) * float(n)
    model = EntityGAE(n, dim, seed=seed, init_std=init_std)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best, best_w = [], float("inf"), None
    for epoch in range(int(max_epochs)):
        opt.zero_grad()
        diff = model(Pt) - Tt
        loss = (1.0 - alpha) * (diff ** 2).mean()
        if alpha > 0:
            loss = loss + alpha * diff.abs().mean()
        value = float(loss.item())
        history.append(value)
        if value < best:
            best, best_w = value, model.W.detach().clone()
        if epoch >= patience:
            ref = history[epoch - patience]
            if (ref - min(history[epoch - patience + 1:])) / max(abs(ref), 1e-30) < tol:
                break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    with torch.no_grad():
        Z = (Pt @ best_w).numpy().copy()
    baseline = float(((Tt - Tt.mean()) ** 2).mean().item())
    info = {"epochs": len(history), "final_loss": best, "history": history,
            "target_variance": baseline,
            "explained_share": float(1.0 - best / baseline) if baseline > 0 else 0.0,
            "n_parameters": int(n * dim)}
    return Z, info

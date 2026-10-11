"""LSTM autoencoder on 48-step multichannel windows (deep detector).

Reconstruction-based: trained to rebuild normal 24 h windows of
``P, Q, V, I`` and the four sub-meters; the per-window reconstruction error is the
anomaly score. Reusable for Phase 2 federated continual learning (the same model
family as the reference FL paper).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from energymind import io
from energymind.detect.base import to_score

HOUSEHOLD_COL = io.HOUSEHOLD_COL

#: Multichannel input (aggregate + sub-meters).
CHANNELS = [
    "active_power_kw",
    "reactive_power_kvar",
    "voltage_v",
    "current_a",
    "sub1_heating_kwh",
    "sub2_cooling_kwh",
    "sub3_lighting_kwh",
    "sub4_appliance_kwh",
]


class _LSTMAE(nn.Module):
    def __init__(self, n_channels: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_channels, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_channels)

    def forward(self, x):  # x: (batch, window, channels)
        _, (h, _) = self.encoder(x)
        h = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h)
        return self.head(out)


def _windows(values: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Sliding windows ``(n, window, channels)`` from a ``(T, channels)`` array."""
    n = (len(values) - window) // stride + 1
    if n <= 0:
        return np.empty((0, window, values.shape[1]), dtype=np.float32)
    idx = np.arange(window)[None, :] + stride * np.arange(n)[:, None]
    return values[idx].astype(np.float32)


class LSTMAutoencoder:
    """Windowed LSTM autoencoder; higher reconstruction error = more anomalous."""

    name = "lstm_ae"

    def __init__(
        self,
        window: int = 48,
        hidden: int = 32,
        epochs: int = 10,
        batch: int = 256,
        lr: float = 1e-3,
        stride: int = 2,
        max_windows: int = 40_000,
        channels: list[str] | None = None,
        random_state: int = 0,
    ) -> None:
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.stride = stride
        self.max_windows = max_windows
        self.channels = channels or CHANNELS
        self.random_state = random_state
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.model: _LSTMAE | None = None

    # -- scaling -------------------------------------------------------------
    def _scale(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.std_

    def _household_array(self, frame: pd.DataFrame, household: str) -> np.ndarray:
        block = frame[frame[HOUSEHOLD_COL] == household]
        return block[self.channels].astype(float).to_numpy()

    # -- fit -----------------------------------------------------------------
    def fit(self, frame: pd.DataFrame) -> LSTMAutoencoder:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        stack = frame[self.channels].astype(float).to_numpy()
        self.mean_ = np.nanmean(stack, axis=0)
        self.std_ = np.nanstd(stack, axis=0)
        self.std_[self.std_ == 0] = 1.0

        batches = []
        for household in frame[HOUSEHOLD_COL].unique():
            values = self._household_array(frame, household)
            values = np.nan_to_num(self._scale(values), nan=0.0)
            batches.append(_windows(values, self.window, self.stride))
        windows = np.concatenate(batches, axis=0)
        if len(windows) > self.max_windows:
            rng = np.random.default_rng(self.random_state)
            pick = rng.choice(len(windows), size=self.max_windows, replace=False)
            windows = windows[pick]

        model = _LSTMAE(len(self.channels), self.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        tensor = torch.from_numpy(windows)
        n = len(tensor)
        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch):
                batch = tensor[perm[start : start + self.batch]]
                opt.zero_grad()
                recon = model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
        model.eval()
        self.model = model
        return self

    # -- score ---------------------------------------------------------------
    @torch.no_grad()
    def score(self, frame: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("LSTMAutoencoder.fit must be called first")
        scores = pd.Series(np.nan, index=frame.index)
        for household in frame[HOUSEHOLD_COL].unique():
            rows = frame.index[frame[HOUSEHOLD_COL] == household]
            values = np.nan_to_num(self._scale(self._household_array(frame, household)), nan=0.0)
            windows = _windows(values, self.window, self.stride)
            if len(windows) == 0:
                continue
            errors = np.empty(len(windows), dtype=float)
            tensor = torch.from_numpy(windows)
            for start in range(0, len(tensor), self.batch):
                batch = tensor[start : start + self.batch]
                recon = self.model(batch)
                err = ((recon - batch) ** 2).mean(dim=(1, 2)).numpy()
                errors[start : start + len(batch)] = err
            # assign each window's error to its last (causal) interval
            end_positions = self.window - 1 + self.stride * np.arange(len(windows))
            scores.loc[rows[end_positions]] = errors
        return to_score(scores.to_numpy(), frame.index)

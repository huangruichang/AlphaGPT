import torch
import torch.nn as nn


class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class TechnicalIndicators:
    """Generic OHLCV-based technical indicators. No crypto-specific fields."""

    @staticmethod
    def robust_norm(t: torch.Tensor) -> torch.Tensor:
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)

    @staticmethod
    def candlestick_pressure(open_: torch.Tensor, high: torch.Tensor,
                             low: torch.Tensor, close: torch.Tensor) -> torch.Tensor:
        """Generic candle pressure: where close sits within the hl range."""
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def volume_acceleration(volume: torch.Tensor, window: int = 5) -> torch.Tensor:
        """Second derivative (acceleration) of volume."""
        vol_prev = torch.roll(volume, 1, dims=1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - torch.roll(vol_chg, 1, dims=1)
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def price_deviation(close: torch.Tensor, window: int = 20) -> torch.Tensor:
        """(close - MA(close, window)) / MA(close, window)."""
        pad = torch.zeros((close.shape[0], window - 1), device=close.device)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close: torch.Tensor, window: int = 10) -> torch.Tensor:
        """Rolling std of squared returns (volatility clustering proxy)."""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret_sq = ret ** 2
        pad = torch.zeros((ret_sq.shape[0], window - 1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close: torch.Tensor, window: int = 5) -> torch.Tensor:
        """Sign flip in rolling momentum (reversal signal)."""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        pad = torch.zeros((ret.shape[0], window - 1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        mom_prev = torch.roll(mom, 1, dims=1)
        reversal = (mom * mom_prev < 0).float()
        return reversal

    @staticmethod
    def relative_strength(close: torch.Tensor, high: torch.Tensor,
                          low: torch.Tensor, window: int = 14) -> torch.Tensor:
        """RSI-like indicator, normalised to [-1, 1]."""
        delta = close - torch.roll(close, 1, dims=1)
        gains = torch.relu(delta)
        losses = torch.relu(-delta)
        pad = torch.zeros((gains.shape[0], window - 1), device=close.device)
        avg_gain = (torch.cat([pad, gains], dim=1)
                    .unfold(1, window, 1).mean(dim=-1))
        avg_loss = (torch.cat([pad, losses], dim=1)
                    .unfold(1, window, 1).mean(dim=-1))
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return (rsi - 50) / 50   # ∈ [-1, 1]


class FeatureEngineer:
    """
    Basic feature engineer for A-share factor mining.
    Input:  numpy array `raw` of shape (n_stocks, hist_len, 5)
            last axis = [open, high, low, close, volume]
    Output: torch tensor of shape (n_stocks, 5, hist_len)
    """
    INPUT_DIM = 5

    @staticmethod
    def compute_features(raw: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        raw : torch.Tensor
            Shape (n, T, 5) with columns:
                0 = open, 1 = high, 2 = low, 3 = close, 4 = volume

        Returns
        -------
        torch.Tensor
            Shape (n, 5, T) — one channel per factor, ready for Conv1d / LSTM.
        """
        if isinstance(raw, torch.Tensor):
            raw_t = raw
        else:
            raw_t = torch.from_numpy(raw).float()

        open_ = raw_t[:, :, 0]
        high  = raw_t[:, :, 1]
        low   = raw_t[:, :, 2]
        close = raw_t[:, :, 3]
        vol   = raw_t[:, :, 4]

        def robust_norm(t: torch.Tensor) -> torch.Tensor:
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        # 1. log return
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))

        # 2. candlestick pressure
        pressure = TechnicalIndicators.candlestick_pressure(open_, high, low, close)

        # 3. volume acceleration (FOMO / second derivative of volume)
        fomo = TechnicalIndicators.volume_acceleration(vol)

        # 4. price deviation from 20-day MA
        dev = TechnicalIndicators.price_deviation(close)

        # 5. log volume
        log_vol = torch.log1p(vol)

        features = torch.stack([
            robust_norm(ret),
            pressure,
            robust_norm(fomo),
            robust_norm(dev),
            robust_norm(log_vol),
        ], dim=1)   # (n, 5, T)

        return features


class AdvancedFactorEngineer:
    """
    Extended feature set (8 factors) built from the same OHLCV input.
    INPUT_DIM = 8 when all advanced factors are used,
    or 5 if only the basic set is selected.
    """
    INPUT_DIM = 8

    def __init__(self):
        self.rms_norm = RMSNormFactor(1)

    def robust_norm(self, t: torch.Tensor) -> torch.Tensor:
        """Robust normalisation using median absolute deviation."""
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)

    def compute_features(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        raw : torch.Tensor | np.ndarray
            Shape (n, T, 5): [open, high, low, close, volume]

        Returns
        -------
        torch.Tensor
            Shape (n, 8, T)
        """
        if isinstance(raw, torch.Tensor):
            raw_t = raw
        else:
            raw_t = torch.from_numpy(raw).float()

        open_ = raw_t[:, :, 0]
        high  = raw_t[:, :, 1]
        low   = raw_t[:, :, 2]
        close = raw_t[:, :, 3]
        vol   = raw_t[:, :, 4]

        # --- basic 5 factors ---
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        pressure = TechnicalIndicators.candlestick_pressure(open_, high, low, close)
        fomo = TechnicalIndicators.volume_acceleration(vol)
        dev = TechnicalIndicators.price_deviation(close)
        log_vol = torch.log1p(vol)

        # --- 3 additional advanced factors ---
        vol_cluster = TechnicalIndicators.volatility_clustering(close)
        momentum_rev = TechnicalIndicators.momentum_reversal(close)
        rel_strength = TechnicalIndicators.relative_strength(close, high, low)

        features = torch.stack([
            self.robust_norm(ret),          # 0
            pressure,                       # 1  (already ±1 from tanh)
            self.robust_norm(fomo),         # 2
            self.robust_norm(dev),          # 3
            self.robust_norm(log_vol),      # 4
            self.robust_norm(vol_cluster),  # 5
            momentum_rev,                   # 6  (0 or 1)
            self.robust_norm(rel_strength), # 7
        ], dim=1)   # (n, 8, T)

        return features

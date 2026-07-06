import torch


class FactorBacktest:
    """
    A-share factor backtest engine.

    Assumptions
    -----------
    * raw_data dict keys: open, high, low, close, volume  (all torch.Tensor
      [n_stocks, hist_len], dtype float32, on same device)
    * `close` is adjusted close price
    * `volume` is in shares (already converted from lots if needed)
    * target_ret: tensor [n_stocks, hist_len-1] – close-to-close return
      (same length as position[:, 1:])
    * factor_signals: tensor [n_stocks, hist_len-1] – model output (raw logits)

    A-share specific rules
    -----------------------
    1. Commission: 0.03% one-way  (~0.025% broker + exchange fees)
    2. Stamp duty: 0.1% on sell only → approximated as 0.05% round-trip
       averaged into base_fee = 0.001 (0.1% round-trip)
    3. Limit-up mask: if pct_chg >= 0.095 (9.5%) assume limit-up → cannot buy
       if pct_chg <= -0.095 assume limit-down → cannot sell
    4. Suspended stock mask: volume == 0 → skip (no trade, position forced 0)
    """

    def __init__(self):
        # 0.1% round-trip: ~0.03% commission + 0.1% stamp duty on sell
        self.base_fee = 0.0010
        # signal threshold – converts raw factor into binary position
        self.signal_threshold = 0.85
        # activity floor for scoring (A-share universe is larger)
        self.min_activity = 20

    def evaluate(self, factor_signals, raw_data, target_ret):
        """
        Parameters
        ----------
        factor_signals : torch.Tensor [n_stocks, hist_len-1]
        raw_data : dict with keys {open, high, low, close, volume}
                   each tensor [n_stocks, hist_len]
        target_ret : torch.Tensor [n_stocks, hist_len-1]

        Returns
        -------
        result : dict with keys
            score, sharpe, max_dd, win_rate, activity_bonus, final_fitness
        """
        device = factor_signals.device
        close = raw_data["close"]          # [N, T]
        volume = raw_data["volume"]        # [N, T]
        T_minus_1 = factor_signals.shape[1]

        # ------------------------------------------------------------------ #
        # 1. Build tradability mask (volume == 0 → suspended)
        # ------------------------------------------------------------------ #
        # volume is [N, T]; we need [N, T-1] to align with target_ret
        not_suspended = (volume[:, 1:] > 0).float()   # [N, T-1]

        # ------------------------------------------------------------------ #
        # 2. Limit-up / limit-down mask
        # ------------------------------------------------------------------ #
        # pct_chg[t] = (close[t] - close[t-1]) / close[t-1]
        # close is [N, T]; compute pct_chg for t = 1..T-1 → [N, T-1]
        prev_close = close[:, :-1]
        curr_close = close[:, 1:]
        pct_chg = (curr_close - prev_close) / (prev_close + 1e-9)   # [N, T-1]

        # If limit-up (pct_chg >= 0.095): cannot buy at open of that bar
        #   → force position to 0 at that timestep
        # If limit-down (pct_chg <= -0.095): cannot sell
        #   → approximate by also forcing position to 0 (cannot exit cleanly)
        limit_up_mask = (pct_chg < 0.095).float()        # 1 = can trade, 0 = limit-up
        # can only hold position where not suspended AND not limit-up
        can_trade = not_suspended * limit_up_mask         # [N, T-1]

        # ------------------------------------------------------------------ #
        # 3. Position from factor signals
        # ------------------------------------------------------------------ #
        signal = torch.sigmoid(factor_signals)                # [N, T-1]
        raw_position = (signal > self.signal_threshold).float()  # [N, T-1]

        # apply tradability mask
        position = raw_position * can_trade                    # [N, T-1]

        # ------------------------------------------------------------------ #
        # 4. Transaction cost (constant commission + stamp duty model)
        # ------------------------------------------------------------------ #
        prev_pos = torch.roll(position, 1, dims=1)            # [N, T-1]
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)             # [N, T-1]

        # Constant round-trip cost applied on turnover
        tx_cost = turnover * self.base_fee                     # [N, T-1]

        # ------------------------------------------------------------------ #
        # 5. PnL
        # ------------------------------------------------------------------ #
        gross_pnl = position * target_ret                      # [N, T-1]
        net_pnl = gross_pnl - tx_cost                          # [N, T-1]

        # ------------------------------------------------------------------ #
        # 6. Per-stock metrics
        # ------------------------------------------------------------------ #
        cum_ret = net_pnl.sum(dim=1)                           # [N]

        # Sharpe ratio per stock
        mean_ret = net_pnl.mean(dim=1)                         # [N]
        std_ret = net_pnl.std(dim=1) + 1e-9                   # [N]
        sqrt_T = torch.sqrt(torch.tensor(T_minus_1, dtype=torch.float32, device=device))
        sharpe_per_stock = mean_ret / std_ret * sqrt_T          # [N]
        sharpe = sharpe_per_stock.mean().item()

        # Max drawdown per stock (guard against running_max == 0)
        cum = torch.cumsum(net_pnl, dim=1)                    # [N, T-1]
        running_max = torch.cummax(cum, dim=1).values          # [N, T-1]
        # Avoid div-by-zero: if running_max == 0, drawdown = 0 (no pnl yet)
        dd_denom = running_max.abs() + 1e-9
        drawdown = (cum - running_max) / dd_denom              # [N, T-1]
        max_dd_per_stock = (-drawdown.min(dim=1).values).clamp(0.0, 1.0)  # [N]
        max_dd = max_dd_per_stock.mean().item()

        # Win rate per stock
        wins = (net_pnl > 0).float().sum(dim=1)               # [N]
        win_rate_per_stock = wins / float(T_minus_1)            # [N]
        win_rate = win_rate_per_stock.mean().item()

        # Activity per stock
        activity = position.sum(dim=1)                         # [N]

        # ------------------------------------------------------------------ #
        # 7. Composite score per stock (RL reward)
        # ------------------------------------------------------------------ #
        # activity_bonus: 1.0 if activity >= min_activity, else interpolates down
        activity_bonus_per_stock = torch.where(
            activity >= self.min_activity,
            torch.tensor(1.0, device=device),
            activity / float(self.min_activity)
        )                                                      # [N]

        score_per_stock = (
            sharpe_per_stock * 0.4
            + (1.0 - max_dd_per_stock) * 0.3
            + win_rate_per_stock * 0.2
            + activity_bonus_per_stock * 0.1
        )                                                      # [N]

        # Penalise low-activity stocks
        score_per_stock = torch.where(
            activity < self.min_activity,
            torch.tensor(-10.0, device=device),
            score_per_stock
        )

        final_fitness = torch.median(score_per_stock).item()

        activity_bonus = activity_bonus_per_stock.mean().item()

        return {
            "score": score_per_stock.mean().item(),
            "sharpe": sharpe,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "activity_bonus": activity_bonus,
            "final_fitness": final_fitness,
            "cum_ret_mean": cum_ret.mean().item(),
        }

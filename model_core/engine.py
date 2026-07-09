import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json

from .config import ModelConfig
from .data_loader import CryptoDataLoader as StockDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import FactorBacktest

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.

        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.loader = StockDataLoader()
        self.loader.load_data()

        self.model = AlphaGPT().to(ModelConfig.DEVICE)

        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = FactorBacktest()

        self.best_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': []
        }

    def _convert_raw_data_format(self):
        """
        转换 raw_data_cache 格式以适配 backtest.py 的期望格式。

        raw_data_cache 当前格式:
            {ts_code: {"close": np.array([T]), "volume": np.array([T]), ...}}

        backtest.py 期望格式:
            {"close": torch.Tensor([N, T]), "volume": torch.Tensor([N, T])}

        Returns:
            dict: {"close": Tensor([N, T]), "volume": Tensor([N, T])}
        """
        import numpy as np
        from torch import from_numpy

        # 获取所有股票代码（已排序）
        stocks = sorted(self.loader.raw_data_cache.keys())
        N = len(stocks)

        # 获取最短序列长度（所有股票对齐到这个长度）
        lengths = [len(self.loader.raw_data_cache[c]["close"]) for c in stocks]
        T = min(lengths)

        # 初始化二维数组
        close_mat = np.zeros((N, T), dtype=np.float32)
        volume_mat = np.zeros((N, T), dtype=np.float32)

        # 填充数据
        for i, code in enumerate(stocks):
            rec = self.loader.raw_data_cache[code]
            close_mat[i, :] = rec["close"][:T]
            volume_mat[i, :] = rec["volume"][:T]

        # 转换为Tensor并移到正确设备
        device = ModelConfig.DEVICE
        raw_data_converted = {
            "close": from_numpy(close_mat).to(device),
            "volume": from_numpy(volume_mat).to(device),
        }

        return raw_data_converted

    def train(self):
        print("Starting A-share Alpha Mining with LoRD Regularization..." if self.use_lord else "Starting A-share Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

            log_probs = []
            tokens_list = []

            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)

            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

            # =============================================
            # 转换 raw_data_cache 格式以适配 backtest.py
            # backtest.py 期望: raw_data["close"] 是 [N, T] 二维张量
            # 当前格式: raw_data_cache[code]["close"] 是 [T] 一维数组
            # =============================================
            raw_data_converted = self._convert_raw_data_format()
            
            for i in range(bs):
                formula = seqs[i].tolist()

                res = self.vm.execute(formula, self.loader.feat_tensor)

                if res is None:
                    rewards[i] = -5.0
                    continue

                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue

                # Note: res shape is [N, T], but target_returns is [N, T-1]
                #  Truncate res to match target_returns
                if res.shape[1] != self.loader.target_returns.shape[1]:
                    res = res[:, :-1]  # Drop last timestep

                result = self.bt.evaluate(res, raw_data_converted, self.loader.target_returns)
                score = result["score"]
                ret_val = result.get("cum_ret_mean", 0.0)
                rewards[i] = score

                if score > self.best_score:
                    self.best_score = score
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}")

            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv

            loss = loss.mean()

            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()

            # Logging
            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}

            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)

            pbar.set_postfix(postfix_dict)

        # Save best formula
        with open("best_ashare_strategy.json", "w") as f:
            json.dump(self.best_formula, f)

        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)

        print(f"\nTraining completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()

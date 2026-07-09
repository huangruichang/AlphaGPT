"""
基于项目架构的A股因子挖掘模块
使用 model_core/data_loader.py 和 model_core/factors.py
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
from .data_loader import CryptoDataLoader
from .factors import FeatureEngineer, AdvancedFactorEngineer, TechnicalIndicators
from .config import ModelConfig
import warnings
warnings.filterwarnings('ignore')


class FactorMiner:
    """因子挖掘与评估类"""

    def __init__(self, use_advanced_factors=True):
        """
        初始化因子挖掘器

        Parameters
        ----------
        use_advanced_factors : bool
            是否使用扩展因子集（8个因子 vs 5个因子）
        """
        self.use_advanced = use_advanced_factors
        self.data_loader = CryptoDataLoader()
        self.factor_names = []
        self.factor_ic = {}
        self.factor_returns = {}

    def load_data(self):
        """加载数据"""
        print("="*60)
        print("加载A股数据...")
        print("="*60)
        self.data_loader.load_data()

        # 获取因子名称
        if self.use_advanced:
            self.factor_names = [
                'log_return', 'candlestick_pressure', 'volume_acceleration',
                'price_deviation', 'log_volume', 'volatility_clustering',
                'momentum_reversal', 'relative_strength'
            ]
        else:
            self.factor_names = [
                'log_return', 'candlestick_pressure', 'volume_acceleration',
                'price_deviation', 'log_volume'
            ]

        print(f"因子数量: {len(self.factor_names)}")
        print(f"股票数量: {self.data_loader.get_stock_count()}")
        print(f"时间序列长度: {self.data_loader.feat_tensor.shape[1]}")
        print("="*60 + "\n")

    def calculate_factor_ic(self, forward_periods=[1, 3, 5, 10]):
        """
        计算每个因子的IC (Information Coefficient)

        IC = spearman correlation(factor_value, future_return)
        """
        print("计算因子IC...")
        print("-"*60)

        feat_tensor = self.data_loader.feat_tensor  # (n, T, input_dim)
        price_tensor = self.data_loader.price_tensor  # (n, T)

        n_stocks, T, input_dim = feat_tensor.shape

        # 计算未来收益率
        future_returns = {}
        for period in forward_periods:
            # future_return[t] = log(close[t+period] / close[t])
            future_ret = torch.log(
                price_tensor[:, period:] / price_tensor[:, :-period]
            )
            # 对齐时间维度
            # feat_tensor[:, :T-period, :] 对应 future_ret[:, :T-period]
            future_returns[period] = future_ret[:, :T-period]

        # 对每个因子计算IC
        for factor_idx, factor_name in enumerate(self.factor_names):
            print(f"\n处理因子: {factor_name}")

            self.factor_ic[factor_name] = {}

            for period in forward_periods:
                factor_values = feat_tensor[:, :T-period, factor_idx].cpu().numpy()
                future_ret = future_returns[period].cpu().numpy()

                # 计算每个时间点的IC
                ic_values = []
                for t in range(T - period):
                    valid_mask = ~(np.isnan(factor_values[:, t]) | np.isnan(future_ret[:, t]))
                    if valid_mask.sum() > 10:  # 至少10个有效样本
                        ic = stats.spearmanr(
                            factor_values[valid_mask, t],
                            future_ret[valid_mask, t]
                        )[0]
                        if not np.isnan(ic):
                            ic_values.append(ic)

                # 平均IC
                if ic_values:
                    avg_ic = np.mean(ic_values)
                    std_ic = np.std(ic_values)
                    ic_ir = avg_ic / (std_ic + 1e-8)  # IC_IR = IC / std(IC)

                    self.factor_ic[factor_name][period] = {
                        'ic': avg_ic,
                        'ic_std': std_ic,
                        'ic_ir': ic_ir,
                        'ic_values': ic_values
                    }

                    print(f"  未来{period}期: IC={avg_ic:.4f}, IC_IR={ic_ir:.4f}")

        print("\n" + "="*60)
        print("IC计算完成!")
        print("="*60 + "\n")

    def calculate_factor_returns(self, forward_periods=[1, 3, 5, 10], n_quantiles=5):
        """
        计算因子收益率（分层回测）

        将股票按因子值分为n_quantiles层，计算每层的未来收益率
        """
        print("计算因子收益率（分层回测）...")
        print("-"*60)

        feat_tensor = self.data_loader.feat_tensor
        price_tensor = self.data_loader.price_tensor

        n_stocks, T, input_dim = feat_tensor.shape

        # 计算未来收益率
        future_returns = {}
        for period in forward_periods:
            future_ret = torch.log(
                price_tensor[:, period:] / price_tensor[:, :-period]
            )
            future_returns[period] = future_ret[:, :T-period]

        # 对每个因子进行分层回测
        for factor_idx, factor_name in enumerate(self.factor_names):
            print(f"\n因子: {factor_name}")

            self.factor_returns[factor_name] = {}

            for period in forward_periods:
                factor_values = feat_tensor[:, :T-period, factor_idx].cpu().numpy()
                future_ret = future_returns[period].cpu().numpy()

                quantile_returns = []
                long_short_returns = []

                for t in range(T - period):
                    valid_mask = ~(np.isnan(factor_values[:, t]) | np.isnan(future_ret[:, t]))

                    if valid_mask.sum() < n_quantiles * 10:  # 每层至少10只股票
                        quantile_returns.append(np.full(n_quantiles, np.nan))
                        long_short_returns.append(np.nan)
                        continue

                    fv_valid = factor_values[valid_mask, t]
                    fr_valid = future_ret[valid_mask, t]

                    # 分层
                    quantiles = pd.qcut(fv_valid, n_quantiles, labels=False, duplicates='drop')

                    # 计算每层的平均收益
                    q_ret = []
                    for q in range(n_quantiles):
                        mask = quantiles == q
                        if mask.sum() > 0:
                            q_ret.append(fr_valid[mask].mean())
                        else:
                            q_ret.append(np.nan)

                    quantile_returns.append(q_ret)

                    # 多空收益 (Top层 - Bottom层)
                    if len(q_ret) == n_quantiles and not any(np.isnan(q_ret)):
                        long_short_returns.append(q_ret[-1] - q_ret[0])
                    else:
                        long_short_returns.append(np.nan)

                # 平均收益率
                quantile_returns = np.array(quantile_returns)
                avg_quantile_ret = np.nanmean(quantile_returns, axis=0)

                avg_long_short = np.nanmean(long_short_returns)

                self.factor_returns[factor_name][period] = {
                    'quantile_returns': avg_quantile_ret,
                    'long_short_return': avg_long_short,
                    'long_short_returns_series': long_short_returns
                }

                print(f"  未来{period}期: 多空收益={avg_long_short:.6f}, "
                      f"Top层收益={avg_quantile_ret[-1]:.6f}, "
                      f"Bottom层收益={avg_quantile_ret[0]:.6f}")

        print("\n" + "="*60)
        print("因子收益率计算完成!")
        print("="*60 + "\n")

    def rank_factors_by_ic(self, period=5):
        """
        按IC绝对值排序因子

        Returns
        -------
        pd.DataFrame
            排序后的因子结果
        """
        results = []

        for factor_name in self.factor_names:
            if period in self.factor_ic.get(factor_name, {}):
                ic_info = self.factor_ic[factor_name][period]
                ret_info = self.factor_returns.get(factor_name, {}).get(period, {})

                results.append({
                    'factor': factor_name,
                    'ic': ic_info['ic'],
                    'abs_ic': abs(ic_info['ic']),
                    'ic_ir': ic_info['ic_ir'],
                    'long_short_return': ret_info.get('long_short_return', np.nan),
                    'top_quantile_return': ret_info.get('quantile_returns', [np.nan])[-1] if 'quantile_returns' in ret_info else np.nan,
                    'bottom_quantile_return': ret_info.get('quantile_returns', [np.nan])[0] if 'quantile_returns' in ret_info else np.nan,
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('abs_ic', ascending=False)

        return results_df

    def print_top_factors(self, top_n=10):
        """打印Top有效因子"""
        print("\n" + "="*60)
        print("Top 有效因子排行")
        print("="*60)

        for period in [1, 3, 5, 10]:
            print(f"\n【未来 {period} 期收益率预测能力】")
            print("-"*60)

            ranking = self.rank_factors_by_ic(period)

            if len(ranking) == 0:
                print("无有效因子")
                continue

            top_factors = ranking.head(top_n)

            for idx, row in top_factors.iterrows():
                print(f"{row['factor']:30s} | IC: {row['ic']:7.4f} | "
                      f"IC_IR: {row['ic_ir']:6.3f} | "
                      f"多空收益: {row['long_short_return']:8.6f} | "
                      f"Top层收益: {row['top_quantile_return']:8.6f}")

    def visualize_factor_ic(self, factor_name, save_path=None):
        """
        可视化单个因子的IC时间序列

        Parameters
        ----------
        factor_name : str
            因子名称
        save_path : str
            保存路径
        """
        if factor_name not in self.factor_ic:
            print(f"因子 {factor_name} 不存在")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Factor: {factor_name}', fontsize=16)

        periods = list(self.factor_ic[factor_name].keys())

        for idx, period in enumerate(periods[:4]):
            ax = axes[idx // 2, idx % 2]

            ic_values = self.factor_ic[factor_name][period]['ic_values']
            ic_cumsum = np.cumsum(ic_values)

            ax.plot(ic_values, alpha=0.5, label='IC')
            ax.plot(ic_cumsum, label='Cumulative IC')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

            ax.set_title(f'Future {period} Period(s)')
            ax.set_xlabel('Time')
            ax.set_ylabel('IC')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到 {save_path}")

        plt.show()

    def save_results(self, filename='factor_mining_results.csv'):
        """保存因子挖掘结果"""
        all_results = []

        for period in [1, 3, 5, 10]:
            ranking = self.rank_factors_by_ic(period)
            if len(ranking) > 0:
                ranking['period'] = period
                all_results.append(ranking)

        if all_results:
            result_df = pd.concat(all_results, ignore_index=True)
            result_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n结果已保存到 {filename}")

    def run_full_mining(self):
        """运行完整的因子挖掘流程"""
        # 1. 加载数据
        self.load_data()

        # 2. 计算因子IC
        self.calculate_factor_ic(forward_periods=[1, 3, 5, 10])

        # 3. 计算因子收益率
        self.calculate_factor_returns(forward_periods=[1, 3, 5, 10])

        # 4. 打印Top因子
        self.print_top_factors(top_n=10)

        # 5. 保存结果
        self.save_results()

        print("\n" + "="*60)
        print("因子挖掘完成!")
        print("="*60)


def main():
    """主函数"""
    # 使用基础因子集（5个因子）
    miner = FactorMiner(use_advanced_factors=False)
    miner.run_full_mining()

    # 使用扩展因子集（8个因子）
    # miner_advanced = FactorMiner(use_advanced_factors=True)
    # miner_advanced.run_full_mining()


if __name__ == '__main__':
    main()

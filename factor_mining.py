"""
A股因子挖掘脚本
使用已有的日线数据进行因子挖掘和回测
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_pipeline.ashare_db import SessionLocal, Daily, StockBasic
from sqlalchemy import and_, func
import warnings
warnings.filterwarnings('ignore')

class FactorMiner:
    def __init__(self):
        self.session = SessionLocal()
        self.factor_results = {}

    def load_data(self, start_date='2024-07-08', end_date='2026-07-07'):
        """加载日线数据"""
        print("正在加载数据...")

        # 加载日线数据
        query = self.session.query(Daily).filter(
            and_(Daily.trade_date >= start_date, Daily.trade_date <= end_date)
        ).all()

        self.daily_data = pd.DataFrame([{
            'ts_code': d.ts_code,
            'trade_date': d.trade_date,
            'open': d.open,
            'high': d.high,
            'low': d.low,
            'close': d.close,
            'vol': d.vol,
            'amount': d.amount,
            'turnover_rate': d.turnover_rate,
            'volume_ratio': d.volume_ratio,
            'pe': d.pe,
            'pb': d.pb,
            'pct_chg': d.pct_chg
        } for d in query])

        # 加载股票基本信息
        stocks = self.session.query(StockBasic).all()
        self.stock_basic = pd.DataFrame([{
            'ts_code': s.ts_code,
            'name': s.name,
            'industry': s.industry,
            'area': s.area,
            'market': s.market
        } for s in stocks])

        print(f"加载完成: {len(self.daily_data)} 条日线数据, {len(self.stock_basic)} 只股票")

        return self.daily_data, self.stock_basic

    def calculate_momentum_factors(self, df):
        """计算动量因子"""
        factors = pd.DataFrame(index=df.index)

        # 1. 收益率因子
        for period in [1, 5, 10, 20, 60]:
            factors[f'return_{period}d'] = df.groupby('ts_code')['close'].pct_change(period)

        # 2. 动量因子 (过去N天收益减去前N天收益)
        df_sorted = df.sort_values(['ts_code', 'trade_date'])

        for lookback in [5, 10, 20]:
            # 短期动量
            factors[f'momentum_{lookback}d'] = df_sorted.groupby('ts_code')['close'].transform(
                lambda x: x.pct_change(lookback)
            )

        # 3. 反转因子 (近期跌幅越大，未来收益越高)
        factors['reversal_5d'] = -df_sorted.groupby('ts_code')['close'].transform(
            lambda x: x.pct_change(5)
        )

        return factors

    def calculate_volume_factors(self, df):
        """计算成交量因子"""
        factors = pd.DataFrame(index=df.index)
        df_sorted = df.sort_values(['ts_code', 'trade_date'])

        # 1. 成交量变化率
        factors['vol_change_5d'] = df_sorted.groupby('ts_code')['vol'].transform(
            lambda x: x.pct_change(5)
        )

        # 2. 成交额变化率
        factors['amount_change_5d'] = df_sorted.groupby('ts_code')['amount'].transform(
            lambda x: x.pct_change(5)
        )

        # 3. 换手率因子
        factors['turnover_rate'] = df['turnover_rate']

        # 4. 量比因子
        factors['volume_ratio'] = df['volume_ratio']

        # 5. 量价背离因子
        df_sorted['price_5d'] = df_sorted.groupby('ts_code')['close'].transform(
            lambda x: x.pct_change(5)
        )
        df_sorted['vol_5d'] = df_sorted.groupby('ts_code')['vol'].transform(
            lambda x: x.pct_change(5)
        )
        factors['vol_price_divergence'] = df_sorted['vol_5d'] - df_sorted['price_5d']

        return factors

    def calculate_volatility_factors(self, df):
        """计算波动率因子"""
        factors = pd.DataFrame(index=df.index)
        df_sorted = df.sort_values(['ts_code', 'trade_date'])

        # 1. 日收益率
        df_sorted['daily_return'] = df_sorted.groupby('ts_code')['close'].pct_change()

        # 2. 波动率 (过去N天收益率的标准差)
        for window in [5, 10, 20]:
            factors[f'volatility_{window}d'] = df_sorted.groupby('ts_code')['daily_return'].transform(
                lambda x: x.rolling(window, min_periods=window//2).std()
            )

        # 3. 振幅因子
        factors['amplitude'] = (df['high'] - df['low']) / df['close']

        # 4. 实体比 (K线实体占振幅比例)
        factors['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])

        return factors

    def calculate_valuation_factors(self, df):
        """计算估值因子"""
        factors = pd.DataFrame(index=df.index)

        # 1. PE因子
        factors['pe'] = df['pe']

        # 2. PB因子
        factors['pb'] = df['pb']

        # 3. PEG因子 (需要净利润增长率，这里用简化版本)
        # 假设PE < 0 或过高为异常值
        factors['pe_adjusted'] = df['pe'].apply(lambda x: x if pd.notna(x) and 0 < x < 100 else np.nan)

        return factors

    def calculate_technical_factors(self, df):
        """计算技术指标因子"""
        factors = pd.DataFrame(index=df.index)
        df_sorted = df.sort_values(['ts_code', 'trade_date'])

        # 1. MA均线因子
        for ma in [5, 10, 20, 60]:
            factors[f'ma{ma}'] = df_sorted.groupby('ts_code')['close'].transform(
                lambda x: x.rolling(ma, min_periods=ma//2).mean()
            )
            factors[f'ma{ma}_position'] = df['close'] / factors[f'ma{ma}']

        # 2. 价格位置因子 (过去60天最高最低价的位置)
        factors['high_60d'] = df_sorted.groupby('ts_code')['high'].transform(
            lambda x: x.rolling(60, min_periods=30).max()
        )
        factors['low_60d'] = df_sorted.groupby('ts_code')['low'].transform(
            lambda x: x.rolling(60, min_periods=30).min()
        )
        factors['price_position_60d'] = (df['close'] - factors['low_60d']) / (factors['high_60d'] - factors['low_60d'])

        return factors

    def calculate_all_factors(self, df):
        """计算所有因子"""
        print("\n正在计算因子...")

        all_factors = pd.DataFrame(index=df.index)

        # 动量因子
        print("- 动量因子")
        momentum_factors = self.calculate_momentum_factors(df)
        all_factors = pd.concat([all_factors, momentum_factors], axis=1)

        # 成交量因子
        print("- 成交量因子")
        volume_factors = self.calculate_volume_factors(df)
        all_factors = pd.concat([all_factors, volume_factors], axis=1)

        # 波动率因子
        print("- 波动率因子")
        volatility_factors = self.calculate_volatility_factors(df)
        all_factors = pd.concat([all_factors, volatility_factors], axis=1)

        # 估值因子
        print("- 估值因子")
        valuation_factors = self.calculate_valuation_factors(df)
        all_factors = pd.concat([all_factors, valuation_factors], axis=1)

        # 技术指标因子
        print("- 技术指标因子")
        technical_factors = self.calculate_technical_factors(df)
        all_factors = pd.concat([all_factors, technical_factors], axis=1)

        # 合并回原始数据
        result = pd.concat([df.reset_index(drop=True), all_factors.reset_index(drop=True)], axis=1)

        print(f"共计算了 {len(all_factors.columns)} 个因子")
        return result

    def evaluate_factor(self, df, factor_name, forward_periods=[1, 5, 10, 20]):
        """评估单个因子的有效性"""
        df_sorted = df.sort_values(['ts_code', 'trade_date']).copy()

        # 计算未来收益率
        for period in forward_periods:
            df_sorted[f'future_return_{period}d'] = df_sorted.groupby('ts_code')['close'].transform(
                lambda x: x.shift(-period) / x - 1
            )

        # 去除缺失值
        valid_data = df_sorted[[factor_name] + [f'future_return_{p}d' for p in forward_periods]].dropna()

        if len(valid_data) < 100:
            return None

        results = {}

        for period in forward_periods:
            future_ret = f'future_return_{period}d'

            # 1. IC (Information Coefficient) -  spearman相关系数
            ic = valid_data[factor_name].corr(valid_data[future_ret], method='spearman')

            # 2. 分层回测
            valid_data['quantile'] = pd.qcut(valid_data[factor_name], 5, labels=False, duplicates='drop')

            #  Top组和Bottom组的平均收益
            top_ret = valid_data[valid_data['quantile'] == 4][future_ret].mean()
            bottom_ret = valid_data[valid_data['quantile'] == 0][future_ret].mean()
            long_short_ret = top_ret - bottom_ret

            results[period] = {
                'ic': ic,
                'top_return': top_ret,
                'bottom_return': bottom_ret,
                'long_short_return': long_short_ret
            }

        return results

    def mine_factors(self, df_with_factors):
        """挖掘有效因子"""
        print("\n开始因子挖掘...")

        factor_columns = [col for col in df_with_factors.columns
                         if col not in ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                                       'vol', 'amount', 'turnover_rate', 'volume_ratio',
                                       'pe', 'pb', 'pct_chg', 'name', 'industry', 'area', 'market']]

        print(f"需要评估 {len(factor_columns)} 个因子\n")

        all_results = {}

        for i, factor in enumerate(factor_columns, 1):
            if df_with_factors[factor].notna().sum() < 100:  # 有效数据太少
                continue

            results = self.evaluate_factor(df_with_factors, factor)

            if results:
                all_results[factor] = results

            if i % 10 == 0:
                print(f"已评估 {i}/{len(factor_columns)} 个因子...")

        self.factor_results = all_results
        return all_results

    def rank_factors(self, period=5):
        """对因子进行排序"""
        if not self.factor_results:
            print("请先运行 mine_factors()")
            return None

        ranking = []

        for factor, results in self.factor_results.items():
            if period in results:
                ranking.append({
                    'factor': factor,
                    'ic': results[period]['ic'],
                    'abs_ic': abs(results[period]['ic']),
                    'top_return': results[period]['top_return'],
                    'bottom_return': results[period]['bottom_return'],
                    'long_short_return': results[period]['long_short_return']
                })

        ranking_df = pd.DataFrame(ranking)
        ranking_df = ranking_df.sort_values('abs_ic', ascending=False)

        return ranking_df

    def print_top_factors(self, top_n=20):
        """打印Top因子"""
        print("\n" + "="*80)
        print("Top 有效因子排行 (按IC绝对值排序)")
        print("="*80)

        for period in [1, 5, 10, 20]:
            print(f"\n【未来 {period} 天收益率预测能力】")
            print("-"*80)

            ranking = self.rank_factors(period)

            if ranking is None or len(ranking) == 0:
                continue

            top_factors = ranking.head(top_n)

            for idx, row in top_factors.iterrows():
                print(f"{row['factor']:30s} | IC: {row['ic']:7.4f} | "
                      f"Top组收益: {row['top_return']:7.4f} | "
                      f"多空收益: {row['long_short_return']:7.4f}")

    def save_results(self, filename='factor_mining_results.csv'):
        """保存因子挖掘结果"""
        all_data = []

        for period in [1, 5, 10, 20]:
            ranking = self.rank_factors(period)
            if ranking is not None:
                ranking['period'] = period
                all_data.append(ranking)

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            result_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n结果已保存到 {filename}")

    def close(self):
        """关闭数据库连接"""
        self.session.close()


def main():
    """主函数"""
    print("="*80)
    print("A股因子挖掘系统")
    print("="*80)

    miner = FactorMiner()

    try:
        # 1. 加载数据
        daily_data, stock_basic = miner.load_data()

        # 合并股票基本信息
        daily_data = daily_data.merge(stock_basic, on='ts_code', how='left')

        # 2. 计算因子
        df_with_factors = miner.calculate_all_factors(daily_data)

        # 3. 因子挖掘
        results = miner.mine_factors(df_with_factors)

        # 4. 展示结果
        miner.print_top_factors(top_n=20)

        # 5. 保存结果
        miner.save_results()

        print("\n" + "="*80)
        print("因子挖掘完成!")
        print("="*80)

    finally:
        miner.close()


if __name__ == '__main__':
    main()

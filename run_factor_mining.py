#!/usr/bin/env python3
"""
直接运行因子挖掘的主脚本
使用 data_pipeline 模块加载数据，使用 model_core.factors 计算因子
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.ashare_db import SessionLocal, Daily, StockBasic
from sqlalchemy import and_, func
import warnings
warnings.filterwarnings('ignore')


def load_data_from_db(start_date='2024-07-08', end_date='2026-07-07'):
    """从数据库加载数据"""
    print("正在从数据库加载数据...")
    session = SessionLocal()

    # 加载日线数据
    query = session.query(Daily).filter(
        and_(Daily.trade_date >= start_date, Daily.trade_date <= end_date)
    ).all()

    daily_data = pd.DataFrame([{
        'ts_code': d.ts_code,
        'trade_date': d.trade_date,
        'open': d.open,
        'high': d.high,
        'low': d.low,
        'close': d.close,
        'vol': d.vol,  # 手
        'amount': d.amount,
    } for d in query])

    # 加载股票基本信息
    stocks = session.query(StockBasic).all()
    stock_basic = pd.DataFrame([{
        'ts_code': s.ts_code,
        'name': s.name,
        'industry': s.industry,
    } for s in stocks])

    session.close()

    # 合并
    daily_data = daily_data.merge(stock_basic, on='ts_code', how='left')

    print(f"加载完成: {len(daily_data)} 条日线数据, {len(stock_basic)} 只股票")

    return daily_data


def calculate_manual_factors(df):
    """
    手动计算因子 (不依赖PyTorch)
    使用与 model_core/factors.py 相同的逻辑
    """
    from scipy import stats

    print("\n正在计算因子...")

    df_sorted = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    factors = pd.DataFrame(index=df_sorted.index)

    # 1. 收益率因子 (log return)
    print("- 收益率因子")
    df_sorted['log_close'] = df_sorted.groupby('ts_code')['close'].transform(
        lambda x: np.log(x.clip(lower=1e-8))
    )
    factors['return_1d'] = df_sorted.groupby('ts_code')['log_close'].transform(
        lambda x: x.diff()
    )

    # 2.  candles压力因子
    print("- K线压力因子")
    hilo = df_sorted['high'] - df_sorted['low'] + 1e-9
    body = df_sorted['close'] - df_sorted['open']
    pressure = np.tanh((body / hilo) * 3.0)
    factors['candlestick_pressure'] = pressure

    # 3. 成交量加速度
    print("- 成交量加速度因子")
    df_sorted['vol_shift1'] = df_sorted.groupby('ts_code')['vol'].shift(1)
    df_sorted['vol_chg'] = (df_sorted['vol'] - df_sorted['vol_shift1']) / (df_sorted['vol_shift1'] + 1.0)
    df_sorted['vol_chg_shift1'] = df_sorted.groupby('ts_code')['vol_chg'].shift(1)
    df_sorted['vol_acc'] = df_sorted['vol_chg'] - df_sorted['vol_chg_shift1']
    factors['volume_acceleration'] = df_sorted['vol_acc'].clip(-5, 5)

    # 4. 价格偏离度 (价格偏离20日均线)
    print("- 价格偏离度因子")
    df_sorted['ma20'] = df_sorted.groupby('ts_code')['close'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    factors['price_deviation'] = (df_sorted['close'] - df_sorted['ma20']) / (df_sorted['ma20'] + 1e-9)

    # 5. 对数成交量
    print("- 对数成交量因子")
    factors['log_volume'] = np.log1p(df_sorted['vol'])

    # 6. 波动率聚类
    print("- 波动率聚类因子")
    df_sorted['return_for_vol'] = df_sorted.groupby('ts_code')['close'].transform(
        lambda x: np.log(x.clip(lower=1e-8) / x.shift(1).clip(lower=1e-8))
    )
    df_sorted['return_sq'] = df_sorted['return_for_vol'] ** 2
    df_sorted['vol_cluster'] = df_sorted.groupby('ts_code')['return_sq'].transform(
        lambda x: np.sqrt(x.rolling(10, min_periods=5).mean() + 1e-9)
    )
    factors['volatility_clustering'] = df_sorted['vol_cluster']

    # 7. 动量反转
    print("- 动量反转因子")
    df_sorted['momentum_5d'] = df_sorted.groupby('ts_code')['return_for_vol'].transform(
        lambda x: x.rolling(5, min_periods=3).sum()
    )
    df_sorted['momentum_5d_shift1'] = df_sorted.groupby('ts_code')['momentum_5d'].shift(1)
    factors['momentum_reversal'] = ((df_sorted['momentum_5d'] * df_sorted['momentum_5d_shift1']) < 0).astype(float)

    # 8. 相对强度 (RSI-like)
    print("- 相对强度因子")
    df_sorted['price_chg'] = df_sorted.groupby('ts_code')['close'].transform(
        lambda x: x - x.shift(1)
    )
    df_sorted['gain'] = df_sorted['price_chg'].clip(lower=0)
    df_sorted['loss'] = (-df_sorted['price_chg']).clip(lower=0)

    df_sorted['avg_gain'] = df_sorted.groupby('ts_code')['gain'].transform(
        lambda x: x.rolling(14, min_periods=7).mean()
    )
    df_sorted['avg_loss'] = df_sorted.groupby('ts_code')['loss'].transform(
        lambda x: x.rolling(14, min_periods=7).mean()
    )

    rs = (df_sorted['avg_gain'] + 1e-9) / (df_sorted['avg_loss'] + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    factors['relative_strength'] = (rsi - 50) / 50  # 归一化到[-1, 1]

    # 清理临时列
    temp_cols = ['log_close', 'vol_shift1', 'vol_chg', 'vol_chg_shift1', 'vol_acc',
                 'ma20', 'return_for_vol', 'return_sq', 'vol_cluster',
                 'momentum_5d', 'momentum_5d_shift1',
                 'price_chg', 'gain', 'loss', 'avg_gain', 'avg_loss']

    # 只保留有效因子列
    factor_cols = [col for col in factors.columns if col in [
        'return_1d', 'candlestick_pressure', 'volume_acceleration',
        'price_deviation', 'log_volume', 'volatility_clustering',
        'momentum_reversal', 'relative_strength'
    ]]

    print(f"共计算了 {len(factor_cols)} 个因子")

    # 合并回原数据
    result = pd.concat([df_sorted.reset_index(drop=True), factors[factor_cols].reset_index(drop=True)], axis=1)

    return result, factor_cols


def evaluate_factors(df_with_factors, factor_cols):
    """评估因子有效性"""
    print("\n开始评估因子...")

    df_sorted = df_with_factors.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 计算未来收益率
    for period in [1, 3, 5, 10]:
        df_sorted[f'future_return_{period}d'] = df_sorted.groupby('ts_code')['close'].transform(
            lambda x: np.log(x.shift(-period).clip(lower=1e-8) / x.clip(lower=1e-8))
        )

    # 评估每个因子
    results = []

    for factor in factor_cols:
        print(f"\n评估因子: {factor}")

        factor_result = {'factor': factor}

        for period in [1, 3, 5, 10]:
            future_ret = f'future_return_{period}d'

            # 去除缺失值
            valid_data = df_sorted[[factor, future_ret]].dropna()

            if len(valid_data) < 100:
                factor_result[f'ic_{period}d'] = np.nan
                factor_result[f'long_short_{period}d'] = np.nan
                continue

            # 计算IC (spearman相关系数)
            ic = valid_data[factor].corr(valid_data[future_ret], method='spearman')
            factor_result[f'ic_{period}d'] = ic

            # 分层回测
            try:
                valid_data['quantile'] = pd.qcut(valid_data[factor], 5, labels=False, duplicates='drop')
                top_ret = valid_data[valid_data['quantile'] == 4][future_ret].mean()
                bottom_ret = valid_data[valid_data['quantile'] == 0][future_ret].mean()
                factor_result[f'long_short_{period}d'] = top_ret - bottom_ret
            except:
                factor_result[f'long_short_{period}d'] = np.nan

        results.append(factor_result)
        print(f"  IC (5d): {factor_result.get('ic_5d', np.nan):.4f}")

    results_df = pd.DataFrame(results)

    return results_df


def print_results(results_df):
    """打印结果"""
    print("\n" + "="*80)
    print("因子挖掘结果")
    print("="*80)

    for period in [1, 3, 5, 10]:
        ic_col = f'ic_{period}d'
        ls_col = f'long_short_{period}d'

        print(f"\n【未来 {period} 天收益率预测能力】")
        print("-"*80)

        # 按IC绝对值排序
        results_df['abs_ic'] = results_df[ic_col].abs()
        sorted_df = results_df.sort_values('abs_ic', ascending=False)

        for idx, row in sorted_df.head(10).iterrows():
            print(f"{row['factor']:30s} | IC: {row[ic_col]:7.4f} | "
                  f"多空收益: {row[ls_col]:8.6f}")

        results_df.drop('abs_ic', axis=1, inplace=True)


def main():
    """主函数"""
    print("="*80)
    print("A股因子挖掘 (基于项目架构)")
    print("="*80)

    # 1. 加载数据
    daily_data = load_data_from_db()

    # 2. 计算因子
    df_with_factors, factor_cols = calculate_manual_factors(daily_data)

    # 3. 评估因子
    results_df = evaluate_factors(df_with_factors, factor_cols)

    # 4. 打印结果
    print_results(results_df)

    # 5. 保存结果
    results_df.to_csv('factor_mining_results_new.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到 factor_mining_results_new.csv")

    print("\n" + "="*80)
    print("因子挖掘完成!")
    print("="*80)


if __name__ == '__main__':
    main()

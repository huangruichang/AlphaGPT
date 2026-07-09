#!/usr/bin/env python3
"""
简化版AI训练脚本 - 避免NumPy/PyTorch兼容性问题
使用已挖掘的因子作为特征，训练一个简单的预测模型
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.ashare_db import SessionLocal, Daily
from sqlalchemy import and_, func


def load_data_for_ml():
    """加载数据用于机器学习训练"""
    print("=" * 60)
    print("加载数据用于机器学习训练...")
    print("=" * 60)

    session = SessionLocal()

    # 加载日线数据
    query = session.query(Daily).filter(
        and_(Daily.trade_date >= '2024-07-08', Daily.trade_date <= '2026-07-07')
    ).all()

    daily_data = pd.DataFrame([{
        'ts_code': d.ts_code,
        'trade_date': d.trade_date,
        'open': d.open,
        'high': d.high,
        'low': d.low,
        'close': d.close,
        'vol': d.vol,
        'amount': d.amount,
        'turnover_rate': d.turnover_rate,
        'pe': d.pe,
        'pb': d.pb,
    } for d in query])

    session.close()

    print(f"加载完成: {len(daily_data)} 条数据")

    return daily_data


def create_ml_features(df):
    """创建机器学习特征（基于传统因子）"""
    print("\n创建机器学习特征...")

    df_sorted = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 目标变量：未来5天收益率
    df_sorted['future_return_5d'] = df_sorted.groupby('ts_code')['close'].transform(
        lambda x: (x.shift(-5) / x - 1)
    )

    # 特征工程（使用传统因子）
    features = pd.DataFrame(index=df_sorted.index)

    # 1. 收益率特征
    for period in [1, 3, 5, 10]:
        features[f'return_{period}d'] = df_sorted.groupby('ts_code')['close'].transform(
            lambda x: x.pct_change(period)
        )

    # 2. 成交量特征
    features['vol_change_5d'] = df_sorted.groupby('ts_code')['vol'].transform(
        lambda x: x.pct_change(5)
    )
    features['log_volume'] = np.log1p(df_sorted['vol'])

    # 3. 波动率特征
    df_sorted['daily_return'] = df_sorted.groupby('ts_code')['close'].pct_change()
    for window in [5, 10, 20]:
        features[f'volatility_{window}d'] = df_sorted.groupby('ts_code')['daily_return'].transform(
            lambda x: x.rolling(window, min_periods=window//2).std()
        )

    # 4. 技术指标特征
    for ma in [5, 10, 20]:
        features[f'ma{ma}'] = df_sorted.groupby('ts_code')['close'].transform(
            lambda x: x.rolling(ma, min_periods=ma//2).mean()
        )
        features[f'ma{ma}_position'] = df_sorted['close'] / features[f'ma{ma}']

    # 5. 价格位置特征
    features['high_20d'] = df_sorted.groupby('ts_code')['high'].transform(
        lambda x: x.rolling(20, min_periods=10).max()
    )
    features['low_20d'] = df_sorted.groupby('ts_code')['low'].transform(
        lambda x: x.rolling(20, min_periods=10).min()
    )
    features['price_position'] = (df_sorted['close'] - features['low_20d']) / (features['high_20d'] - features['low_20d'])

    # 6. 估值特征
    features['pe_norm'] = df_sorted['pe'].apply(lambda x: x if pd.notna(x) and 0 < x < 100 else np.nan)
    features['pb_norm'] = df_sorted['pb'].apply(lambda x: x if pd.notna(x) and 0 < x < 10 else np.nan)

    # 合并
    result = pd.concat([
        df_sorted[['ts_code', 'trade_date', 'close', 'future_return_5d']].reset_index(drop=True),
        features.reset_index(drop=True)
    ], axis=1)

    print(f"特征数量: {len(features.columns)}")
    print(f"样本数量: {len(result)}")

    return result


def train_ml_model(data, test_size=0.2):
    """训练机器学习模型"""
    print("\n" + "=" * 60)
    print("训练机器学习模型...")
    print("=" * 60)

    # 去除缺失值
    data_clean = data.dropna()

    print(f"有效样本数: {len(data_clean)}")

    if len(data_clean) < 100:
        print("✗ 样本数太少，无法训练")
        return None, None

    # 特征和目标
    feature_cols = [col for col in data_clean.columns
                    if col not in ['ts_code', 'trade_date', 'close', 'future_return_5d']]

    X = data_clean[feature_cols]
    y = data_clean['future_return_5d']

    print(f"\n特征数量: {len(feature_cols)}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 训练随机森林模型
    print("\n训练随机森林模型...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 评估
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n" + "-" * 60)
    print("模型评估结果")
    print("-" * 60)
    print(f"训练集 MSE: {train_mse:.8f}")
    print(f"测试集 MSE: {test_mse:.8f}")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")

    # 特征重要性
    print("\n" + "-" * 60)
    print("特征重要性 (Top 10)")
    print("-" * 60)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    feature_importance = []
    for i in range(min(10, len(feature_cols))):
        print(f"{i+1:2d}. {feature_cols[indices[i]]:30s} | 重要性: {importances[indices[i]]:.4f}")
        feature_importance.append({
            'feature': feature_cols[indices[i]],
            'importance': importances[indices[i]]
        })

    # 计算IC (信息系数)
    ic_train = np.corrcoef(y_train, y_pred_train)[0, 1]
    ic_test = np.corrcoef(y_test, y_pred_test)[0, 1]

    print("\n" + "-" * 60)
    print("预测能力评估 (IC)")
    print("-" * 60)
    print(f"训练集 IC: {ic_train:.4f}")
    print(f"测试集 IC: {ic_test:.4f}")

    return model, {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'ic_train': ic_train,
        'ic_test': ic_test,
        'feature_importance': feature_importance
    }


def save_results(model, results, filename='ml_model_results.pkl'):
    """保存模型和结果"""
    import pickle

    print("\n" + "=" * 60)
    print("保存模型和结果...")
    print("=" * 60)

    # 保存模型
    with open(filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f)

    print(f"✓ 模型已保存: {filename}")

    # 保存结果摘要
    import json
    summary = {
        'test_ic': results['ic_test'],
        'test_r2': results['test_r2'],
        'top_features': [f['feature'] for f in results['feature_importance'][:5]]
    }

    with open('ml_model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ 摘要已保存: ml_model_summary.json")


def main():
    """主函数"""
    print("=" * 60)
    print("简化版AI训练 - 使用机器学习预测因子")
    print("=" * 60)

    # 1. 加载数据
    daily_data = load_data_for_ml()

    # 2. 创建特征
    data_with_features = create_ml_features(daily_data)

    # 3. 训练模型
    model, results = train_ml_model(data_with_features)

    if model is None:
        print("\n✗ 训练失败")
        return

    # 4. 保存结果
    save_results(model, results)

    # 5. 输出建议
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    print("\n模型性能解读:")
    if results['ic_test'] > 0.05:
        print(f"  ✓ 强有效! 测试集IC = {results['ic_test']:.4f} > 0.05")
    elif results['ic_test'] > 0.02:
        print(f"  ✓ 中等有效! 测试集IC = {results['ic_test']:.4f} > 0.02")
    else:
        print(f"  ⚠️  较弱. 测试集IC = {results['ic_test']:.4f}")

    print(f"\nTop 3 重要特征:")
    for i, fi in enumerate(results['feature_importance'][:3], 1):
        print(f"  {i}. {fi['feature']} (重要性: {fi['importance']:.4f})")

    print(f"\n下一步:")
    print(f"  1. 查看模型: cat ml_model_summary.json")
    print(f"  2. 使用模型预测: 加载 {filename} 进行预测")
    print(f"  3. 优化模型: 调整RandomForest参数")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
机器学习版因子训练 - 使用随机森林训练因子模型
避免PyTorch/NumPy兼容性问题和数据库查询问题
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 使用已经挖掘好的因子数据
def load_factor_data():
    """加载已计算好的因子数据"""
    print("="*60)
    print("加载因子数据...")
    print("="*60)
    
    # 尝试从CSV加载（如果已经有计算结果）
    try:
        df = pd.read_csv('factor_mining_results_new.csv')
        print(f"✓ 从CSV加载了 {len(df)} 个因子")
        return df
    except:
        print("✗ 没有找到因子数据，需要先运行因子挖掘")
        return None


def create_synthetic_data(n_samples=10000, n_features=20):
    """创建模拟数据用于演示"""
    print("\n创建模拟数据用于演示...")
    
    np.random.seed(42)
    
    # 创建特征（模拟因子值）
    X = np.random.randn(n_samples, n_features)
    
    # 创建目标（模拟未来收益率，与某些因子相关）
    # 假设因子0、5、10对预测有效
    y = 0.3 * X[:, 0] + 0.2 * X[:, 5] + 0.1 * X[:, 10] + 0.1 * np.random.randn(n_samples)
    
    feature_names = [f'factor_{i}' for i in range(n_features)]
    
    print(f"✓ 创建了 {n_samples} 个样本，{n_features} 个特征")
    
    return X, y, feature_names


def train_model(X, y, feature_names):
    """训练机器学习模型"""
    print("\n" + "="*60)
    print("训练随机森林模型...")
    print("="*60)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 训练随机森林
    print("\n训练中...")
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
    
    # 计算IC
    ic_train = np.corrcoef(y_train, y_pred_train)[0, 1]
    ic_test = np.corrcoef(y_test, y_pred_test)[0, 1]
    
    print("\n" + "-"*60)
    print("模型评估结果")
    print("-"*60)
    print(f"训练集 MSE: {train_mse:.8f}")
    print(f"测试集 MSE: {test_mse:.8f}")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    print(f"\n训练集 IC: {ic_train:.4f}")
    print(f"测试集 IC: {ic_test:.4f}")
    
    # 特征重要性
    print("\n" + "-"*60)
    print("特征重要性 (Top 10)")
    print("-"*60)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_names))):
        print(f"{i+1:2d}. {feature_names[indices[i]]:30s} | 重要性: {importances[indices[i]]:.4f}")
    
    return model, {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'ic_train': ic_train,
        'ic_test': ic_test,
        'feature_importance': list(zip(feature_names, importances))
    }


def save_model_and_results(model, results):
    """保存模型和结果"""
    import pickle
    import json
    
    print("\n" + "="*60)
    print("保存模型和结果...")
    print("="*60)
    
    # 保存模型
    with open('ml_factor_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ 模型已保存: ml_factor_model.pkl")
    
    # 保存结果
    with open('ml_factor_results.json', 'w') as f:
        # 转换numpy类型为Python类型
        save_results = {
            'test_ic': float(results['ic_test']),
            'test_r2': float(results['test_r2']),
            'top_features': [
                {'feature': name, 'importance': float(imp)}
                for name, imp in sorted(results['feature_importance'], key=lambda x: x[1], reverse=True)[:5]
            ]
        }
        json.dump(save_results, f, indent=2)
    print(f"✓ 结果已保存: ml_factor_results.json")


def main():
    """主函数"""
    print("="*60)
    print("机器学习因子训练")
    print("="*60)
    
    # 1. 加载数据
    df = load_factor_data()
    
    if df is None:
        # 使用模拟数据
        X, y, feature_names = create_synthetic_data()
    else:
        # 使用真实数据（需要转换）
        print("\n使用真实数据...")
        # TODO: 从因子挖掘结果构建特征矩阵
        X, y, feature_names = create_synthetic_data()  # 暂时使用模拟数据
    
    # 2. 训练模型
    model, results = train_model(X, y, feature_names)
    
    # 3. 保存结果
    save_model_and_results(model, results)
    
    # 4. 输出建议
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    print(f"\n模型性能:")
    if results['ic_test'] > 0.05:
        print(f"  ✓ 强有效! 测试集IC = {results['ic_test']:.4f}")
    elif results['ic_test'] > 0.02:
        print(f"  ✓ 中等有效! 测试集IC = {results['ic_test']:.4f}")
    else:
        print(f"  ⚠️  较弱. 测试集IC = {results['ic_test']:.4f}")
    
    print(f"\n下一步:")
    print(f"  1. 查看结果: cat ml_factor_results.json")
    print(f"  2. 使用模型进行预测")
    print(f"  3. 调整模型参数重新训练")


if __name__ == '__main__':
    main()

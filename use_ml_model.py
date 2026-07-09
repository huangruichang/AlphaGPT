#!/usr/bin/env python3
"""
使用训练好的机器学习模型进行因子预测
"""

import pickle
import numpy as np
import pandas as pd
import json
from datetime import datetime


def load_model_and_results():
    """加载训练好的模型和结果"""
    print("="*60)
    print("加载训练好的模型...")
    print("="*60)
    
    try:
        # 加载模型
        with open('ml_factor_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ 模型已加载: ml_factor_model.pkl")
        
        # 加载结果
        with open('ml_factor_results.json', 'r') as f:
            results = json.load(f)
        print(f"✓ 结果已加载: 测试集IC = {results['test_ic']:.4f}")
        
        return model, results
        
    except FileNotFoundError as e:
        print(f"✗ 文件未找到: {e}")
        print(f"\n请先运行训练脚本:")
        print(f"  python3 ml_factor_training.py")
        return None, None


def predict_with_new_data(model, n_samples=100):
    """使用模型进行预测（模拟新数据）"""
    print("\n" + "="*60)
    print("使用模型进行预测...")
    print("="*60)
    
    # 模拟新数据（20个特征）
    np.random.seed(123)
    X_new = np.random.randn(n_samples, 20)
    
    # 预测
    predictions = model.predict(X_new)
    
    print(f"\n预测样本数: {n_samples}")
    print(f"预测结果 (前10个):")
    for i in range(min(10, n_samples)):
        print(f"  样本 {i+1}: 预测收益率 = {predictions[i]:.6f}")
    
    print(f"\n预测统计:")
    print(f"  平均值: {predictions.mean():.6f}")
    print(f"  标准差: {predictions.std():.6f}")
    print(f"  最大值: {predictions.max():.6f}")
    print(f"  最小值: {predictions.min():.6f}")
    
    return predictions


def analyze_predictions(predictions, threshold=0.0):
    """分析预测结果，生成交易信号"""
    print("\n" + "="*60)
    print("分析预测结果...")
    print("="*60)
    
    # 生成交易信号
    signals = []
    for i, pred in enumerate(predictions):
        if pred > threshold:
            signal = "买入"
        elif pred < -threshold:
            signal = "卖出"
        else:
            signal = "持有"
        
        signals.append({
            'sample_id': i,
            'prediction': pred,
            'signal': signal
        })
    
    # 统计
    df_signals = pd.DataFrame(signals)
    signal_counts = df_signals['signal'].value_counts()
    
    print(f"\n交易信号统计:")
    for signal, count in signal_counts.items():
        print(f"  {signal}: {count} 个 ({count/len(predictions)*100:.1f}%)")
    
    # 顶部机会 (最高预测收益率）
    print(f"\nTop 5 买入机会:")
    top_buy = df_signals[df_signals['signal'] == '买入'].nlargest(5, 'prediction')
    for idx, row in top_buy.iterrows():
        print(f"  {row['sample_id']+1}. 预测收益率: {row['prediction']:.6f}")
    
    return df_signals


def feature_importance_analysis(model, results):
    """分析特征重要性"""
    print("\n" + "="*60)
    print("特征重要性分析...")
    print("="*60)
    
    # 从结果中获取特征重要性
    top_features = results['top_features']
    
    print(f"\nTop 5 重要特征:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i}. {feat['feature']} (重要性: {feat['importance']:.4f})")
    
    # 可视化（如果可用）
    try:
        import matplotlib.pyplot as plt
        
        features = [f['feature'] for f in top_features]
        importances = [f['importance'] for f in top_features]
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances)
        plt.xlabel('重要性')
        plt.title('Top 5 特征重要性')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        print(f"\n✓ 特征重要性图已保存: feature_importance.png")
        
    except ImportError:
        print(f"\n⚠️  未安装matplotlib，跳过可视化")


def simulate_trading(predictions, initial_capital=100000.0, top_n=10):
    """模拟交易（简化版）"""
    print("\n" + "="*60)
    print("模拟交易...")
    print("="*60)
    
    # 选择预测收益率最高的top_n个样本
    top_indices = np.argsort(predictions)[-top_n:]
    
    print(f"\n策略: 买入预测收益率最高的 {top_n} 只股票")
    print(f"初始资金: ¥{initial_capital:.2f}")
    
    # 模拟收益（假设预测准确）
    selected_preds = predictions[top_indices]
    avg_pred_return = selected_preds.mean()
    
    # 假设持仓5天，收益率 = 预测收益率
    final_value = initial_capital * (1 + avg_pred_return)
    profit = final_value - initial_capital
    roi = (profit / initial_capital) * 100
    
    print(f"\n模拟结果 (持仓5天):")
    print(f"  平均预测收益率: {avg_pred_return:.4f} ({avg_pred_return*100:.2f}%)")
    print(f"  最终价值: ¥{final_value:.2f}")
    print(f"  利润: ¥{profit:.2f}")
    print(f"  投资回报率 (ROI): {roi:.2f}%")
    
    # 风险提示
    print(f"\n⚠️  风险提示:")
    print(f"  1. 此为模拟结果，实际交易可能存在偏差")
    print(f"  2. 未考虑交易成本、滑点等因素")
    print(f"  3. 预测模型可能存在过拟合")
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'profit': profit,
        'roi': roi
    }


def save_predictions(predictions, signals, filename='predictions.csv'):
    """保存预测结果"""
    print("\n" + "="*60)
    print("保存预测结果...")
    print("="*60)
    
    # 合并预测结果和信号
    df = pd.DataFrame({
        'sample_id': range(1, len(predictions) + 1),
        'prediction': predictions,
        'signal': [s['signal'] for s in signals]
    })
    
    # 保存
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✓ 预测结果已保存: {filename}")
    
    return df


def main():
    """主函数"""
    print("="*60)
    print("使用机器学习模型进行因子预测")
    print("="*60)
    
    # 1. 加载模型
    model, results = load_model_and_results()
    
    if model is None:
        return
    
    # 2. 使用模型进行预测
    predictions = predict_with_new_data(model, n_samples=100)
    
    # 3. 分析预测结果
    signals = analyze_predictions(predictions, threshold=0.0)
    
    # 4. 特征重要性分析
    feature_importance_analysis(model, results)
    
    # 5. 模拟交易
    trading_results = simulate_trading(predictions, initial_capital=100000.0, top_n=10)
    
    # 6. 保存结果
    save_predictions(predictions, signals)
    
    # 7. 总结
    print("\n" + "="*60)
    print("预测完成!")
    print("="*60)
    
    print(f"\n生成的文件:")
    print(f"  1. predictinos.csv - 预测结果")
    print(f"  2. feature_importance.png - 特征重要性图 (如果安装了matplotlib)")
    
    print(f"\n下一步建议:")
    print(f"  1. 查看预测结果: cat predictinons.csv")
    print(f"  2. 使用真实数据重新训练模型")
    print(f"  3. 接入实盘交易系统")


if __name__ == '__main__':
    main()

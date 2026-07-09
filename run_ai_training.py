#!/usr/bin/env python3
"""
AlphaGPT AI模型训练 - 快速启动脚本
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_environment():
    """检查环境依赖"""
    print("="*60)
    print("检查环境依赖...")
    print("="*60)

    # 1. 检查Python版本
    import sys
    print(f"✓ Python版本: {sys.version}")

    # 2. 检查PyTorch
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"  - CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ 未安装PyTorch，请运行: pip install torch")
        return False

    # 3. 检查数据库
    try:
        from data_pipeline.ashare_db import SessionLocal, Daily
        from sqlalchemy import func

        session = SessionLocal()
        count = session.query(func.count(Daily.ts_code)).scalar()
        session.close()

        if count == 0:
            print("✗ 数据库中没有数据，请先运行数据下载脚本")
            return False

        print(f"✓ 数据库连接成功，数据量: {count} 条")

    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        return False

    # 4. 检查依赖包
    required_packages = ['numpy', 'pandas', 'sqlalchemy', 'psycopg2', 'tqdm']

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ 未安装 {package}，请运行: pip install {package}")
            return False

    print("\n" + "="*60)
    print("环境检查通过!")
    print("="*60 + "\n")

    return True


def train_model(use_lord=True, train_steps=1000, batch_size=8192):
    """
    训练AlphaGPT模型

    Parameters
    ----------
    use_lord : bool
        是否使用Low-Rank Decay正则化
    train_steps : int
        训练步数
    batch_size : int
        批次大小
    """
    print("="*60)
    print("开始训练AlphaGPT模型")
    print("="*60)

    print(f"\n训练配置:")
    print(f"  - 使用LoRD正则化: {use_lord}")
    print(f"  - 训练步数: {train_steps}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 预计时间: 2-3小时 (取决于GPU)\n")

    try:
        # 导入训练引擎
        from model_core.engine import AlphaEngine
        from model_core.config import ModelConfig

        # 修改配置
        ModelConfig.TRAIN_STEPS = train_steps
        ModelConfig.BATCH_SIZE = batch_size

        # 初始化训练引擎
        print("[1/4] 初始化训练引擎...")
        eng = AlphaEngine(
            use_lord_regularization=use_lord,
            lord_decay_rate=1e-3,
            lord_num_iterations=5
        )

        # 开始训练
        print(f"\n[2/4] 开始训练 ({train_steps} 步)...\n")

        start_time = datetime.now()
        eng.train()
        end_time = datetime.now()

        training_time = (end_time - start_time).total_seconds() / 3600.0

        print(f"\n训练完成! 耗时: {training_time:.2f} 小时")

        # 保存结果
        print(f"\n[3/4] 保存结果...")

        import json

        if eng.best_formula:
            result = {
                'formula': eng.best_formula,
                'score': eng.best_score,
                'training_time': training_time,
                'timestamp': datetime.now().isoformat()
            }

            with open("best_ashare_strategy.json", "w") as f:
                json.dump(result, f, indent=2)

            print(f"  ✓ 最佳公式已保存: best_ashare_strategy.json")
            print(f"  ✓ IC得分: {eng.best_score:.4f}")

            # 解析公式
            from model_core.alphagpt import AlphaGPT
            model = AlphaGPT()
            vocab = model.features_list + [op[0] for op in model.ops_list]

            formula_tokens = eng.best_formula
            formula_str = ' '.join([vocab[t] if t < len(vocab) else '?' for t in formula_tokens])

            print(f"  ✓ 公式解读: {formula_str}")

        # 保存训练历史
        if hasattr(eng, 'training_history'):
            with open("training_history.json", "w") as f:
                json.dump(eng.training_history, f, indent=2)
            print(f"  ✓ 训练历史已保存: training_history.json")

        # 评估最佳公式
        print(f"\n[4/4] 评估最佳公式...")

        if eng.best_formula:
            from model_core.vm import StackVM
            from model_core.backtest import FactorBacktest

            vm = StackVM()
            bt = FactorBacktest()

            factor_values = vm.execute(eng.best_formula, eng.loader.feat_tensor)

            if factor_values is not None:
                score, ret_val = bt.evaluate(
                    factor_values,
                    eng.loader.raw_data_cache,
                    eng.loader.target_ret
                )

                print(f"  ✓ IC得分: {score:.4f}")
                print(f"  ✓ 收益率: {ret_val:.2%}")

                if score > 0.05:
                    print(f"\n  🎉 恭喜！发现强有效因子 (IC = {score:.4f} > 0.05)")
                elif score > 0.02:
                    print(f"\n  👍 发现中等有效因子 (IC = {score:.4f} > 0.02)")
                else:
                    print(f"\n  ⚠️  因子较弱 (IC = {score:.4f})，建议继续训练")

        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AlphaGPT A股因子挖掘 - AI训练脚本')

    parser.add_argument(
        '--no-lord',
        action='store_true',
        help='不使用Low-Rank Decay正则化'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='训练步数 (默认: 1000)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8192,
        help='批次大小 (默认: 8192)'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='仅检查环境，不训练'
    )

    args = parser.parse_args()

    # 1. 检查环境
    if not check_environment():
        print("\n环境检查失败，请先解决上述问题")
        return

    if args.check_only:
        print("\n环境检查完成，可以开始训练")
        return

    # 2. 训练模型
    success = train_model(
        use_lord=not args.no_lord,
        train_steps=args.steps,
        batch_size=args.batch_size
    )

    if success:
        print("\n✅ 训练成功完成!")
        print("\n下一步:")
        print("  1. 查看最佳公式: cat best_ashare_strategy.json")
        print("  2. 查看训练历史: cat training_history.json")
        print("  3. 使用最佳公式进行回测")
    else:
        print("\n❌ 训练失败，请查看错误信息")


if __name__ == '__main__':
    main()

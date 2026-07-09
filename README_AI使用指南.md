# AlphaGPT AI模型使用指南

## 一、项目AI架构概述

AlphaGPT是一个**基于AI的因子挖掘系统**，使用强化学习训练Transformer模型自动生成因子公式。

### 核心模块

```
model_core/
├── alphagpt.py         # AI模型架构 (Transformer + 强化学习)
├── engine.py           # 训练引擎 (强化学习训练循环)
├── factors.py          # 因子计算 (技术指标)
├── data_loader.py      # 数据加载 (PostgreSQL)
├── backtest.py        # 回测模块 (因子评估)
├── vm.py              # 公式解析器 (执行因子公式)
└── config.py          # 配置 (设备、数据库连接)
```

---

## 二、AI模型工作原理
****
### 1. 核心思想

**目标**: 使用AI自动发现有效的因子公式（无需人工设计）

**方法**: 
- 将因子公式看作"程序"
- 使用Transformer模型生成因子公式（类似代码生成）
- 使用强化学习优化模型（奖励 = 因子IC）

### 2. 模型架构 (`alphagpt.py`)

```
AlphaGPT模型:
├── Token Embedding: 将因子和操作符映射为向量
├── Positional Encoding: 位置编码
├── Looped Transformer: 增强的Transformer (带循环迭代)
│   ├── QK-Norm: Query-Key归一化 (稳定训练)
│   ├── RMSNorm: Root Mean Square归一化 (替代LayerNorm)
│   ├── SwiGLU: Swish-GLU激活 (改进FFN)
│   └── Multi-Loop: 层内循环迭代 (增强表达)
├── RMSNorm: 最终归一化
└── MTPHead: 多任务输出头
    ├── 因子公式生成 (离散token)
    └── 因子值预测 (连续值)
```

### 3. 训练流程 (`engine.py`)

```python
# 强化学习训练循环

For each training step:
    1. 生成因子公式 (Transformer + 采样)
       - 输入: 起始token [BOS]
       - 输出: 因子公式 (token序列, 最长12个token)
       
    2. 执行因子公式 (VM虚拟机器)
       - 输入: 因子公式 + 历史数据 (OHLCV)
       - 输出: 因子值序列 (n_stocks, hist_len)
       
    3. 评估因子 (Backtest)
       - 输入: 因子值序列
       - 输出: IC (信息系数) → 作为奖励
       
    4. 强化学习更新 (PPO/REINFORCE)
       - 目标: 最大化IC (奖励)
       - 优化: AdamW + Low-Rank Decay正则化
       
    5. 记录最佳公式
       - 如果IC超过历史最佳 → 保存公式
```

### 4. 因子公式表示

**Token词汇表** (`alphagpt.py:224-228`):

```python
self.features_list = ['RET', 'VOL', 'V_CHG', 'PV', 'TREND']  # 基础因子
self.ops_list = ['+', '-', '*', '/', 'abs', 'log', 'shift', ...]  # 操作符

self.vocab = self.features_list + self.ops_list  # 完整词汇表
```

**示例公式**:

```
公式1 (简单动量):
    RET shift(1) - RET
    → 昨日收益率减去前日收益率

公式2 (复杂因子):
    VOL log() * RET abs()
    → 成交量的对数乘以收益率的绝对值

公式3 (真实发现):
    [由AI自动生成，可能包含人类未发现的模式]
```

---

## 三、如何使用AI模型

### 方式1: 训练新模型 (从头开始)

#### Step 1: 准备数据

```bash
# 确保数据库中有A股数据
cd /Users/richirhuang/Desktop/AlphaGPT

# 检查数据
python3 -c "from data_pipeline.ashare_db import SessionLocal, Daily; \
from sqlalchemy import func; \
session = SessionLocal(); \
print('数据量:', session.query(func.count(Daily.ts_code)).scalar()); \
session.close()"
```

**预期输出**:
```
数据量: 10407
```

#### Step 2: 配置参数

编辑 `model_core/config.py`:

```python
class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = "postgresql://postgres:password@localhost:5432/ashare"
    
    BATCH_SIZE = 8192          # 批次大小 (根据GPU显存调整)
    TRAIN_STEPS = 1000        # 训练步数
    MAX_FORMULA_LEN = 12      # 最大公式长度 (token数)
    
    # 其他参数...
```

#### Step 3: 启动训练

```bash
cd /Users/richirhuang/Desktop/AlphaGPT

# 方式A: 使用Low-Rank Decay正则化 (推荐)
python3 -c "from model_core.engine import AlphaEngine; \
eng = AlphaEngine(use_lord_regularization=True); \
eng.train()"

# 方式B: 不使用正则化
python3 -c "from model_core.engine import AlphaEngine; \
eng = AlphaEngine(use_lord_regularization=False); \
eng.train()"
```

**训练输出示例**:

```
Starting A-share Alpha Mining with LoRD Regularization...
   LoRD Regularization enabled
   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']

[!] New King: Score 0.02 | Ret 1.23% | Formula [3, 7, 2, 5, 9]
[!] New King: Score 0.05 | Ret 2.45% | Formula [3, 8, 1, 4]
...

100%|████████████████| 1000/1000 [2:30:00] 

Training completed!
  Best score: 0.0823
  Best formula: [3, 8, 1, 4, 9, 2]
```

#### Step 4: 查看结果

```bash
# 最佳公式 (JSON格式)
cat best_ashare_strategy.json

# 训练历史 (JSON格式)
cat training_history.json
```

---

### 方式2: 使用预训练模型 (快速测试)

#### Step 1: 加载预训练模型

```python
from model_core.alphagpt import AlphaGPT
from model_core.config import ModelConfig

# 初始化模型
model = AlphaGPT().to(ModelConfig.DEVICE)

# 加载预训练权重 (如果有)
# model.load_state_dict(torch.load('pretrained_model.pth'))
```

#### Step 2: 生成因子公式

```python
import torch
from torch.distributions import Categorical

model.eval()

# 生成公式 (采样)
with torch.no_grad():
    inp = torch.zeros((1, 1), dtype=torch.long, device=ModelConfig.DEVICE)
    tokens_list = []
    
    for _ in range(ModelConfig.MAX_FORMULA_LEN):
        logits, _, _ = model(inp)
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        tokens_list.append(action.item())
        inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
    
    print(f"生成的公式: {tokens_list}")
```

#### Step 3: 解析公式

```python
from model_core.vm import StackVM

# 词汇表
vocab = ['RET', 'VOL', 'V_CHG', 'PV', 'TREND', '+', '-', '*', '/', 'abs', 'log', ...]

# 将token转换为可读公式
formula_tokens = [3, 8, 1, 4, 9, 2]
formula_str = ' '.join([vocab[t] for t in formula_tokens])

print(f"公式: {formula_str}")
# 输出: "公式: RET log() VOL + ..."
```

---

### 方式3: 评估生成的因子

#### Step 1: 执行因子公式

```python
from model_core.vm import StackVM
from model_core.data_loader import CryptoDataLoader

# 加载数据
loader = CryptoDataLoader()
loader.load_data()

# 执行公式
vm = StackVM()
formula = [3, 8, 1, 4, 9, 2]  # token序列
factor_values = vm.execute(formula, loader.feat_tensor)

print(f"因子值形状: {factor_values.shape}")
# 输出: 因子值形状: (804, 486)
```

#### Step 2: 评估因子

```python
from model_core.backtest import FactorBacktest

# 回测
bt = FactorBacktest()
score, ret_val = bt.evaluate(factor_values, loader.raw_data_cache, loader.target_ret)

print(f"IC (信息系数): {score:.4f}")
print(f"收益率: {ret_val:.2%}")

# 解读:
# - IC > 0.05: 强有效因子
# - IC > 0.02: 中等有效
# - IC < 0.01: 弱有效
```

---

## 四、高级功能

### 1. Low-Rank Decay正则化 (LoRD)

**目的**: 防止注意力参数过拟合，提升泛化能力

**原理**: 使用Newton-Schulz迭代法，将注意力权重矩阵"推"向低秩矩阵

**使用**:

```python
from model_core.alphagpt import NewtonSchulzLowRankDecay

# 在训练循环中使用
lord_opt = NewtonSchulzLowRankDecay(
    model.named_parameters(),
    decay_rate=1e-3,        # 衰减率 (越大正则化越强)
    num_iterations=5,        # Newton-Schulz迭代次数
    target_keywords=["q_proj", "k_proj", "attention"]
)

# 每个训练步后调用
lord_opt.step()
```

**监控**:

```python
from model_core.alphagpt import StableRankMonitor

# 监控参数秩 (stable rank)
rank_monitor = StableRankMonitor(model)

# 每个训练步后调用
stable_rank = rank_monitor.compute()
print(f"Stable Rank: {stable_rank:.2f}")

# 解读:
# - Stable Rank越低 → 正则化越强
# - 建议保持在原始秩的50-80%
```

### 2. 多任务学习 (MTPHead)

**目的**: 同时优化多个目标（IC、收益率、夏普比率）

**使用**:

```python
# 模型输出
logits, value, task_probs = model(idx)

# logits: 因子公式生成 (离散)
# value: 因子值预测 (连续)
# task_probs: 任务权重 (自适应)

# 可以设计多目标奖励
reward = 0.5 * ic_score + 0.3 * return_score + 0.2 * sharpe_score
```

### 3. 因子公式复杂度控制

**目的**: 避免生成过长/过复杂的公式（过拟合）

**方法**:

```python
# 在奖励函数中加入复杂度惩罚
def compute_reward(formula, factor_values, raw_data, target_ret):
    # 1. IC得分
    ic_score = bt.evaluate(factor_values, raw_data, target_ret)
    
    # 2. 复杂度惩罚
    complexity_penalty = len(formula) * 0.01  # 公式越长，惩罚越大
    
    # 3. 有效性惩罚
    if factor_values.std() < 1e-4:
        complexity_penalty += 2.0  # 因子无差异，惩罚
    
    return ic_score - complexity_penalty
```

---

## 五、训练技巧

### 1. 超参数调优

| 参数                  | 推荐值    | 说明                      |
| --------------------- | --------- | ------------------------- |
| `BATCH_SIZE`          | 8192      | 批次大小，根据GPU显存调整 |
| `TRAIN_STEPS`         | 1000-5000 | 训练步数，越多训练越充分  |
| `MAX_FORMULA_LEN`     | 8-12      | 公式最大长度，建议8-12    |
| `learning_rate`       | 1e-3      | AdamW学习率               |
| `lord_decay_rate`     | 1e-3      | LoRD正则化强度            |
| `lord_num_iterations` | 5         | Newton-Schulz迭代次数     |

### 2. 训练监控

```python
# 监控指标
training_history = {
    'step': [],              # 训练步数
    'avg_reward': [],        # 平均奖励 (IC)
    'best_score': [],        # 最佳IC
    'stable_rank': [],       # 参数秩 (如果启用LoRD)
}

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 平均奖励
axes[0,0].plot(training_history['step'], training_history['avg_reward'])
axes[0,0].set_title('Average Reward (IC)')

# 最佳分数
axes[0,1].plot(training_history['step'], training_history['best_score'])
axes[0,1].set_title('Best IC')

# 参数秩
if training_history['stable_rank']:
    axes[1,0].plot(training_history['step'][::100], training_history['stable_rank'])
    axes[1,0].set_title('Stable Rank (LoRD)')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
```

### 3. 常见问题

#### Q1: 训练不收敛 (IC始终很低)

**A**: 
1. 检查数据质量（是否有缺失值、异常值）
2. 增加训练步数（至少1000步）
3. 调整奖励函数（加入复杂度惩罚）
4. 使用更小的学习率（1e-4）

#### Q2: 生成的公式无意义 (VM执行失败)

**A**:
1. 检查VM实现（`vm.py`）是否支持所有操作符
2. 在奖励函数中加入"语法正确性"奖励
3. 使用课程学习（先训练简单公式，再训练复杂公式）

#### Q3: 过拟合 (训练IC高，样本外IC低)

**A**:
1. 启用LoRD正则化
2. 减小模型规模（减少层数、隐藏维度）
3. 增加数据量（使用更长时间序列）
4. 使用早停（监控验证集IC）

---

## 六、完整训练示例

### 脚本: `train_alphagpt.py`

```python
#!/usr/bin/env python3
"""
训练AlphaGPT模型 (完整示例)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.engine import AlphaEngine
import json

def main():
    print("="*60)
    print("AlphaGPT A股因子挖掘 - 训练脚本")
    print("="*60)
    
    # 1. 初始化训练引擎
    print("\n[1/4] 初始化训练引擎...")
    eng = AlphaEngine(
        use_lord_regularization=True,  # 启用LoRD正则化
        lord_decay_rate=1e-3,
        lord_num_iterations=5
    )
    
    # 2. 开始训练
    print("\n[2/4] 开始训练 (可能需要2-3小时)...")
    eng.train()
    
    # 3. 保存结果
    print("\n[3/4] 保存结果...")
    
    # 最佳公式
    if eng.best_formula:
        with open("best_ashare_strategy.json", "w") as f:
            json.dump({
                'formula': eng.best_formula,
                'score': eng.best_score
            }, f, indent=2)
        print(f"  ✅ 最佳公式已保存: best_ashare_strategy.json")
        print(f"     IC = {eng.best_score:.4f}")
    
    # 4. 评估最佳公式
    print("\n[4/4] 评估最佳公式...")
    # TODO: 添加样本外测试
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

if __name__ == '__main__':
    main()
```

### 运行

```bash
cd /Users/richirhuang/Desktop/AlphaGPT
python3 train_alphagpt.py
```

---

## 七、下一步工作

### 1. 模型改进

- [ ] 使用更强大的Transformer架构 (GPT-2/3)
- [ ] 加入交叉注意力 (参考其他股票因子)
- [ ] 使用遗传编程初始化 (加速训练)

### 2. 训练优化

- [ ] 分布式训练 (多GPU)
- [ ] 混合精度训练 (FP16)
- [ ] 课程学习 (Curriculum Learning)

### 3. 应用扩展

- [ ] 多市场适配 (美股、港股)
- [ ] 多时间框架 (分钟级、小时级)
- [ ] 实时因子计算 (在线学习)

---

## 八、文件速查

| 文件             | 功能       | 位置                        |
| ---------------- | ---------- | --------------------------- |
| `alphagpt.py`    | AI模型架构 | `model_core/alphagpt.py`    |
| `engine.py`      | 训练引擎   | `model_core/engine.py`      |
| `factors.py`     | 因子计算   | `model_core/factors.py`     |
| `data_loader.py` | 数据加载   | `model_core/data_loader.py` |
| `backtest.py`    | 回测模块   | `model_core/backtest.py`    |
| `vm.py`          | 公式解析器 | `model_core/vm.py`          |
| `config.py`      | 配置       | `model_core/config.py`      |

---

**祝训练顺利，发现Alpha！** 🚀📈

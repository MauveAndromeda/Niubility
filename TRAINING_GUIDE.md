# UltraTrader 训练和测试完全指南

## 📦 环境准备

### 1. 安装依赖
```bash
pip install -r requirements_ultra.txt
```

### 2. 验证安装
```bash
pytest test_ultra.py -v
```

---

## 🚀 快速开始（推荐）

### 一键运行完整流程
```bash
./quick_start.sh
```
这将依次执行：
1. 单元测试（验证环境）
2. 训练模型（20K步）
3. 单次评估（10个episode）
4. 多种子评估（10个种子）

**预计时间：** ~30-45分钟（CPU）/ ~10-15分钟（GPU）

---

## 🎯 分步训练

### 选项1：快速测试训练（10K步）
```bash
./train_fast.sh
```
- 步数：10,000
- 时间：~5分钟（CPU）/ ~2分钟（GPU）
- 用途：验证代码运行无误

### 选项2：标准训练（100K步）
```bash
./train_standard.sh
```
- 步数：100,000
- 时间：~30-60分钟（CPU）/ ~10-15分钟（GPU）
- 用途：生产环境模型

### 选项3：深度训练（500K步）
```bash
./train_deep.sh
```
- 步数：500,000
- 时间：~4-8小时（CPU）/ ~1-2小时（GPU）
- 用途：研究和发表

### 手动设置参数
```bash
export ULTRA_STEPS=50000    # 自定义步数
export ULTRA_SEED=789       # 自定义种子
python train_ultra.py
```

---

## 📊 评估模型

### 单次评估（10个episode）
```bash
python eval_ultra.py
```

**输出文件：**
- `reports/equity_curve_best.csv` - 权益曲线数据
- `reports/equity_curve_best.png` - 权益曲线图
- `reports/drawdown_curve_best.png` - 回撤曲线图
- `reports/performance_summary_single.csv` - 性能摘要

**关键指标：**
- Mean Reward: 平均奖励（目标 >10）
- Sharpe-like Ratio: 夏普比率（目标 >1.5）
- Max Drawdown: 最大回撤（目标 <15%）
- Final Equity: 最终权益（目标 >1.3）

### 多种子稳健性测试（50个episode）
```bash
python seedeval.py
```

**输出文件：**
- `reports/seed_stats.csv` - 各种子统计
- `reports/equity_multiseed.png` - 跨种子均值±标准差

**稳健性指标：**
- Reward Std: 标准差（越小越稳定，目标 <3.0）
- Sharpe: 夏普比率（目标 >1.5）
- Avg Max Drawdown: 平均最大回撤（目标 <20%）

---

## 🧪 测试

### 运行所有单元测试
```bash
pytest test_ultra.py -v
```

### 运行特定测试
```bash
pytest test_ultra.py::test_env_rollout_no_crash -v
pytest test_ultra.py::test_env_spaces -v
pytest test_ultra.py::test_expert_ensemble_output -v
```

### 带覆盖率报告
```bash
pytest --cov=. test_ultra.py
```

---

## 📁 输出目录结构

```
UltraTrader/
├── checkpoints/
│   ├── best/
│   │   └── best_model.zip              # 最佳模型
│   └── periodic/
│       ├── ultra_sac_5000_steps.zip    # 5K步检查点
│       ├── ultra_sac_10000_steps.zip   # 10K步检查点
│       ├── ultra_sac_15000_steps.zip   # 15K步检查点
│       └── ultra_sac_20000_steps.zip   # 20K步检查点
│
├── logs/
│   └── [训练日志]
│
└── reports/
    ├── equity_curve_best.csv           # 最佳权益曲线数据
    ├── equity_curve_best.png           # 权益曲线图
    ├── drawdown_curve_best.png         # 回撤曲线图
    ├── performance_summary_single.csv   # 单次评估摘要
    ├── seed_stats.csv                  # 多种子统计
    └── equity_multiseed.png            # 多种子权益曲线
```

---

## 🎓 解读结果

### 优秀模型的特征

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| **Mean Reward** | >15 | 10-15 | <10 |
| **Sharpe Ratio** | >2.0 | 1.0-2.0 | <1.0 |
| **Max Drawdown** | <10% | 10-20% | >20% |
| **Final Equity** | >1.5 | 1.2-1.5 | <1.2 |
| **Episode Length** | >400 | 300-400 | <300 |

### 权益曲线分析

**理想曲线：**
- 平滑上升趋势
- 回撤幅度小且恢复快
- 跨种子标准差带窄

**需改进的曲线：**
- 平坦或下降趋势
- 剧烈震荡
- 长时间回撤不恢复

---

## 🔧 故障排查

### 问题1：训练时reward全是NaN
**原因：** 数值不稳定
**解决：** 检查 `ultra_trading_env.py` 的奖励计算，确保没有除零

### 问题2：找不到模型文件
**原因：** 训练未完成或评估回调未触发
**解决：**
```bash
ls -la checkpoints/best/
ls -la checkpoints/periodic/
```
确保至少有periodic检查点

### 问题3：GPU未被识别
**检查：**
```python
import torch
print(torch.cuda.is_available())  # 应返回True
```
**解决：** 安装CUDA版本的PyTorch

### 问题4：测试失败
**解决：**
```bash
pytest test_ultra.py -v  # 查看详细错误信息
```
根据错误信息修复代码

---

## 📈 训练监控

### 训练过程中的关键指标

训练时终端会显示：
```
| rollout/ep_rew_mean       | 12.3
| rollout/ep_len_mean       | 245
| train/actor_loss          | -3.21
| train/critic_loss         | 0.45
| train/ent_coef            | 0.18
| time/total_timesteps      | 5000
```

**好的训练信号 ✓**
- `ep_rew_mean` 逐步上升（>10为盈利）
- `ep_len_mean` 保持在400+（长期存活）
- `ent_coef` 逐渐下降（从探索到利用）

**坏的训练信号 ✗**
- `ep_len_mean` <100（频繁爆仓）
- `ep_rew_mean` 剧烈震荡不收敛
- 所有奖励都是负数

---

## 🚀 高级用法

### 自定义配置文件
编辑 `config/training_config.yaml`（如存在）并加载：
```python
from config_manager import ConfigManager
config = ConfigManager.load('config/training_config.yaml')
```

### 使用不同的SAC参数
修改 `train_ultra.py` 中的：
```python
model = SAC(
    policy="MlpPolicy",
    learning_rate=5e-4,    # 更高的学习率
    buffer_size=300000,    # 更大的replay buffer
    batch_size=512,        # 更大的batch size
    ...
)
```

### 并行训练多个种子
```bash
for seed in 123 456 789; do
    export ULTRA_SEED=$seed
    export ULTRA_STEPS=100000
    python train_ultra.py &
done
wait
```

---

## 💡 最佳实践

1. **先运行快速测试** (`./train_fast.sh`) 验证代码无误
2. **查看测试结果** 确保所有单元测试通过
3. **运行标准训练** (`./train_standard.sh`) 获取生产模型
4. **评估稳健性** (`python seedeval.py`) 验证跨种子表现
5. **分析权益曲线** 检查平滑度和回撤恢复
6. **调整超参数** 根据结果优化训练配置
7. **深度训练** 如果需要发表级别的模型

---

## 📚 相关文档

- `README.md` - 项目总体介绍
- `CLAUDE.md` - 开发者指南
- `test_ultra.py` - 测试用例参考

---

## 🆘 获取帮助

如遇到问题：
1. 检查 `pytest test_ultra.py -v` 是否全部通过
2. 查看 `logs/` 目录的训练日志
3. 检查 `reports/` 目录的评估结果
4. 参考本文档的故障排查章节

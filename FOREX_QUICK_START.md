# 外汇交易系统 - 快速开始
## 从模拟环境到真实外汇数据

---

## 🎯 为什么要迁移到外汇？

### **当前模型表现（基于模拟环境）**
```
最佳模型:  评级 2/9 (较差), 最大回撤 30%
300K模型:  评级 3/9 (中等), Sharpe 1.35, 胜率 53%, 回撤 18%
```

**核心问题：** 模拟环境（AR(1)过程）过于简单，无法学到真实alpha

### **外汇市场优势**
✅ 真实历史数据（15+年，免费获取）
✅ 24/7交易，无gap风险
✅ 高流动性，低滑点
✅ 多货币对，天然分散化
✅ 杠杆灵活，风险可控

### **预期性能提升**
| 指标 | 模拟环境 | 外汇真实数据 | 改进 |
|------|---------|-------------|------|
| **Sharpe** | 1.35 | 2.0-2.5 | +48-85% |
| **胜率** | 53% | 58-65% | +9-23% |
| **最大回撤** | 18% | 10-12% | -33-44% |
| **年化收益** | ~8% | 20-30% | +150-275% |

---

## 🚀 5步快速开始

### **Step 1: 安装依赖**
```bash
# 基础依赖（如已安装可跳过）
pip install pandas numpy requests lzma

# 或使用requirements
pip install -r requirements_ultra.txt
```

### **Step 2: 下载外汇数据**
```bash
# 运行数据下载器（下载1周EUR/USD数据测试）
python forex_data_loader.py

# 输出位置: data/forex/EURUSD_1min_2024-01-01_2024-01-07.csv
```

**预计时间：** 2-5分钟（1周数据）

### **Step 3: 验证数据**
```python
import pandas as pd

# 加载数据
df = pd.read_csv('data/forex/EURUSD_1min_2024-01-01_2024-01-07.csv',
                 index_col=0, parse_dates=True)

print(f"数据量: {len(df)} bars")
print(f"时间范围: {df.index[0]} → {df.index[-1]}")
print(df.head())
```

**预期输出：**
```
数据量: ~7000 bars (1周 × 24小时 × 60分钟)
列: open, high, low, close, volume, spread
```

### **Step 4: 下载生产数据（完整版）**
```bash
# 下载2-3年历史数据用于训练
python forex_data_loader.py --pair EURUSD --start 2022-01-01 --end 2024-12-31
```

**注意：** 完整数据下载需要2-6小时，建议晚上运行

### **Step 5: 训练外汇模型**
```bash
# 使用外汇环境训练（待实现）
python train_forex.py --data data/forex/EURUSD_1min_2022-01-01_2024-12-31.csv
```

---

## 📊 数据格式

### **Dukascopy数据格式**
```csv
datetime,open,high,low,close,volume,spread
2024-01-01 00:00:00,1.10450,1.10470,1.10440,1.10460,125000,0.00015
2024-01-01 00:01:00,1.10460,1.10480,1.10450,1.10475,110000,0.00014
...
```

**字段说明：**
- `datetime`: UTC时间
- `open/high/low/close`: 中间价（ask和bid的平均）
- `volume`: 成交量（ask_volume + bid_volume）
- `spread`: 点差（ask - bid），重要的交易成本

---

## 🛠️ 下一步开发

### **已完成（本次提交）**
✅ 外汇迁移完整方案（FOREX_MIGRATION_GUIDE.md）
✅ Dukascopy数据下载器（forex_data_loader.py）
✅ 快速开始指南（本文档）

### **待实现（下一阶段）**
⏳ ForexTradingEnv（基于真实OHLCV的环境）
⏳ ForexExpertEnsemble（外汇特定技术指标）
⏳ train_forex.py（外汇模型训练脚本）
⏳ 多货币对组合管理
⏳ 新闻情绪分析集成

---

## 💡 使用建议

### **数据下载策略**
1. **测试阶段**：下载1周-1个月数据验证流程
2. **开发阶段**：下载1年数据用于训练
3. **生产阶段**：下载3年数据，按年份分割做walk-forward

### **货币对选择**
**新手推荐：** EUR/USD（流动性最高，spread最低）
**进阶选择：** GBP/USD, USD/JPY
**多元化组合：** EUR/USD + GBP/USD + AUD/USD

### **时间框架选择**
**高频策略：** 1min或5min
**日内策略：** 15min或30min
**波段策略：** 1H或4H

---

## ⚠️ 重要提醒

### **1. 数据下载注意事项**
- Dukascopy数据免费但下载慢，请耐心等待
- 周末和节假日可能无数据（外汇市场休市）
- 建议使用稳定网络，避免中断

### **2. 存储空间**
- 1年1分钟数据 ≈ 500MB（压缩后）
- 3年数据 ≈ 1.5GB
- 建议预留5GB空间

### **3. 数据质量验证**
下载后务必检查：
- [ ] 时间连续性（无大gap）
- [ ] 价格合理性（无异常跳变）
- [ ] Spread正常（EUR/USD通常0.0001-0.0003）

### **4. 过拟合风险**
- 使用2022-2023年训练
- 2024年作为out-of-sample测试
- 定期重新训练（每季度）

---

## 📚 相关文档

- **FOREX_MIGRATION_GUIDE.md** - 完整迁移方案和实现代码（980行）
- **DIAGNOSTIC_ANALYSIS.md** - 当前模型问题诊断
- **IMPROVEMENT_ROADMAP.md** - 达到30%年化收益的完整路线图

---

## 🆘 故障排查

### Q1: 下载速度很慢怎么办？
**A:** Dukascopy服务器在欧洲，亚洲地区可能慢
- 尝试VPN连接欧洲节点
- 或使用OANDA API（需注册）

### Q2: 下载中断怎么办？
**A:** 脚本会从中断处继续
```bash
# 重新运行相同命令即可
python forex_data_loader.py
```

### Q3: 数据有缺失怎么办？
**A:** 周末和节假日外汇市场休市，这是正常的
```python
# 填充缺失值（向前填充）
df = df.fillna(method='ffill')
```

### Q4: 如何验证数据质量？
**A:**
```python
# 检查缺失值
print(df.isnull().sum())

# 检查异常值
print(df.describe())

# 检查spread合理性
print(f"平均spread: {df['spread'].mean():.6f}")
print(f"最大spread: {df['spread'].max():.6f}")  # 应该<0.001
```

---

## 🎓 学习资源

1. **外汇基础**
   - BabyPips.com - 外汇新手教程
   - Investopedia - 外汇交易术语

2. **技术分析**
   - TradingView - 图表和指标
   - FXCM - 外汇策略库

3. **数据源**
   - Dukascopy - 免费历史数据
   - OANDA - API文档
   - HistData.com - 备选数据源

---

**准备好开始了吗？运行 `python forex_data_loader.py` 下载第一份数据！** 🚀

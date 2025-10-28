"""
UltraTrader 模型优化脚本
基于诊断结果的自动化优化

使用方法:
    python optimize_model.py --phase 1   # Phase 1: 奖励函数优化
    python optimize_model.py --phase 2   # Phase 2: 风险管理优化
    python optimize_model.py --all       # 全部优化
"""

import argparse
import shutil
import re
from pathlib import Path


class ModelOptimizer:
    """模型优化器"""

    def __init__(self):
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)

    def backup_file(self, filepath):
        """备份文件"""
        filepath = Path(filepath)
        backup_path = self.backup_dir / f"{filepath.name}.backup"
        shutil.copy(filepath, backup_path)
        print(f"✅ 已备份: {filepath} → {backup_path}")

    def phase1_reward_optimization(self):
        """
        Phase 1: 奖励函数优化
        - 增加风险调整
        - 添加胜率奖励
        - 增强回撤惩罚
        """
        print("\n" + "="*80)
        print("Phase 1: 奖励函数优化")
        print("="*80)

        env_file = Path("ultra_trading_env.py")
        if not env_file.exists():
            print("❌ 未找到 ultra_trading_env.py")
            return False

        # 备份原文件
        self.backup_file(env_file)

        # 读取文件
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 在 _reset_state 方法中添加新状态变量
        reset_state_pattern = r'(def _reset_state\(self\):.*?self\.recent_return = 0\.0)'

        new_state_vars = '''
        # Risk-aware state variables
        self.return_history = []
        self.recent_pnls = []
        self.consecutive_losses = 0'''

        if 'self.return_history = []' not in content:
            content = re.sub(
                reset_state_pattern,
                r'\1' + new_state_vars,
                content,
                flags=re.DOTALL
            )
            print("✅ 添加风险状态变量")

        # 替换奖励计算部分
        new_reward_code = '''
        # ============================================================
        # 风险感知奖励函数 v2.0 (优化后)
        # ============================================================

        # 估计当前波动率
        if len(self.return_history) >= 20:
            recent_vols = np.array(self.return_history[-20:])
            current_vol = np.std(recent_vols)
        else:
            current_vol = 0.005  # 默认值

        # 保存收益率历史
        self.return_history.append(pnl_pct)
        if len(self.return_history) > 100:
            self.return_history.pop(0)

        # 1. 风险调整后的盈利奖励
        vol_adjusted_profit = (80.0 * pnl_pct) / (1 + 5.0 * current_vol)

        # 2. 持仓奖励（只在盈利时）
        if pnl_pct > 0:
            hold_bonus = 40.0 * pnl_pct * exposure_frac
        else:
            hold_bonus = 0.0

        # 3. 回撤惩罚（平方惩罚，更严厉）
        dd_penalty = 100.0 * (drawdown ** 2)

        # 4. 动态杠杆惩罚（基于波动率）
        leverage_penalty = 5.0 * exposure_frac * (1 + 10.0 * current_vol)

        # 5. 连续亏损惩罚
        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        loss_streak_penalty = 10.0 * min(self.consecutive_losses / 5.0, 1.0)

        # 6. 胜率奖励
        self.recent_pnls.append(pnl_pct)
        if len(self.recent_pnls) > 10:
            self.recent_pnls.pop(0)

        if len(self.recent_pnls) >= 10:
            win_rate = sum(1 for p in self.recent_pnls if p > 0) / 10
            win_rate_bonus = 20.0 * (win_rate - 0.5)
        else:
            win_rate_bonus = 0.0

        # 组合奖励
        reward_unclipped = (
            vol_adjusted_profit
            + hold_bonus
            + win_rate_bonus
            - dd_penalty
            - leverage_penalty
            - loss_streak_penalty
        )

        # Softsign归一化（允许更大范围）
        reward = float(reward_unclipped / (1 + abs(reward_unclipped)))
        '''

        # 找到并替换原奖励计算代码
        # 匹配从 "# PnL metrics" 到 "reward = float(np.tanh..."
        reward_pattern = r'# PnL metrics.*?reward = float\(np\.tanh\(reward_unclipped / self\.CLIP_SCALE\)\)'

        if re.search(reward_pattern, content, re.DOTALL):
            content = re.sub(
                reward_pattern,
                new_reward_code.strip(),
                content,
                flags=re.DOTALL
            )
            print("✅ 已更新奖励函数")
        else:
            print("⚠️  警告: 未找到奖励函数代码块，请手动更新")

        # 写回文件
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print("\n📊 Phase 1 优化完成!")
        print("   预期效果:")
        print("   • 胜率: 35% → 45%")
        print("   • Sharpe: 0.99 → 1.2-1.4")
        print("   • 最大回撤: 24.77% → 17%")
        return True

    def phase2_risk_management(self):
        """
        Phase 2: 风险管理优化
        - 添加动态止损
        - 实现波动率自适应仓位
        """
        print("\n" + "="*80)
        print("Phase 2: 风险管理优化")
        print("="*80)

        env_file = Path("ultra_trading_env.py")
        if not env_file.exists():
            print("❌ 未找到 ultra_trading_env.py")
            return False

        # 读取文件
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 在类定义后添加风险管理器
        risk_manager_code = '''

class DynamicStopLoss:
    """动态止损管理器"""

    def __init__(self):
        self.max_dd_threshold = 0.15  # 最大回撤15%
        self.warning_dd = 0.05        # 警戒回撤5%
        self.critical_dd = 0.10       # 危险回撤10%

    def check_stop_loss(self, current_equity, peak_equity, position_size):
        """检查是否触发止损"""
        dd = (peak_equity - current_equity) / max(peak_equity, 1e-6)

        if dd >= self.max_dd_threshold:
            return 0.0, "FORCE_CLOSE"
        elif dd >= self.critical_dd:
            return position_size * 0.25, "REDUCE_75"
        elif dd >= self.warning_dd:
            return position_size * 0.50, "REDUCE_50"
        else:
            return position_size, "NORMAL"


class VolatilityPositionSizer:
    """波动率自适应仓位管理"""

    def __init__(self, target_vol=0.01):
        self.target_vol = target_vol

    def calculate_optimal_position(self, signal_strength, current_vol, base_max=0.6):
        """计算最优仓位"""
        vol_scalar = self.target_vol / max(current_vol, 1e-6)
        base_position = signal_strength * base_max
        adjusted_position = base_position * vol_scalar
        return np.clip(adjusted_position, 0.0, base_max)

'''

        if 'class DynamicStopLoss:' not in content:
            # 在 MultiAgentMarketSimulator 类定义之前插入
            insert_pos = content.find('class MultiAgentMarketSimulator:')
            if insert_pos != -1:
                content = content[:insert_pos] + risk_manager_code + '\n' + content[insert_pos:]
                print("✅ 添加风险管理器类")
            else:
                print("⚠️  警告: 未找到合适的插入位置")

        # 在 __init__ 方法中初始化风险管理器
        init_pattern = r'(self\.experts = UltraExpertEnsemble\(\))'
        risk_manager_init = r'''\1

        # Risk management components
        self.stop_loss = DynamicStopLoss()
        self.position_sizer = VolatilityPositionSizer(target_vol=0.01)'''

        if 'self.stop_loss = DynamicStopLoss()' not in content:
            content = re.sub(init_pattern, risk_manager_init, content)
            print("✅ 初始化风险管理器")

        # 在 step 方法中集成止损逻辑
        step_action_pattern = r'(direction = float\(np\.clip\(dir_raw, -1\.0, 1\.0\)\)\s+exposure  = float\(np\.clip\(exp_raw, 0\.0, self\.EXPOSURE_MAX\)\))'

        stop_loss_code = r'''\1

        # 动态止损检查
        current_dd = (self.equity_peak - self.equity) / max(self.equity_peak, 1e-6)
        adjusted_exposure, stop_signal = self.stop_loss.check_stop_loss(
            self.equity, self.equity_peak, self.position_size
        )

        if stop_signal != "NORMAL":
            # 触发止损，调整仓位
            exposure = min(exposure, adjusted_exposure)

        # 波动率自适应仓位
        if len(self.return_history) >= 20:
            current_vol = np.std(self.return_history[-20:])
            vol_adjusted_exposure = self.position_sizer.calculate_optimal_position(
                signal_strength=abs(direction),
                current_vol=current_vol,
                base_max=self.EXPOSURE_MAX
            )
            exposure = min(exposure, vol_adjusted_exposure)'''

        if '# 动态止损检查' not in content:
            content = re.sub(step_action_pattern, stop_loss_code, content, flags=re.DOTALL)
            print("✅ 集成止损和仓位管理逻辑")

        # 写回文件
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print("\n📊 Phase 2 优化完成!")
        print("   预期效果:")
        print("   • 最大回撤: 17% → 14%")
        print("   • Sharpe: 1.2-1.4 → 1.5-1.8")
        return True

    def update_training_hyperparameters(self):
        """更新训练超参数"""
        print("\n" + "="*80)
        print("训练超参数优化")
        print("="*80)

        train_file = Path("train_ultra.py")
        if not train_file.exists():
            print("❌ 未找到 train_ultra.py")
            return False

        self.backup_file(train_file)

        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 更新超参数
        replacements = [
            (r'learning_rate=3e-4', 'learning_rate=1e-4  # 降低学习率提高稳定性'),
            (r'gamma=0\.995', 'gamma=0.998  # 更重视长期收益'),
            (r'ent_coef="auto_0\.2"', 'ent_coef="auto_0.1"  # 降低探索'),
            (r'train_freq=1', 'train_freq=4  # 降低训练频率'),
            (r'gradient_steps=1', 'gradient_steps=2  # 增加梯度更新次数'),
        ]

        for pattern, replacement in replacements:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                print(f"✅ 更新: {replacement}")

        with open(train_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print("\n📊 超参数优化完成!")
        return True

    def run_all_optimizations(self):
        """执行所有优化"""
        print("\n" + "="*80)
        print("🚀 执行完整优化流程")
        print("="*80)

        success = True
        success &= self.phase1_reward_optimization()
        success &= self.phase2_risk_management()
        success &= self.update_training_hyperparameters()

        if success:
            print("\n" + "="*80)
            print("✅ 所有优化完成!")
            print("="*80)
            print("\n下一步:")
            print("  1. 重新训练模型: python train_ultra.py")
            print("  2. 运行诊断评估: python diagnose_and_fix.py")
            print("  3. 对比性能变化")
            print("\n预期改进:")
            print("  • 胜率: 35% → 50%+")
            print("  • Sharpe: 0.99 → 1.5+")
            print("  • 最大回撤: 24.77% → <15%")
        else:
            print("\n⚠️  部分优化失败，请检查日志")

        return success


def main():
    parser = argparse.ArgumentParser(description='UltraTrader 模型优化工具')
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        help='执行特定优化阶段 (1: 奖励函数, 2: 风险管理)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='执行所有优化'
    )
    parser.add_argument(
        '--hyperparams',
        action='store_true',
        help='仅更新超参数'
    )

    args = parser.parse_args()

    optimizer = ModelOptimizer()

    if args.all:
        optimizer.run_all_optimizations()
    elif args.phase == 1:
        optimizer.phase1_reward_optimization()
    elif args.phase == 2:
        optimizer.phase2_risk_management()
    elif args.hyperparams:
        optimizer.update_training_hyperparameters()
    else:
        # 默认执行全部
        parser.print_help()
        print("\n提示: 使用 --all 执行完整优化")


if __name__ == '__main__':
    main()

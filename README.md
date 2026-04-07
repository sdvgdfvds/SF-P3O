# SF-P3O: Spectral-Fisher Dual-Pathway Plasticity-Preserving Policy Optimization

SF-P3O: 基于谱-Fisher双通道的可塑性保持策略优化

## 简介

本仓库包含论文 **"SF-P3O: 基于谱-Fisher双通道的可塑性保持策略优化"** 的源代码与实验数据。

SF-P3O 针对深度强化学习中的可塑性丧失问题，提出三项核心创新：
1. **谱可塑性诊断** — 基于权重矩阵有效秩的自适应触发机制
2. **Fisher引导的谱扰动** — 在KL散度约束下定向恢复退化的奇异值
3. **双通道架构** — 具有状态依赖门控的稳定-可塑双通道

## 仓库结构

```
SF-P3O/
├── redraw_figures.py           # 实验图表绘制脚本（从原始CSV数据）
├── figures/                    # 论文图表（17张PNG）
├── data/
│   ├── sf_p3o_runs/            # 核心实验数据（1M步，5环境×8算法）
│   ├── sf_p3o_highdim/         # 高维环境数据（Ant, Humanoid）
│   ├── sf_p3o_longrun/         # 长时域实验数据（3M步）
│   └── sf_p3o_probe/           # 奖励反转实验数据
└── README.md
```

## 环境依赖

```bash
pip install matplotlib numpy
```

## 使用方法

### 从原始数据绘制图表
```bash
python redraw_figures.py
```

## 实验环境

- MuJoCo物理引擎（通过Gymnasium接口）
- 5个连续控制环境：HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4, Humanoid-v4
- 每组实验5个随机种子

## 主要结果

| 环境 | PPO | SF-P3O | 提升 |
|------|-----|--------|------|
| HalfCheetah | 1419±183 | 1737±323 | +22% |
| Hopper | 561±90 | 603±49 | +7% |
| Walker2d | 513±92 | 577±146 | +12% |
| Humanoid | 354±19 | 365±7 | +3% |

长时域训练（3M步）优势扩大至 **+47%~49%**。

## 引用

如果本工作对您有帮助，请引用：
```
@article{sfp3o2026,
  title={SF-P3O: 基于谱-Fisher双通道的可塑性保持策略优化},
  year={2026}
}
```

## License

MIT License

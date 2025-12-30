# MP-KVM: Manifold-Partitioned Key-Value Memory

## 🚀 快速开始

### 一键运行所有实验
```bash
# 使用真实数据运行完整实验 (推荐)
python run_complete_experiment.py all --use-real-model

# 使用合成数据运行完整实验（快速测试用）
python run_complete_experiment.py all
```

### 分阶段运行实验
```bash
# 运行单个阶段 (使用真实数据)
python run_complete_experiment.py phase --phase 1 --use-real-model  # 基准线比较
python run_complete_experiment.py phase --phase 2 --use-real-model  # 流形可视化
python run_complete_experiment.py phase --phase 3 --use-real-model  # 压缩实验
python run_complete_experiment.py phase --phase 4 --use-real-model  # 消融研究
python run_complete_experiment.py phase --phase 5 --use-real-model  # 注意力分析
```

### 生成图表
```bash
# 使用已有的实验数据生成论文图表
python run_complete_experiment.py figures
```

## 📁 项目结构

```
MP-KVM/
├── run_complete_experiment.py     # 🚀 主入口：实验运行
├── generate_paper_figures.py      # 📊 图表生成
├── run_real_model_experiment.py   # 🔧 真实模型数据提取
├── run_phase_by_phase.py         # 🔬 分阶段运行（可选）
├── core/                         # 🧠 核心算法
├── adapters/                     # 🔌 模型适配器
├── experiments/                  # 🧪 具体实验实现
├── analysis/                     # 📈 数据分析
└── results/                      # 💾 实验结果输出
```

## 🎯 实验阶段

1. **Phase 1**: 基准线比较 - 与现有方法比较性能
2. **Phase 2**: 流形可视化 - 展示语义聚类效果
3. **Phase 3**: 压缩实验 - 测试不同压缩比的性能
4. **Phase 4**: 消融研究 - 验证各组件贡献
5. **Phase 5**: 注意力分析 - 分析注意力权重分布

## 📊 输出结果

运行完成后，结果保存在 `results/` 目录：

```
results/
├── baseline/           # Phase 1 结果
├── figures/           # 📈 论文图表 (6个PNG文件)
├── compression_sweep/ # Phase 3 结果
├── ablation/          # Phase 4 结果
├── attention_analysis/ # Phase 5 结果
└── experiment_summary.json  # 实验总结
```

## 🔧 核心特性

- **真实模型数据**: 支持使用真实的Llama-3.1-8B模型数据
- **Pre-RoPE聚类**: 解决位置编码对相似度计算的干扰
- **渐进式压缩**: 实时流式处理，支持任意长度序列
- **GPU加速**: 高效的GPU聚合器
- **学术级输出**: 自动生成论文发表质量的图表

## 📈 性能亮点

- **压缩比**: 支持10:1到100:1的压缩比
- **召回率**: 在4%压缩比下保持97.5%的needle召回率
- **实时性**: GPU聚合保持推理性能
- **鲁棒性**: 处理各种序列长度和语义内容

## 🎨 生成的论文图表

1. **Figure 1**: 语义流形分布 - UMAP/t-SNE投影
2. **Figure 2**: Needle热力图 - 性能vs序列长度和深度
3. **Figure 3**: 压缩性能曲线 - PPL和召回率vs压缩比
4. **Figure 4**: 消融研究对比 - 组件贡献分析
5. **Figure 5**: 注意力能量谱 - score_bias效果
6. **Figure 6**: 效率分析 - CPU/GPU性能对比

## 🔧 故障排除

### PyTorch环境问题
如果遇到PyTorch加载错误，请确保：
1. 使用正确的conda环境：`conda activate py`
2. CUDA版本匹配：检查`nvcc --version`

### 内存不足
对于大模型实验，建议：
- 使用GPU环境：`--use-real-model`需要GPU
- 减少序列长度：在实验脚本中调整参数

### 实验失败
- 每个阶段都可以独立运行用于调试
- 检查`results/`目录中的错误日志
- 使用合成数据进行快速测试：移除`--use-real-model`参数

## 📝 技术细节

MP-KVM通过以下创新解决了Transformer KV缓存压缩的关键问题：

1. **RoPE位置干扰**: 提取Pre-RoPE向量进行聚类
2. **实时流处理**: 增量更新避免历史数据重复累积
3. **注意力补偿**: log-count权重防止centroid饥饿
4. **GPU优化**: 异步聚合保持推理性能

更多技术细节请参考代码注释和实验结果。
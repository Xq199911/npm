# MP-KVM: Manifold-Partitioned Key-Value Memory

## ⚠️ 重要注意事项

**论文复现必须使用真实数据**：所有实验命令都应包含 `--use-real-model` 参数，以确保使用真实的Llama模型数据进行测试。合成数据仅用于快速环境验证。

##  快速开始

### 环境准备
```bash
# 1. 激活conda环境
conda activate py

# 2. 验证环境
python demo.py

# 3. 确保模型文件存在 (用于真实数据实验)
ls model/Llama-3.1-8B-Instruct/
```

### 完整实验流程 (推荐使用真实数据)
```bash
# Step 1: 运行主要实验套件 (使用真实Llama模型数据)
python run_complete_experiment.py --use-real-model

# Step 2: 运行组件贡献分析实验 (必需 - 生成组件消融数据)
python experiments/run_ablation_study.py

# Step 3: 可选 - 运行扩展Needle-in-a-Haystack实验
python experiments/run_niah.py

# Step 4: 生成论文图表
python generate_paper_figures.py
```

### 快速测试模式 (仅用于验证环境)
```bash
# 仅用于快速验证代码是否正常运行，不产生论文数据
python run_complete_experiment.py --quick
```

### 分阶段运行实验 (全部使用真实数据)
```bash
# Phase 1: 基准线比较 (Baseline Comparison)
python run_complete_experiment.py --phase 1 --use-real-model

# Phase 2: 流形可视化 (Manifold Visualization)
python run_complete_experiment.py --phase 2 --use-real-model

# Phase 3: 压缩实验 (Compression Experiments)
python run_complete_experiment.py --phase 3 --use-real-model

# Phase 4: 一般消融研究 (General Ablation Studies)
python run_complete_experiment.py --phase 4 --use-real-model

# Phase 5: 性能分析 (Performance Profiling)
python run_complete_experiment.py --phase 5 --use-real-model

# 额外必需: 组件贡献分析 (Component Contribution Analysis)
python experiments/run_ablation_study.py

# 可选: 扩展Needle-in-a-Haystack实验
python experiments/run_niah.py
```

### 生成论文图表
```bash
# 使用已有的实验数据生成论文图表
python generate_paper_figures.py
```

### 高级配置选项
```bash
# 跳过GPU性能分析 (节省时间)
python run_complete_experiment.py --use-real-model --skip-performance

# 使用自定义模型路径
python run_complete_experiment.py --use-real-model --model-path "path/to/your/model"

# 显示所有可用选项
python run_complete_experiment.py --help

# 仅验证环境 (不产生有效实验数据)
python run_complete_experiment.py --quick
```

##  项目结构

```
MP-KVM/
├── run_complete_experiment.py     #  主入口：实验运行
├── generate_paper_figures.py      #  图表生成
├── run_real_model_experiment.py   #  真实模型数据提取
├── run_phase_by_phase.py         #  分阶段运行（可选）
├── core/                         #  核心算法
├── adapters/                     #  模型适配器
├── experiments/                  #  具体实验实现
├── analysis/                     #  数据分析
└── results/                      #  实验结果输出
```

##  实验阶段详解

1. **Phase 1**: 基准线比较
   - 与H2O、StreamingLLM、Random等方法比较
   - 评估在相同内存约束下的性能差异
   - 输出：`results/baseline/enhanced_baseline_results.json`

2. **Phase 2**: 流形可视化
   - 展示MP-KVM的语义聚类效果
   - 生成UMAP/t-SNE投影图
   - 输出：`results/figures/manifold_clustering.png`

3. **Phase 3**: 压缩实验
   - 测试不同压缩比(10:1到100:1)的性能
   - 评估PPL和召回率vs压缩比曲线
   - 输出：`results/compression_sweep/`

4. **Phase 4**: 一般消融研究
   - 测试相似度阈值和质心数量参数
   - 对比不同配置的性能差异
   - 输出：`results/ablation/ablation_results.json`

5. **Phase 5**: 注意力分析
   - 分析注意力权重分布和能量补偿效果
   - 生成注意力能量谱图表
   - 输出：`results/attention_analysis/`

6. **组件贡献分析** (必需)
   - 详细测试四个MP-KVM组件的贡献度
   - Standard Clustering / w/o Positionless / w/o Energy Compensation / Full MP-KVM
   - 输出：`results/ablation/ablation_study_results.json`



### 学术研究者 (论文复现)
```bash
# 完整实验流程 - 复现论文所有结果
python run_complete_experiment.py --use-real-model
python experiments/run_ablation_study.py
python experiments/run_niah.py
python generate_paper_figures.py

# 只验证核心压缩实验结果
python run_complete_experiment.py --phase 3 --use-real-model

# 生成论文图表 (需先完成以上实验)
python generate_paper_figures.py
```

### 开发者/工程师
```bash
# 快速验证代码运行 (使用合成数据)
python run_complete_experiment.py --quick

# 性能基准测试 (使用真实数据)
python run_complete_experiment.py --phase 5 --use-real-model

# 自定义模型测试
python run_complete_experiment.py --phase 1 --use-real-model --model-path "your/model"
```

### 新手用户
```bash
# 环境验证
python demo.py

# 快速上手验证 (合成数据)
python run_complete_experiment.py --quick

# 完整实验运行 (真实数据 - 推荐)
python run_complete_experiment.py --use-real-model
python experiments/run_ablation_study.py
python generate_paper_figures.py

# 分阶段学习 (真实数据)
python run_complete_experiment.py --phase 1 --use-real-model  # 基准线比较
python run_complete_experiment.py --phase 2 --use-real-model  # 流形可视化
```

##  输出结果

运行完成后，所有结果保存在 `results/` 目录：

```
results/
├── baseline/           # Phase 1: 基准线比较结果
│   └── enhanced_baseline_results.json
├── figures/           #  论文图表 (6个PNG文件)
│   ├── manifold_clustering.png
│   ├── topic_transitions.png
│   └── [其他图表文件...]
├── compression_sweep/ # Phase 3: 压缩实验结果
│   ├── h2o_results.json
│   ├── no_compression_results.json
│   ├── random_eviction_results.json
│   └── streamingllm_results.json
├── ablation/          # Phase 4: 消融研究结果
│   ├── ablation_results.json      # 一般消融实验
│   └── ablation_study_results.json # 组件贡献分析 (必需)
├── attention_analysis/ # Phase 5: 注意力分析结果
├── synthetic/         # 合成数据实验结果
└── experiment_summary.json  #  完整实验总结
```

##  核心特性

- **真实模型数据**: 支持使用真实的Llama-3.1-8B模型数据
- **Pre-RoPE聚类**: 解决位置编码对相似度计算的干扰
- **渐进式压缩**: 实时流式处理，支持任意长度序列
- **GPU加速**: 高效的GPU聚合器
- **学术级输出**: 自动生成论文发表质量的图表

##  性能亮点

- **压缩比**: 支持10:1到100:1的压缩比
- **召回率**: 在4%压缩比下保持97.5%的needle召回率
- **实时性**: GPU聚合保持推理性能
- **鲁棒性**: 处理各种序列长度和语义内容

##  生成的论文图表

1. **Figure 1**: 语义流形分布 - UMAP/t-SNE投影
2. **Figure 2**: Needle热力图 - 性能vs序列长度和深度
3. **Figure 3**: 压缩性能曲线 - PPL和召回率vs压缩比
4. **Figure 4**: 消融研究对比 - 组件贡献分析
5. **Figure 5**: 注意力能量谱 - score_bias效果
6. **Figure 6**: 效率分析 - CPU/GPU性能对比

##  技术细节

MP-KVM通过以下创新解决了Transformer KV缓存压缩的关键问题：

1. **RoPE位置干扰**: 提取Pre-RoPE向量进行聚类
2. **实时流处理**: 增量更新避免历史数据重复累积
3. **注意力补偿**: log-count权重防止centroid饥饿
4. **GPU优化**: 异步聚合保持推理性能

更多技术细节请参考代码注释和实验结果。
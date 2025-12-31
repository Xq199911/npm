# MP-KVM: Manifold-Partitioned Key-Value Memory

## 🚀 快速开始

### 环境准备
```bash
# 1. 激活conda环境
conda activate py

# 2. 验证环境
python demo.py

# 3. 确保模型文件存在 (可选，用于真实数据实验)
ls model/Llama-3.1-8B-Instruct/
```

### 一键运行所有实验
```bash
# 使用真实数据运行完整实验 (推荐 - 需要GPU)
python run_complete_experiment.py --use-real-model

# 使用合成数据运行完整实验（快速测试用）
python run_complete_experiment.py

# 快速测试模式（减少参数，5-10分钟完成）
python run_complete_experiment.py --quick
```

### 分阶段运行实验
```bash
# Phase 1: 基准线比较 (Baseline Comparison)
python run_complete_experiment.py --phase 1 --use-real-model

# Phase 2: 流形可视化 (Manifold Visualization)
python run_complete_experiment.py --phase 2 --use-real-model

# Phase 3: 压缩实验 (Compression Experiments)
python run_complete_experiment.py --phase 3 --use-real-model

# Phase 4: 消融研究 (Ablation Studies)
python run_complete_experiment.py --phase 4 --use-real-model

# Phase 5: 注意力分析 (Attention Analysis)
python run_complete_experiment.py --phase 5 --use-real-model
```

### 生成论文图表
```bash
# 使用已有的实验数据生成论文图表
python generate_paper_figures.py
```

### 其他常用命令
```bash
# 快速测试（减少参数，合成数据）
python run_complete_experiment.py --quick

# 跳过性能分析
python run_complete_experiment.py --phase 5 --skip-performance

# 指定自定义模型路径
python run_complete_experiment.py --use-real-model --model-path "path/to/your/model"

# 显示帮助信息
python run_complete_experiment.py --help
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

## 🎯 实验阶段详解

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

4. **Phase 4**: 消融研究
   - 验证各组件(RoPE、权重、聚类)的贡献
   - 对比不同配置的性能差异
   - 输出：`results/ablation/`

5. **Phase 5**: 注意力分析
   - 分析注意力权重分布和能量补偿效果
   - 生成注意力能量谱图表
   - 输出：`results/attention_analysis/`

## 💡 使用场景示例

### 学术研究者
```bash
# 完整实验流程 (论文复现)
python run_complete_experiment.py --use-real-model

# 只验证关键结果
python run_complete_experiment.py --phase 3 --use-real-model

# 生成论文图表
python generate_paper_figures.py
```

### 开发者/工程师
```bash
# 快速验证代码
python run_complete_experiment.py --quick

# 性能基准测试
python run_complete_experiment.py --phase 5 --use-real-model

# 自定义配置测试
python run_complete_experiment.py --phase 1 --model-path "your/model"
```

### 新手用户
```bash
# 环境验证
python demo.py

# 快速上手
python run_complete_experiment.py --quick

# 分阶段学习
python run_complete_experiment.py --phase 1  # 从基准线开始
python run_complete_experiment.py --phase 2  # 然后可视化
```

## 📊 输出结果

运行完成后，所有结果保存在 `results/` 目录：

```
results/
├── baseline/           # Phase 1: 基准线比较结果
│   └── enhanced_baseline_results.json
├── figures/           # 📈 论文图表 (6个PNG文件)
│   ├── manifold_clustering.png
│   ├── topic_transitions.png
│   └── [其他图表文件...]
├── compression_sweep/ # Phase 3: 压缩实验结果
│   ├── h2o_results.json
│   ├── no_compression_results.json
│   ├── random_eviction_results.json
│   └── streamingllm_results.json
├── ablation/          # Phase 4: 消融研究结果
├── attention_analysis/ # Phase 5: 注意力分析结果
├── synthetic/         # 合成数据实验结果
└── experiment_summary.json  # 📋 完整实验总结
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

### 命令格式错误
如果遇到"unrecognized arguments"错误，请使用正确格式：
```bash
# ❌ 错误格式
python run_complete_experiment.py phase --phase 1

# ✅ 正确格式
python run_complete_experiment.py --phase 1
```

### PyTorch环境问题
如果遇到PyTorch加载错误：
1. 检查CUDA版本兼容性：`nvcc --version`
2. 确保使用正确的conda环境
3. 重新安装PyTorch：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### 内存不足 (CUDA out of memory)
对于大模型实验：
- **必须使用GPU**：`--use-real-model`需要GPU环境
- 减少批次大小：在实验脚本中调整`batch_size`参数
- 减少序列长度：修改实验配置中的`max_length`参数
- 使用更小的模型：尝试`--model-path`指定较小模型

### 模型文件缺失
如果提示找不到Llama模型：
```bash
# 下载模型到指定目录
# 或者修改默认路径
python run_complete_experiment.py --use-real-model --model-path "your/model/path"
```

### 实验失败调试
- **分阶段运行**：每个阶段都可以独立测试
  ```bash
  python run_complete_experiment.py --phase 1  # 只运行Phase 1
  ```
- **使用合成数据**：快速验证代码逻辑
  ```bash
  python run_complete_experiment.py --quick  # 合成数据快速测试
  ```
- **检查日志**：查看`results/`目录中的错误信息
- **环境验证**：运行`python demo.py`检查基本环境

## 📝 技术细节

MP-KVM通过以下创新解决了Transformer KV缓存压缩的关键问题：

1. **RoPE位置干扰**: 提取Pre-RoPE向量进行聚类
2. **实时流处理**: 增量更新避免历史数据重复累积
3. **注意力补偿**: log-count权重防止centroid饥饿
4. **GPU优化**: 异步聚合保持推理性能

更多技术细节请参考代码注释和实验结果。
# MLS_Project - Federated Learning for Smart City Traffic

## 简介 (Introduction)
本项目旨在构建一个针对智慧城市交通监控的联邦学习系统。核心目标是在不需要将原始视频数据传输到中央服务器的情况下，通过边缘计算节点进行协作模型训练，从而提高模型在异构环境下的适应能并保护隐私。

## 环境要求 (Requirements)
* Python 3.8+
* 建议使用 Conda 或 venv 创建虚拟环境。

安装依赖 (Install Dependencies):
```bash
pip install grpcio grpcio-tools protobuf torch numpy onnxruntime psutil
```

## 运行说明 (Running Instructions)

本项目包含服务端 (Server) 和客户端 (Client) 两部分。需要分别在两个终端中运行。

### 1. 启动服务端 (Start Server)
服务端负责协调联邦学习过程，接收客户端的参数更新并聚合全局模型。

在项目根目录下运行：
```bash
python server/server.py
```
成功启动后终端会显示: `Starting server on [::]:50051...`

### 2. 启动客户端 (Start Client)
客户端模拟边缘设备，进行本地每一轮的训练，应用差分隐私 (Differential Privacy) 处理，并将更新发送给服务端。

在新的终端窗口，切换到项目根目录运行：
```bash
python client/client.py
```
客户端将连接到本地服务端 (`localhost:50051`) 并开始模拟联邦学习过程。

### 3. 运行 ONNX 基准测试 (Run ONNX Baseline)
验证 ONNX 模型的导出和推理功能：
```bash
python client/run_onnx_baseline.py
```

## 模型训练与压缩 (Model Training & Compression)

### 4. 训练 Baseline 模型 (Train Baseline Model)
如果只需要训练基础的 YOLOv11n 模型作为性能基准：

```bash
python client/train_baseline.py
```

**可选参数**:
- `--epochs`: 训练轮数 (默认: 50)
- `--batch-size`: 批次大小 (默认: 16)  
- `--lr`: 学习率 (默认: 0.001)
- `--data-dir`: 数据集路径 (默认: `./data/ua-detrac-10k`)
- `--api-key`: Roboflow API密钥 (默认已提供)

**示例**:
```bash
python client/train_baseline.py --epochs 100 --batch-size 32 --lr 0.0005
```

训练完成后，模型将保存到 `client/baseline_model.pth`。

### 5. 获取模型压缩变体 (Get Model Compression Variants)
运行完整的压缩和部署流水线，获得三种压缩变体：

```bash
python client/compress_and_deploy.py
```

**该脚本将演示以下压缩技术**:

1. **结构化剪枝 (Structured Pruning)**
   - 移除50%的模型权重
   - 生成 `pruned_model.onnx`

2. **动态量化 (Dynamic Quantization)**
   - INT8量化减少模型大小
   - 生成 `quantized_model.onnx`

3. **知识蒸馏 (Knowledge Distillation)**
   - Teacher模型: 标准YOLOv11n (width_mult=1.0)
   - Student模型: 压缩版YOLOv11n (width_mult=0.5)
   - 生成 `distilled_model.onnx`

**输出文件**:
所有压缩模型将保存到 `exported_models/` 目录：
- `pruned_model.onnx` - 剪枝版本
- `quantized_model.onnx` - 量化版本  
- `distilled_model.onnx` - 蒸馏版本

**性能对比**:
脚本会自动显示各模型的参数量和压缩比：
```
Teacher model: 2,800,000 parameters
Student model: 700,000 parameters
Compression ratio: 4.00x
```

## 项目结构 (Project Structure)
* `server/`: 包含服务端逻辑 (`server.py`)。
* `client/`: 包含客户端逻辑 (`client.py`) 和训练代码。
* `protos/`: gRPC 服务定义文件及生成的 Python 代码。
* `utils/`: 工具函数（序列化、隐私引擎、硬件感知优化、ONNX 助手等）。
  * `hardware.py`: 实现设备性能分析与资源调度。
  * `privacy.py`: 差分隐私引擎。
  * `compression.py` & `deployment.py`: 模型压缩与部署工具。
* `Project.md`: 项目详细说明文档。

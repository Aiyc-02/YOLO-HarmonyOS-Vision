# YOLO-HarmonyOS-Vision: 基于 ArkTS 的端侧目标检测部署探索

本项目记录了在 HarmonyOS NEXT 系统下，尝试使用纯 ArkTS 语言结合 MindSpore Lite 框架，将轻量级 YOLO 目标检测模型部署到移动端侧的开发过程与阶段性成果。

## 📌 项目背景与尝试

目前关于 HarmonyOS 端侧部署视觉模型的开源实践相对较少。本项目跳过了常见的 C++ (NAPI) 底层封装路线，尝试直接调用鸿蒙原生的 `@kit.MindSporeLiteKit` 接口，以期验证纯原生链路在端侧 AI 图像推理上的工程可行性。

## 📊 阶段性取得的成果

经过基础的底层调试与逻辑重构，目前工程初步实现了以下验证：

* **原生推理链路打通**：成功利用 MindSpore Lite 离线加载 `.ms` 模型，在真机环境下完成了图像的前向推理。
* **张量数据解析与坐标映射**：实现了对模型输出多维张量的结构解析，处理了坐标归一化及边界裁剪，并完成了从特征图映射到设备物理屏幕宽高的自适应对齐。
* **按类别非极大值抑制 (Class-Specific NMS)**：为解决多目标识别中的重叠问题，编写了基础的按类别独立 NMS 算法，初步改善了异类物体的误抑制现象。
* **交互逻辑层优化**：对推理结果引入了预缓存机制。通过界面滑块动态调节置信度阈值时，直接基于内存缓存进行数据过滤与 Canvas 重绘，避免了重复调用底层模型带来的算力消耗。

---

## ⚙️ 模型获取与端侧格式转换路线

由于鸿蒙底层的 MindSpore 推理框架不支持直接运行 PyTorch 模型，本工程探索了从官方预训练模型到端侧 `.ms` 格式的完整转换链路。若要自行准备模型，请参考以下流程：

### 1. 下载预训练模型 (.pt)
* **来源**：Ultralytics 官方仓库
* **地址**：[https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* **说明**：在该仓库的 README 或 Docs 中下载轻量级的预训练权重（推荐使用参数量极小的 `yolov8n.pt` 或类似规格的模型），也可以使用该框架自行训练得到特定场景的 `.pt` 权重。

### 2. 导出为中间件格式 (.onnx)
利用 Ultralytics 自带的导出功能，将 PyTorch 模型转化为通用的 ONNX 结构。在 Python 环境中执行以下命令（建议开启 `simplify` 消除冗余算子）：
```bash
yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True

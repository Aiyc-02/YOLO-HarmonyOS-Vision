Markdown
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
* **说明**：在该仓库的 README 或 Docs 中下载轻量级的预训练权重（本项目使用的是官方预训练的yolo26n.pt），也可以使用该框架自行训练得到特定场景的 `.pt` 权重。

### 2. 导出为中间件格式 (.onnx)
利用 Ultralytics 自带的导出功能，将 PyTorch 模型转化为通用的 ONNX 结构。在 Python 环境中执行以下命令（建议开启 `simplify` 消除冗余算子）：

```bash
yolo export model=yolov26n.pt format=onnx imgsz=640 simplify=True
```
### 3. 转换为鸿蒙端侧格式 (.ms)
此步骤需使用华为官方提供的 MindSpore Lite Converter（离线模型转换工具），将 ONNX 算子映射为端侧底层的高性能算子。

工具下载地址:https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html
(本文下载的是mindspore-lite-2.9.0-linux-x64，在Windows系统下转换时没有成功，在Ubantu22.04系统上进行转换的)

转换指南参考：https://www.mindspore.cn/lite/docs/zh-CN/master/converter/converter_tool.html

基本转换命令示例：

Bash
# 将解压后的 converter 工具加入环境变量后执行：
```bash
converter_lite --modelFile=yolov26n.onnx --fmk=ONNX --outputFile=yolo26n --targetDevice=CPU
```
注：转换完成后，只需将生成的 yolo26n.ms 放入本工程的 entry/src/main/resources/rawfile/ 目录下即可被代码自动读取。

🛠️ 开发与测试环境设置
1. IDE 与 SDK 环境
开发工具: HUAWEI DevEco Studio

建议版本: 4.0 Release / NEXT Developer Preview 及以上版本

工具下载地址: DevEco Studio 官方下载中心

SDK 环境: HarmonyOS NEXT (API 11 / API 12)

2. 硬件测试设备
运行载体: 华为鸿蒙真机（因涉及底层 NPU/CPU 算力调度与 MindSpore 底层依赖，暂不支持通过电脑本地模拟器进行推理验证）。

📝 后续探索方向
本项目目前主要聚焦于底层检测链路的跑通验证，仍有待进一步探索的空间：

对不同 YOLO 变体导出时产生的差异化张量输出结构（如 [1, 6, 300] 与 [1, 300, 6] 等排布方式），目前的兼容范围仍需进一步扩大测试。

目前工程主要针对单帧静态图像进行检测，关于连续视频流（Camera Frame）检测时的内存开销控制与帧率表现尚未进行压力验证。

欢迎感兴趣的开发者参考代码，交流探讨或提出改进建议。

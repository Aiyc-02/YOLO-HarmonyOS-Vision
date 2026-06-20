# 🎯 YOLO-HarmonyOS-Vision

🚀 **纯原生、零 C++ 依赖、极度流畅的鸿蒙端侧 YOLO 目标检测引擎。**

基于 HarmonyOS NEXT (API 11/12) 的 ArkTS 语言与 MindSpore Lite 端侧推理框架，徒手实现张量解析、动态坐标归一化与按类别的非极大值抑制（Class-Specific NMS）。让你在鸿蒙手机上体验 60 帧丝滑的离线 AI 视觉。

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![HarmonyOS](https://img.shields.io/badge/HarmonyOS-NEXT-green)
![MindSpore](https://img.shields.io/badge/MindSpore-Lite-purple)

---

## ✨ 核心亮点

与传统的“C++ 底层封包”或“云端 API 调用”方案不同，本项目主打 **“纯血原生与极致自适应”**：

- **🧠 纯 ArkTS 端侧推理**：彻底抛弃笨重的 C++ (NAPI) 胶水代码，使用鸿蒙原生 `@kit.MindSporeLiteKit` 直接加载 `.ms` 模型进行本地推理，拒绝内存泄漏。
- **🛡️ 智能张量解析 (Smart Tensor Parsing)**：自动探测模型输出的 Shape（兼容 `[1, 6, 300]` 与 `[1, 300, 6]` 等不同导出的 YOLO 变体），告别坐标乱码。
- **⚔️ 工业级 Class-Specific NMS**：独立手搓按类别的非极大值抑制算法，完美解决不同类别物体（如“人”和“背包”）重叠时的互相误杀。
- **📱 绝对坐标自适应**：不论是横屏全景还是竖屏长图，通过动态监听渲染面积（`onAreaChange`），确保检测框在任何尺寸、任何比例的设备上都严丝合缝地咬住物体。
- **🎛️ 实时置信度过滤**：创新的“预缓存”架构！拖动 UI 滑块调节检测严苛度时，无需重新唤醒 AI 推理，实现 0 延迟的瞬间画面重绘。
- **💎 玻璃拟态 UI (Glassmorphism)**：使用鸿蒙原生 `backdropBlur` 打造深色高级磨砂面板，体验极其优雅。

---

## 🛠️ 快速开始

### 1. 环境要求
- **IDE**: DevEco Studio (推荐 4.0 Release 及以上)
- **SDK**: HarmonyOS NEXT (API 11 / API 12)
- **设备**: 华为鸿蒙真机（因涉及 NPU/CPU 底层推理，不建议使用模拟器）

### 2. 模型准备
本项目默认读取 `yolo26n.ms` 模型。
1. 请确保你已经将训练或转换好的 MindSpore Lite 模型（`.ms` 格式）重命名为 `yolo26n.ms`。
2. 将该文件放入项目的 `entry/src/main/resources/rawfile/` 目录下。

### 3. 编译运行
在 DevEco Studio 中打开项目，直接点击 `Run` 部署到你的鸿蒙设备上即可。由于使用了纯原生的 `photoAccessHelper` 和 `cameraPicker`，**本项目无需申请任何高危相机/图库权限即可安全运行**。

---

## ⚙️ 核心逻辑解析 (对于想学习的开发者)

本项目最值得参考的逻辑集中在 `Index.ets` 中的张量解析阶段：

1. **为什么不需要申请相机权限？**
   摒弃了底层的 `CameraSession` 数据流流转，采用安全极简的 `cameraPicker.pick()` 唤起系统独立进程拍照，杜绝了底层缓冲池卡死黑屏的风险。
   
2. **为什么滑动过滤如此流畅？**
   在 `extractAndCacheBoxes` 阶段，将所有低阈值（>10%）的候选原始框提取到内存数组 `cachedRawBoxes` 中。当用户滑动 UI 的 `confidenceThreshold` 时，直接操作缓存数组进行 NMS 过滤和 Canvas 重绘。

---

## 🤝 贡献与参与

这可能是全网最早一批采用全 ArkTS 原生链路跑通端侧 YOLO 的开源探索。如果你在部署其他 YOLO 变体（如 YOLOv8, YOLO11）时遇到了张量解析的问题，或者有更好的性能优化思路，欢迎提交 **Issue** 或 **Pull Request**！

## 📄 协议

本项目基于 **MIT License** 开源，欢迎自由使用、修改与商用。

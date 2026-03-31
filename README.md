# WonderPi 机械臂 · 自动录视频与拆帧识别

基于树莓派（WonderPi）与机械臂，搭建「录制 → 拆帧 → 识别 → 执行」的闭环，用于环境/任务过程记录与视觉引导动作。

## 一、项目目标

| 能力 | 说明 |
|------|------|
| **自动录制视频** | 按触发条件或定时录制环境/任务过程（USB 摄像头、CSI 相机或 WonderPi 配套视觉模块） |
| **自动拆帧** | 将视频转为有序图像序列，便于批处理与标注 |
| **图像识别** | 对单帧或序列做目标检测 / 分类，输出类别、框或关键点 |
| **机械臂控制** | 根据识别结果执行抓取、指向、跟随等动作（与舵机/总线舵机协议对接） |

## 二、整体流程

```text
相机 ──► 录制(.mp4 等) ──► 拆帧(图像序列) ──► 模型推理 ──► 坐标/类别
                                                      │
                                                      ▼
                                              机械臂运动规划与执行
```

建议将各环节做成可独立运行的脚本或服务，再通过配置文件或消息（如 ROS 2 / 简单 JSON）串联，便于在树莓派上分步调试。

## 三、硬件与环境

- **主控**：树莓派（WonderPi 套件所适配型号）
- **视觉**：CSI / USB 摄像头（分辨率与帧率按算力选择，例如 640×480@30fps 起步）
- **机械臂**：WonderPi 配套机械臂及驱动板（具体型号以官方文档为准）
- **供电与固定**：注意臂与相机相对位姿稳定，便于手眼标定（若需像素坐标到机械坐标转换）

## 四、软件栈建议（可按实际套件调整）

| 环节 | 常用方案 |
|------|----------|
| 录制 / 拆帧 | `ffmpeg`、`OpenCV`（Python：`cv2.VideoCapture` / `VideoWriter`） |
| 检测 / 分类 | `ONNX Runtime`、`TensorFlow Lite`、`PyTorch`（轻量模型如 YOLOv8n、MobileNet） |
| 机械臂 | WonderPi 官方 SDK / 串口或 CAN 协议文档；若有 ROS 2 支持可统一用话题与服务 |

在 ARM 上优先选用 **INT8 量化** 或 **TFLite/ONNX** 小模型，以降低延迟与发热。

## 五、目录规划（建议）

后续可在本仓库中按模块落代码，例如：

```text
ThomasLauWonderPiDemo/
├── README.md                 # 本说明
├── config/                   # 相机参数、臂限位、模型路径等
├── scripts/
│   ├── common.py             # 配置与路径
│   ├── list_cameras.py       # 探测 V4L2 / rpicam / OpenCV 可用相机
│   ├── record_video.py       # 自动录制
│   ├── extract_frames.py     # 视频拆帧
│   ├── infer.py              # 单帧/批量推理
│   ├── arm_demo.py           # 根据识别结果驱动机械臂
│   └── pipeline.py           # 一键串联（录→拆帧→推理→臂控演示）
├── models/                   # 权重文件（勿提交大文件，可用下载脚本）
└── data/
    ├── videos/               # 原始视频
    └── frames/               # 拆帧输出
```

## 六、快速开始

1. **安装依赖**

   ```bash
   cd /home/pi/apps/ThomasLauWonderPiDemo
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip -r requirements.txt
   ```

2. **探测相机（无画面时先跑）**

   ```bash
   python3 scripts/list_cameras.py
   ```

   树莓派新版系统往往**没有** `/dev/video0`，CSI 相机需用 `rpicam-vid` / `picamera2`，不能仅靠 OpenCV 的 `index=0`。

3. **录制（输出到 `data/videos/`）**

   ```bash
   python3 scripts/record_video.py --seconds 10
   ```

   默认 `camera.backend: auto`：先试 OpenCV，再试 `rpicam-vid`，再试 `picamera2`（虚拟环境需 `pip install picamera2`）。也可强制：`--backend opencv` / `rpicam` / `picamera2`。

   若编码失败，可编辑 `config/default.yaml` 将 `recording.fourcc` 改为 `MJPG` 或 `XVID`。

4. **拆帧**

   ```bash
   python3 scripts/extract_frames.py --video data/videos/capture_YYYYMMDD_HHMMSS.mp4
   ```

5. **推理（默认 Haar 人脸检测；ONNX 分类见下）**

   ```bash
   python3 scripts/infer.py --source data/frames/run_YYYYMMDD_HHMMSS/frame_000000.jpg
   python3 scripts/infer.py --source data/frames/run_YYYYMMDD_HHMMSS --json-out data/frames/_batch.json
   ```

   使用 ONNX：将模型放到 `models/model.onnx`，安装 `onnxruntime`，在 `config/default.yaml` 中设置 `inference.method: onnx`，并按模型输入调整 `onnx_input_size`；类别名可选 `config/class_names.example.txt`。

6. **机械臂演示（默认 mock，仅打印归一化坐标）**

   ```bash
   python3 scripts/arm_demo.py --json data/frames/_batch.json
   ```

7. **一键流水线（录一段 → 拆帧 → 对首帧推理 → mock 臂控）**

   ```bash
   python3 scripts/pipeline.py --seconds 5
   ```

   无摄像头时可用任意本地视频手动跑：`extract_frames.py` → `infer.py` → `arm_demo.py`。

8. **Web UI（推荐）**

   ```bash
   python3 app.py --port 5000
   ```

   打开浏览器访问 `http://<树莓派IP>:5000`（本机可用 `http://localhost:5000`）。
   Web 界面整合了：实时预览、录制、拆帧、识别、机械臂控制、一键流水线。

## 七、开发顺序建议

1. 打通 **相机预览 + 短视频录制**，确认存储路径与磁盘空间。
2. 实现 **拆帧** 与命名规则，保证与标注/训练流水线一致。
3. 在 PC 或 Pi 上训练/导出 **轻量模型**，在 Pi 上实测 FPS 与准确率。
4. **单动作开环**（如固定点抓取），再过渡到 **跟踪/跟随**（需控制周期与预测）。

## 八、注意事项

- 长时间录制注意 SD 卡寿命与散热；可将视频写到 USB 硬盘。
- 机械臂运动前务必设置 **软限位与急停**，避免误识别导致碰撞。
- 模型与 `numpy`/推理库版本需与树莓派系统（32/64 位）一致，避免二进制不兼容。

## 九、许可与致谢

硬件与 SDK 以 **WonderPi / 厂商文档** 为准；本 README 描述的是通用实现思路，具体 API 名称以官方为准。

---

**维护者**：Thomas Lau · 项目代号：`ThomasLauWonderPiDemo`

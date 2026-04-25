# ThomasLauWonderPiDemo · 人脸追踪

最小化的人脸追踪 Demo。摄像头检测人脸 → 计算像素误差 → 通过 IK 让机械臂
末端绕 Z 轴旋转 (yaw) 与改变 pitch，使脸保持在画面中心。

## 模块结构

```
app.py                         Flask WebUI（相机 / 检测 / 追踪 / 配置）
scripts/
  arm_bridge.py                ROS2 端长驻进程（Docker 容器内运行，端口 9091）
  common.py                    路径 / YAML 配置 / 工具函数
  deptrum_camera.py            Deptrum Aurora 930 SDK Python 包装
  list_cameras.py              枚举本机相机
  face_detector.py             FaceDetector：Mediapipe 优先 + Haar 回退
  face_tracker.py              FaceTracker：闭环 P 控制 + arm_bridge IK
config/
  default.yaml                 相机 / 检测器 / 追踪参数
web/
  index.html                   单页 WebUI
```

## 快速开始

```bash
# 1. 安装依赖
.venv/bin/pip install -r requirements.txt
# 可选：更高精度
# .venv/bin/pip install mediapipe

# 2. (机械臂联动需要) 在 Docker 容器内启动 arm_bridge
docker exec -d -u ubuntu ArmPiUltra bash -c \
  'source ~/.zshrc 2>/dev/null && python3 /home/ubuntu/arm_bridge.py'

# 3. 启动 WebUI
.venv/bin/python app.py
# 浏览器访问 http://<pi>:5000
```

## 控制思路

- 摄像头随末端运动；末端在 `home_pose` (x, y, z) 处保持注视姿态
- 每个控制周期 (`loop_hz`)：
  1. 读一帧 → 跑检测 → 选「主脸」（默认最大框）
  2. 像素误差 (cx-w/2, cy-h/2) → 角度误差（按 FOV 比例）
  3. P 控制 + EMA 平滑 + 单步限幅 → 累加到 yaw / pitch
  4. 旋转 (x0, y0) 得到新 (x, y)，用 pitch 调用 `/ik_move`
- 连续 N 帧或 T 秒未检测到人脸 → 归位到 `home_pose`

所有参数都可在 WebUI 右侧实时改并写回 `config/default.yaml`。

## API

| 路径 | 方法 | 说明 |
|---|---|---|
| `/`                         | GET  | WebUI 单页 |
| `/api/health`               | GET  | 综合状态（相机/检测器/追踪/arm_bridge） |
| `/api/camera/open`          | POST | 打开相机（自动选 backend） |
| `/api/camera/close`         | POST | 关闭相机（同时停止追踪） |
| `/api/camera/status`        | GET  | 是否已打开 + backend 名 |
| `/api/stream?annotate=1`    | GET  | MJPEG 流（带绿/灰检测框） |
| `/api/snapshot?annotate=1`  | GET  | 单张 JPEG（响应头 X-Faces） |
| `/api/face/detect`          | GET  | 单帧人脸检测（JSON） |
| `/api/track/start`          | POST | 开启追踪线程 |
| `/api/track/stop`           | POST | 停止追踪线程 |
| `/api/track/status`         | GET  | 追踪状态（yaw/pitch/fps/丢失帧/最近错误等） |
| `/api/arm/home`             | POST | 机械臂回到 home_pose |
| `/api/arm/status`           | GET  | arm_bridge 是否在线 |
| `/api/config`               | GET  | 读当前配置（含运行时覆盖） |
| `/api/config`               | POST | `{"patch": {...}, "persist": false}` 更新配置 |

## 配置要点（config/default.yaml）

```yaml
face_detect:
  backend: haar          # mediapipe | haar（mediapipe 未装会自动回退）

face_tracking:
  pick: largest          # largest | center
  deadzone_px: 25        # 像素死区（小于此值不动）
  smoothing: 0.5         # EMA 平滑（0=关）
  loop_hz: 10            # 控制频率
  fov_h_deg: 60          # 估计水平 FOV，错则收敛过慢/过冲
  fov_v_deg: 45
  yaw_kp: 0.45
  pitch_kp: 0.45
  max_yaw_step_deg: 8    # 单步上限，防大动作
  max_pitch_step_deg: 6
  home_pose: {x: 0.18, y: 0.0, z: 0.10, pitch: -10.0}
  yaw_min_deg: -75
  yaw_max_deg:  75
  pitch_min_deg: -45
  pitch_max_deg:  35
```

调参建议：

- **抖动**：增大 `smoothing` 或 `deadzone_px`，减小 Kp
- **跟不上**：增大 Kp 或 `loop_hz`；如果一致地偏一边，调 `fov_*` 与实际相机一致
- **失控/急动**：减小 `max_*_step_deg` 与 Kp

#!/usr/bin/env python3
"""Persistent ROS2 bridge for fast arm control.

Runs INSIDE the Docker container as a long-lived process, keeping a ROS2
node alive so that every command avoids the 2-3 s docker-exec + DDS
discovery overhead.

Start:
    docker exec -d -u ubuntu ArmPiUltra bash -c \
        'source ~/.zshrc 2>/dev/null && python3 /home/ubuntu/arm_bridge.py'

HTTP API (port 9090):
    POST /servo     {"positions": {"6": 500, "1": 700}, "duration": 0.8}
    POST /ik        {"x": 0.18, "y": 0.0, "z": 0.05, ...}  -> {"pulses": [...]}
    POST /ik_move   {"x": 0.18, "y": 0.0, "z": 0.05, ...}  -> move servos
    POST /buzzer    {"freq": 1900, "on_time": 0.1}
    GET  /health
"""

from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from kinematics_msgs.srv import SetRobotPose
from ros_robot_controller_msgs.msg import BuzzerState

SERVO_JOINT_MAP = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2}


class ArmBridgeNode(Node):
    def __init__(self):
        super().__init__("arm_bridge")
        self.servo_pub = self.create_publisher(
            ServosPosition, "/servo_controller", 10
        )
        self.buzzer_pub = self.create_publisher(
            BuzzerState, "/ros_robot_controller/set_buzzer", 10
        )
        self.ik_client = self.create_client(
            SetRobotPose, "/kinematics/set_pose_target"
        )
        self._ik_lock = threading.Lock()
        self.get_logger().info("arm_bridge node ready")

    def pub_servos(self, positions: dict, duration: float) -> None:
        msg = ServosPosition()
        msg.duration = float(duration)
        msg.position_unit = "pulse"
        for sid, pos in positions.items():
            sp = ServoPosition()
            sp.id = int(sid)
            sp.position = float(pos)
            msg.position.append(sp)
        self.servo_pub.publish(msg)

    def pub_buzzer(self, freq: int, on_time: float) -> None:
        msg = BuzzerState()
        msg.freq = int(freq)
        msg.on_time = float(on_time)
        msg.off_time = 0.01
        msg.repeat = 1
        self.buzzer_pub.publish(msg)

    def call_ik(
        self,
        x: float, y: float, z: float,
        pitch: float = -90.0,
        pitch_range: tuple = (-90.0, 90.0),
        resolution: float = 1.0,
        duration: float = 0.8,
    ) -> tuple[list[int] | None, str | None]:
        with self._ik_lock:
            if not self.ik_client.wait_for_service(timeout_sec=3.0):
                return None, "IK service not available"

            req = SetRobotPose.Request()
            req.position = [float(x), float(y), float(z)]
            req.pitch = float(pitch)
            req.pitch_range = [float(pitch_range[0]), float(pitch_range[1])]
            req.resolution = float(resolution)
            req.duration = float(duration)

            future = self.ik_client.call_async(req)
            deadline = time.monotonic() + 10.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.005)

            if not future.done():
                future.cancel()
                return None, "IK timeout"

            result = future.result()
            if result is None or not result.success:
                return None, "IK solver failed"

            return list(result.pulse), None


_node: ArmBridgeNode | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        if self.path == "/health":
            self._json({"ok": True, "uptime": time.monotonic()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        data = self._body()

        if self.path == "/servo":
            positions = {int(k): int(v) for k, v in data.get("positions", {}).items()}
            dur = float(data.get("duration", 1.0))
            _node.pub_servos(positions, dur)
            self._json({"ok": True})

        elif self.path == "/ik":
            pulses, err = _node.call_ik(
                data["x"], data["y"], data["z"],
                pitch=data.get("pitch", -90.0),
                pitch_range=data.get("pitch_range", [-90.0, 90.0]),
                duration=data.get("duration", 0.8),
            )
            if err:
                self._json({"ok": False, "error": err}, 500)
            else:
                self._json({"ok": True, "pulses": pulses})

        elif self.path == "/ik_move":
            dur = float(data.get("duration", 0.8))
            pulses, err = _node.call_ik(
                data["x"], data["y"], data["z"],
                pitch=data.get("pitch", -90.0),
                pitch_range=data.get("pitch_range", [-90.0, 90.0]),
                duration=dur,
            )
            if err:
                self._json({"ok": False, "error": err}, 500)
                return

            servos = {}
            for idx, pulse in enumerate(pulses):
                sid = SERVO_JOINT_MAP.get(idx)
                if sid is not None:
                    servos[sid] = pulse
            _node.pub_servos(servos, dur)
            self._json({"ok": True, "pulses": pulses})

        elif self.path == "/buzzer":
            _node.pub_buzzer(
                int(data.get("freq", 1900)),
                float(data.get("on_time", 0.1)),
            )
            self._json({"ok": True})

        else:
            self._json({"error": "not found"}, 404)


def main():
    global _node

    rclpy.init()
    _node = ArmBridgeNode()

    executor = SingleThreadedExecutor()
    executor.add_node(_node)
    threading.Thread(target=executor.spin, daemon=True).start()

    # Wait for DDS discovery to settle
    time.sleep(1.0)

    server = HTTPServer(("0.0.0.0", 9090), Handler)
    print("arm_bridge HTTP server listening on :9090", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

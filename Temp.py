# 第一步：启动容器
sudo docker start lerobot_v043

# 第二步：进入容器（前面变成root@nvidia-desktop）
sudo docker exec -it lerobot_v043 bash

# teleoperate + cam
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=jiao_follower_arm \
  --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: '242322077448', width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=jiao_leader_arm \
  --display_data=true



(base) root@nvidia-desktop:/workspace# lerobot-teleoperate   --robot.type=so101_follower   --robot.port=/dev/ttyACM0   --robot.id=jiao_follower_arm   --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: '242322077448', width: 640, height: 480, fps: 30} }"   --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1   --teleop.id=jiao_leader_arm   --display_data=true
INFO 2026-03-02 10:40:58 eoperate.py:188 {'display_data': True,
 'fps': 60,
 'robot': {'calibration_dir': None,
           'cameras': {'front': {'color_mode': <ColorMode.RGB: 'rgb'>,
                                 'fps': 30,
                                 'height': 480,
                                 'rotation': <Cv2Rotation.NO_ROTATION: 0>,
                                 'serial_number_or_name': '242322077448',
                                 'use_depth': False,
                                 'warmup_s': 1,
                                 'width': 640}},
           'disable_torque_on_disconnect': True,
           'id': 'jiao_follower_arm',
           'max_relative_target': None,
           'port': '/dev/ttyACM0',
           'use_degrees': False},
 'teleop': {'calibration_dir': None,
            'id': 'jiao_leader_arm',(base) root@nvidia-desktop:/workspace# lerobot-teleoperate   --robot.type=so101_follower   --robot.port=/dev/ttyACM0   --robot.id=jiao_follower_arm   --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: '242322077448', width: 640, height: 480, fps: 30} }"   --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1   --teleop.id=jiao_leader_arm   --display_data=true
INFO 2026-03-02 10:40:58 eoperate.py:188 {'display_data': True,
 'fps': 60,
 'robot': {'calibration_dir': None,
           'cameras': {'front': {'color_mode': <ColorMode.RGB: 'rgb'>,
                                 'fps': 30,
                                 'height': 480,
                                 'rotation': <Cv2Rotation.NO_ROTATION: 0>,
                                 'serial_number_or_name': '242322077448',
                                 'use_depth': False,
                                 'warmup_s': 1,
                                 'width': 640}},
           'disable_torque_on_disconnect': True,
           'id': 'jiao_follower_arm',
           'max_relative_target': None,
           'port': '/dev/ttyACM0',
           'use_degrees': False},
 'teleop': {'calibration_dir': None,
            'id': 'jiao_leader_arm',
            'port': '/dev/ttyACM1',
            'use_degrees': False},
 'teleop_time_s': None}
[2026-03-02T10:40:58Z WARN  re_viewer::native] It looks like you are running the Rerun Viewer inside a Docker container. This is not officially supported, and may lead to performance issues and bugs. See https://github.com/rerun-io/rerun/issues/6835 for more.
[2026-03-02T10:40:58Z INFO  re_grpc_server] Listening for gRPC connections on 0.0.0.0:9876. Connect by running `rerun --connect rerun+http://127.0.0.1:9876/proxy`
Error: winit EventLoopError: os error at /usr/local/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winit-0.30.12/src/platform_impl/linux/mod.rs:788: Failed to load one of xlib's shared libraries
INFO 2026-03-02 10:41:03 01_leader.py:82 jiao_leader_arm SO101Leader connected.
INFO 2026-03-02 10:41:06 ealsense.py:194 RealSenseCamera(242322077448) connected.
INFO 2026-03-02 10:41:06 follower.py:104 jiao_follower_arm SO101Follower connected.
Teleop loop time: 37.17ms (27 Hz))

            'port': '/dev/ttyACM1',
            'use_degrees': False},
 'teleop_time_s': None}
[2026-03-02T10:40:58Z WARN  re_viewer::native] It looks like you are running the Rerun Viewer inside a Docker container. This is not officially supported, and may lead to performance issues and bugs. See https://github.com/rerun-io/rerun/issues/6835 for more.
[2026-03-02T10:40:58Z INFO  re_grpc_server] Listening for gRPC connections on 0.0.0.0:9876. Connect by running `rerun --connect rerun+http://127.0.0.1:9876/proxy`
Error: winit EventLoopError: os error at /usr/local/cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winit-0.30.12/src/platform_impl/linux/mod.rs:788: Failed to load one of xlib's shared libraries
INFO 2026-03-02 10:41:03 01_leader.py:82 jiao_leader_arm SO101Leader connected.
INFO 2026-03-02 10:41:06 ealsense.py:194 RealSenseCamera(242322077448) connected.
INFO 2026-03-02 10:41:06 follower.py:104 jiao_follower_arm SO101Follower connected.
Teleop loop time: 37.17ms (27 Hz))


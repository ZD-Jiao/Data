lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=jiao_follower_arm \
  --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: '242322077448', width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=jiao_leader_arm \
  --display_data=true

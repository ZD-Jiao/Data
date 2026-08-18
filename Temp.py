#!/usr/bin/env python3
"""Shadow Hand keyboard teaching and playback.

Keys:
    S - record the current right-hand pose
    P - play all recorded poses in order
    C - save the recorded poses to CSV (angles are in degrees)
    Q - quit
"""

import argparse
import csv
import math
import os
import select
import sys
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from sr_robot_commander.sr_hand_commander import SrHandCommander


JOINT_NAMES = [
    "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ1", "rh_MFJ2", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4",
    "rh_LFJ1", "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5",
    "rh_THJ1", "rh_THJ2", "rh_THJ3", "rh_THJ4", "rh_THJ5",
    "rh_WRJ1", "rh_WRJ2",
]


class ShadowHandTeaching:
    def __init__(self, csv_path):
        self.csv_path = os.path.abspath(csv_path)
        self.hand = SrHandCommander(name="right_hand")
        self._state_lock = threading.Lock()
        self._joint_state = {}
        self.poses = []
        self.start_time = time.monotonic()
        self.teach_mode_enabled = False

        self.subscriber = rospy.Subscriber(
            "joint_states", JointState, self._joint_state_callback, queue_size=1
        )

    def _joint_state_callback(self, msg):
        with self._state_lock:
            self._joint_state.update(zip(msg.name, msg.position))

    def wait_for_joint_state(self, timeout=5.0):
        """Wait until one complete right-hand joint state has arrived."""
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._state_lock:
                ready = all(name in self._joint_state for name in JOINT_NAMES)
            if ready:
                return True
            rospy.sleep(0.05)
        return False

    def record_pose(self):
        with self._state_lock:
            missing = [name for name in JOINT_NAMES if name not in self._joint_state]
            if missing:
                print("无法保存：尚未收到关节状态：{}".format(", ".join(missing)))
                return
            joints = {name: float(self._joint_state[name]) for name in JOINT_NAMES}

        self.poses.append({
            "time": time.monotonic() - self.start_time,
            "joints": joints,
        })
        print("已保存姿态 #{:d}".format(len(self.poses)))

    def set_teach_mode(self, enabled):
        """Release the hand for teaching, or restore trajectory control."""
        if enabled == self.teach_mode_enabled:
            return True

        action = "进入示教模式（释放关节）" if enabled else "恢复轨迹控制"
        print("正在{}，请稍候……".format(action))
        try:
            rospy.wait_for_service("/teach_mode", timeout=5.0)
            self.hand.set_teach_mode(enabled)
        except (rospy.ROSException, rospy.ServiceException, Exception) as exc:
            # Some sr_robot_commander versions wrap the service exception in a
            # package-specific exception, hence the final Exception fallback.
            rospy.logerr("%s失败：%s", action, exc)
            return False

        self.teach_mode_enabled = enabled
        print("已{}。".format(action))
        return True

    def play(self):
        if not self.poses:
            print("没有可回放的姿态，请先按 S 示教。")
            return

        if not self.set_teach_mode(False):
            print("无法恢复轨迹控制，已取消回放。")
            return

        print("开始顺序回放 {} 个姿态……".format(len(self.poses)))
        try:
            for index, pose in enumerate(self.poses, start=1):
                if rospy.is_shutdown():
                    return
                print("  回放姿态 {}/{}".format(index, len(self.poses)))
                self.hand.move_to_joint_value_target(
                    pose["joints"], wait=True, angle_degrees=False
                )
            print("回放完成。")
        except Exception as exc:
            rospy.logerr("姿态 %d 回放失败：%s", index, exc)
        finally:
            if not rospy.is_shutdown():
                self.set_teach_mode(True)

    def save_csv(self):
        if not self.poses:
            print("没有可保存的姿态，请先按 S 示教。")
            return

        output_dir = os.path.dirname(self.csv_path)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        try:
            with open(self.csv_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["time"] + JOINT_NAMES)
                for pose in self.poses:
                    row = ["{:.6f}".format(pose["time"])]
                    row.extend(
                        "{:.6f}".format(math.degrees(pose["joints"][name]))
                        for name in JOINT_NAMES
                    )
                    writer.writerow(row)
        except (OSError, IOError) as exc:
            rospy.logerr("CSV 保存失败：%s", exc)
            return

        print("已保存 {} 个姿态到：{}".format(len(self.poses), self.csv_path))


class KeyboardReader:
    """Read a single key without Enter, restoring the terminal on exit."""

    def __enter__(self):
        if os.name == "nt":
            import msvcrt
            self._msvcrt = msvcrt
            return self

        import termios
        import tty

        if not sys.stdin.isatty():
            raise RuntimeError("标准输入不是终端，无法读取快捷键")
        self._termios = termios
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if os.name != "nt":
            self._termios.tcsetattr(
                self._fd, self._termios.TCSADRAIN, self._old_settings
            )

    def read(self, timeout=0.1):
        if os.name == "nt":
            if self._msvcrt.kbhit():
                return self._msvcrt.getwch()
            time.sleep(timeout)
            return None

        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if readable else None


def parse_args():
    parser = argparse.ArgumentParser(description="Shadow Hand 键盘示教程序")
    parser.add_argument(
        "--csv",
        default="shadow_hand_teaching.csv",
        help="按 C 时保存的 CSV 路径（默认：shadow_hand_teaching.csv）",
    )
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def main():
    args = parse_args()
    rospy.init_node("shadow_hand_teaching", anonymous=True)
    teacher = ShadowHandTeaching(args.csv)

    print("正在等待 Shadow Hand 关节状态……")
    if not teacher.wait_for_joint_state():
        rospy.logerr("5 秒内未收到完整的右手关节状态，请检查 joint_states。")
        return 1

    if not teacher.set_teach_mode(True):
        rospy.logerr("无法释放关节，示教程序终止。请检查 /teach_mode 服务。")
        return 1

    print("就绪：S 保存姿态 | P 顺序回放 | C 保存 CSV | Q 退出")
    try:
        with KeyboardReader() as keyboard:
            while not rospy.is_shutdown():
                key = keyboard.read()
                if key is None:
                    continue
                key = key.lower()
                if key == "s":
                    teacher.record_pose()
                elif key == "p":
                    teacher.play()
                elif key == "c":
                    teacher.save_csv()
                elif key == "q":
                    print("退出示教程序。")
                    break
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    except RuntimeError as exc:
        rospy.logerr("键盘读取失败：%s", exc)
        return 1
    finally:
        if teacher.teach_mode_enabled and not rospy.is_shutdown():
            teacher.set_teach_mode(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

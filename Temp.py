#!/usr/bin/env python3
import rospy
import moveit_commander
import time

def main():
    # Initialize ROS + MoveIt
    moveit_commander.roscpp_initialize([])
    rospy.init_node("shadow_hand_finger_test")

    hand_groups = {}
    finger_groups = [
        "rh_little_finger",
        "rh_thumb",
    ]

    print("=== Initializing finger MoveGroups ===")
    for g in finger_groups:
        try:
            hand_groups[g] = moveit_commander.MoveGroupCommander(g)
            print(f"Loaded MoveGroup: {g}")
        except Exception as e:
            print(f"[WARN] Could not init MoveGroup {g}: {e}")

    # -------------------------------------------------------------
    #  TEST MOTION: OPEN → CLOSE → OPEN
    # -------------------------------------------------------------
    rate = rospy.Rate(0.2)  # once every 5 seconds

    for finger in hand_groups:
        group = hand_groups[finger]

        print(f"\n==== Testing {finger} ====")

        # 1. Read joint names
        joint_names = group.get_active_joints()
        print(f"Joints: {joint_names}")

        # 2. Read joint limits
        lower = []
        upper = []
        for j in joint_names:
            info = group._g.get_joint(j).limits
            lower.append(info.lower)
            upper.append(info.upper)

        print(f"Lower limits: {lower}")
        print(f"Upper limits: {upper}")

        # ---------------------------------------------
        # MOVE SEQUENCE
        # ---------------------------------------------

        # Step 1: Move to open hand pose (middle of range)
        mid = [(l + u) * 0.5 for l, u in zip(lower, upper)]
        print(" → Moving to mid pose")
        group.set_joint_value_target(mid)
        group.go(wait=True)
        time.sleep(1)

        # Step 2: Close finger (move to upper limit * 0.7)
        close = [u * 0.7 for u in upper]
        print(" → Closing finger")
        group.set_joint_value_target(close)
        group.go(wait=True)
        time.sleep(1)

        # Step 3: Open (lower limit * 0.7)
        open_pose = [l * 0.7 for l in lower]
        print(" → Opening finger")
        group.set_joint_value_target(open_pose)
        group.go(wait=True)
        time.sleep(1)

    print("\n=== Test Finished ===")
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
policy_deployer_node.py (updated for IsaacLab 2025 ObsNormalizer)

Key changes:
- Extract obs normalizer (actor_obs_normalizer.*) from checkpoint model_state_dict if present
- Use those mean/std to normalize observations before feeding the policy
- Still supports loading a scripted model or a state_dict-based actor
"""

import rospy
import numpy as np
import torch
import threading
import time
from sensor_msgs.msg import JointState
import moveit_commander
from sr_robot_commander.sr_hand_commander import SrHandCommander
import os

# ---------- USER CONFIG ----------
POLICY_PATH = "/path/to/model_1999.pt"   # trained checkpoint (scripted or checkpoint with model_state_dict)

# POLICY_JOINT_ORDER = [
    # "rh_LFJ1", "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5",
    # "rh_THJ1", "rh_THJ2", "rh_THJ3", "rh_THJ4", "rh_THJ5",
# ]
# IsaacLab中的关节索引顺序
POLICY_JOINT_ORDER = [
    "robot0_LFJ4",
    "robot0_THJ4",
    "robot0_LFJ3",
    "robot0_THJ3",
    "robot0_LFJ2",
    "robot0_THJ2",
    "robot0_LFJ1",
    "robot0_THJ1",
    "robot0_LFJ0",
    "robot0_THJ0",
]


JOINT_SIGN_MAP = np.ones(len(POLICY_JOINT_ORDER), dtype=np.float32)

# Real joint limits in radians
JOINT_LOWER = np.array([
    0,  # robot0_LFJ0
    0,  # robot0_LFJ1
    -0.262,  # robot0_LFJ2
    -0.349,  # robot0_LFJ3
    0,  # robot0_LFJ4 (小指末节，活动更小)

    -0.262,  # robot0_THJ0
    -0.698,  # robot0_THJ1
    -0.209,  # robot0_THJ2
    0,  # robot0_THJ3
    -1.047,  # robot0_THJ4
], dtype=np.float32)


JOINT_UPPER = np.array([
     1.571,  # robot0_LFJ0
     1.571,  # robot0_LFJ1
     1.571,  # robot0_LFJ2
     0.349,  # robot0_LFJ3
     0.785,  # robot0_LFJ4

     1.571,  # robot0_THJ0
     0.698,  # robot0_THJ1
     0.209,  # robot0_THJ2
     1.222,  # robot0_THJ3
     1.047,  # robot0_THJ4
], dtype=np.float32)


CONTROL_RATE = 20.0  # Hz
MAX_JOINT_STEP = 0.05  # radians per control step (safety)
USE_PREV_ACTION = True
# ----------------------------------

# 把 actor 变成一个「标准的、可单独推理」的模型
class PolicyWrapper(torch.nn.Module):
    def __init__(self, actor_net):
        super().__init__()
        self.actor = actor_net

    def forward(self, obs: torch.Tensor):
        return self.actor(obs)


class DeployerNode:
    def __init__(self):
        rospy.init_node("policy_deployer_node", anonymous=True)
        self.lock = threading.Lock()

        # MoveIt init
        moveit_commander.roscpp_initialize([])
        self.hand_groups = {}
        finger_groups = ["rh_first_finger", "rh_middle_finger", "rh_ring_finger", "rh_little_finger", "rh_thumb"]
        for g in finger_groups:
            try:
                self.hand_groups[g] = moveit_commander.MoveGroupCommander(g)
            except Exception as e:
                rospy.logwarn(f"Could not init MoveGroup {g}: {e}")

        # SrHandCommander for motion commands
        self.hand_commander = SrHandCommander(name="right_hand")

        # Joint state
        self.latest_joint_state = None
        rospy.Subscriber("joint_states", JointState, self.joint_state_cb, queue_size=1)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Attempt to extract obs normalizer from checkpoint model_state_dict (IsaacLab 2025 style)
        self.obs_mean = None
        self.obs_std = None
        ckpt_based_norm = self.try_extract_normalizer_from_checkpoint(POLICY_PATH)
        if ckpt_based_norm is not None:
            mean_np, std_np = ckpt_based_norm
            self.obs_mean = torch.tensor(mean_np, dtype=torch.float32, device=self.device)
            self.obs_std = torch.tensor(std_np, dtype=torch.float32, device=self.device)
            rospy.loginfo("Loaded obs normalizer from checkpoint model_state_dict.")
        
        # previous action
        self.prev_action = np.zeros(len(POLICY_JOINT_ORDER), dtype=np.float32)

        # load policy (scripted or state_dict)
        self.policy = self.load_policy(POLICY_PATH)
        self.policy.to(self.device)
        self.policy.eval()

        # start control loop
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / CONTROL_RATE), self.control_loop)

    # ---------- Normalizer extraction ----------
    def try_extract_normalizer_from_checkpoint(self, ckpt_path):
        """
        Try to extract actor obs normalizer mean/std from a checkpoint's model_state_dict.
        Returns (mean_np, std_np) or None.
        We search for keys that contain 'actor_obs_normalizer' or 'actor_obs_normalizer._mean', etc.
        """
        if not os.path.exists(ckpt_path):
            rospy.logwarn(f"Checkpoint not found at {ckpt_path}")
            return None
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            rospy.logwarn(f"Failed to load checkpoint {ckpt_path}: {e}")
            return None

        # model_state_dict might be nested under 'model_state_dict'
        state_dict = None
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # maybe ckpt itself is a state_dict
            state_dict = ckpt
        else:
            # some checkpoint layouts: ckpt['model'] or ckpt['model_state_dict']
            if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
                state_dict = ckpt["model"]
            else:
                rospy.logwarn("Checkpoint doesn't look like a state_dict container.")
                state_dict = None

        if state_dict is None:
            return None

        # find mean & var keys for actor obs normalizer
        mean_key = None
        var_key = None
        std_key = None
        count_key = None

        for k in state_dict.keys():
            lk = k.lower()
            # match typical naming patterns produced by IsaacLab2025
            if "actor_obs_normalizer" in lk and ("_mean" in lk or ".mean" in lk or "._mean" in lk):
                mean_key = k
            if "actor_obs_normalizer" in lk and ("_var" in lk or ".var" in lk or "._var" in lk):
                var_key = k
            if "actor_obs_normalizer" in lk and ("_std" in lk or ".std" in lk or "._std" in lk):
                std_key = k
            if "actor_obs_normalizer" in lk and ("count" in lk):
                count_key = k

        # If mean & var not found, try more generic 'actor.obs_norm' style:
        if mean_key is None:
            for k in state_dict.keys():
                if "actor" in k.lower() and "mean" in k.lower() and ("obs" in k.lower() or "norm" in k.lower()):
                    mean_key = k
                if "actor" in k.lower() and ("var" in k.lower() or "variance" in k.lower()) and ("obs" in k.lower() or "norm" in k.lower()):
                    var_key = k

        if mean_key is None:
            # nothing found
            return None

        # extract mean & var/std
        try:
            mean_t = state_dict[mean_key].cpu().numpy()
        except Exception:
            rospy.logwarn(f"Failed to read mean at key {mean_key}")
            return None

        # prefer std if available, else compute from var
        if std_key and std_key in state_dict:
            std_t = state_dict[std_key].cpu().numpy()
        elif var_key and var_key in state_dict:
            var_t = state_dict[var_key].cpu().numpy()
            std_t = np.sqrt(var_t + 1e-8)
        else:
            rospy.logwarn("Var or std key not found for actor obs normalizer; cannot form std.")
            return None

        # final shape sanity check
        if mean_t.shape != std_t.shape:
            rospy.logwarn(f"Mean/std shape mismatch: {mean_t.shape} vs {std_t.shape}")
            # try to flatten or align (not ideal)
            min_len = min(mean_t.size, std_t.size)
            mean_t = mean_t.reshape(-1)[:min_len]
            std_t = std_t.reshape(-1)[:min_len]

        return mean_t.astype(np.float32), std_t.astype(np.float32)

    # ---------- model loading ----------
    def load_policy(self, path):
        # 1) try load scripted module
        try:
            scripted = torch.jit.load(path, map_location=self.device)
            rospy.loginfo(f"Loaded scripted policy from {path}")
            return scripted
        except Exception:
            rospy.loginfo("Not a scripted module or torch.jit.load failed, trying state_dict load...")

        # 2) try load checkpoint and build an actor MLP then load actor weights if possible
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt if isinstance(ckpt, dict) else None)

        # If the checkpoint is just a state_dict for the actor (common), try to load directly
        from torch import nn
        obs_dim = self.estimate_obs_dim()
        act_dim = len(POLICY_JOINT_ORDER)
        actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )
        # Attempt 1: if ckpt is directly actor state_dict
        try:
            if isinstance(ckpt, dict) and "state_dict" not in ckpt and ("model_state_dict" not in ckpt):
                # might be raw actor state_dict
                actor.load_state_dict(ckpt)
                rospy.loginfo("Loaded actor state_dict directly from checkpoint.")
                return PolicyWrapper(actor)
        except Exception:
            pass

        # Attempt 2: if model_state_dict exists, try to find actor.* keys and load subset
        if state_dict:
            # filter keys that belong to actor (heuristic)
            actor_keys = {k: v for k, v in state_dict.items() if k.lower().startswith("actor") or "actor" in k.lower()}
            if len(actor_keys) > 0:
                # try to remap common patterns: e.g. "actor.0.weight" -> "0.weight"
                remapped = {}
                for k, v in actor_keys.items():
                    # remove leading "actor." or "policy.actor." etc
                    newk = k
                    if ".actor." in k:
                        newk = k.split(".actor.", 1)[1]
                    elif k.startswith("actor."):
                        newk = k[len("actor.") :]
                    elif "actor" in k and k.startswith("policy."):
                        # policy.actor.* -> remove prefix
                        newk = k.split("actor.", 1)[1]
                    else:
                        # fallback: try keep as-is (may not match)
                        newk = k
                    remapped[newk] = v
                # try to load remapped keys into actor
                try:
                    actor.load_state_dict(remapped, strict=False)
                    rospy.loginfo("Loaded partial actor weights from model_state_dict (non-strict).")
                    return PolicyWrapper(actor)
                except Exception as e:
                    rospy.logwarn(f"Failed to load partial actor weights (non-strict): {e}")

        # Last resort: if checkpoint contains a top-level 'model_state_dict' but no actor keys, try direct sd
        try:
            if isinstance(state_dict, dict):
                # attempt to load whole state_dict into the actor (will usually fail if keys mismatch)
                actor.load_state_dict(state_dict, strict=False)
                rospy.loginfo("Loaded model_state_dict into actor with strict=False (best-effort).")
                return PolicyWrapper(actor)
        except Exception as e:
            rospy.logerr(f"Failed to load model_state_dict into actor: {e}")

        raise RuntimeError("Unable to load policy. Provide a scripted model or an actor state_dict matching the MLP architecture.")

    # ---------- helper functions ----------
    def estimate_obs_dim(self):
        dof = len(POLICY_JOINT_ORDER)
        fingertip_count = 5
        obs_dim = dof + dof + fingertip_count*3 + fingertip_count*6 + fingertip_count*3 + dof
        return obs_dim

    def joint_state_cb(self, msg: JointState):
        with self.lock:
            self.latest_joint_state = msg

    def get_fingertip_poses(self):
        poses = []
        for g_name, group in self.hand_groups.items():
            try:
                ee_link = group.get_end_effector_link()
                p = group.get_current_pose(ee_link).pose
                poses.append([p.position.x, p.position.y, p.position.z])
            except Exception as e:
                rospy.logwarn(f"FK failed for {g_name}: {e}")
                poses.append([0.0,0.0,0.0])
        return poses

    # 根据当前关节状态，计算每个fingertip的6维 twist（线速度 + 角速度）
    def get_fingertip_twists(self, q_list):
        twists = []
        for g_name, group in self.hand_groups.items():
            try:
                active_joints = group.get_active_joints()
                q = [0.0]*len(active_joints)
                qd = [0.0]*len(active_joints)
                if self.latest_joint_state is None:
                    twists.append(np.zeros(6))
                    continue
                for i,jn in enumerate(active_joints):
                    if jn in self.latest_joint_state.name:
                        idx = self.latest_joint_state.name.index(jn)
                        q[i] = float(self.latest_joint_state.position[idx]) if idx < len(self.latest_joint_state.position) else 0.0
                        qd[i] = float(self.latest_joint_state.velocity[idx]) if idx < len(self.latest_joint_state.velocity) else 0.0
                J = np.array(group.get_jacobian_matrix(q), dtype=np.float64)
                twist = J.dot(np.array(qd, dtype=np.float64))
                twists.append(twist)
            except Exception as e:
                rospy.logwarn(f"Jacobian failed for {g_name}: {e}")
                twists.append(np.zeros(6))
        return twists

    # ROS->Isaac coordinate transform (use your earlier formula)
    @staticmethod
    def ros_to_isaac_np(pos_ros: np.ndarray) -> np.ndarray:
        pos_ros = np.asarray(pos_ros)
        x, y, z = pos_ros[..., 0], pos_ros[..., 1], pos_ros[..., 2]
        x_i = -x
        y_i = -z
        z_i = -y
        return np.stack([x_i, y_i, z_i], axis=-1)

    def build_observation(self):
        with self.lock:
            js = self.latest_joint_state
        if js is None:
            return None

        q = np.zeros(len(POLICY_JOINT_ORDER), dtype=np.float32)
        qd = np.zeros_like(q)
        for i, jn in enumerate(POLICY_JOINT_ORDER):
            if jn in js.name:
                idx = js.name.index(jn)
                q[i] = float(js.position[idx]) if idx < len(js.position) else 0.0
                qd[i] = float(js.velocity[idx]) if idx < len(js.velocity) else 0.0

        fingertip_ros = self.get_fingertip_poses()
        fingertip_twists = self.get_fingertip_twists(q.tolist())

        fingertip_ros_np = np.array(fingertip_ros, dtype=np.float32)
        fingertip_isaac = self.ros_to_isaac_np(fingertip_ros_np)
        fingertip_twists_np = np.array([t[:3] for t in fingertip_twists], dtype=np.float32)

        # goal positions: replace with actual source if any
        goal_positions = fingertip_isaac.copy()

        prev_actions = self.prev_action.copy()

        # unscale q -> [-1,1] as in training code (unscale = 2*(q - lower)/(upper-lower)-1)
        q_unscaled = 2.0 * (q - JOINT_LOWER) / (JOINT_UPPER - JOINT_LOWER) - 1.0
        qd_scaled = 0.1 * qd
        fingertip_flat = fingertip_isaac.reshape(-1)
        fingertip_vel_flat = 0.1 * fingertip_twists_np[:, :3].reshape(-1)
        goal_flat = goal_positions.reshape(-1)

        obs = np.concatenate([q_unscaled, qd_scaled, fingertip_flat, fingertip_vel_flat, goal_flat, prev_actions], axis=0)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # If we extracted obs normalizer from the checkpoint, use it:
        if (self.obs_mean is not None) and (self.obs_std is not None):
            # sanity: shapes must match
            if self.obs_mean.numel() != obs_t.numel():
                rospy.logwarn(f"Normalizer dim ({self.obs_mean.numel()}) != obs dim ({obs_t.numel()}). Skipping external normalization. Model may handle normalization internally.")
            else:
                obs_t = (obs_t - self.obs_mean) / (self.obs_std + 1e-8)
                obs_t = torch.clamp(obs_t, -10.0, 10.0)

        return obs_t

    # 把 PPO 动作（通常在 [-1,1]）映射到真实关节角范围; PD / 平滑 / 限幅策略
    def map_action_to_ros(self, action: np.ndarray, current_q: np.ndarray):
        action_signed = JOINT_SIGN_MAP * action
        target = 0.5*(action_signed + 1.0)*(JOINT_UPPER - JOINT_LOWER) + JOINT_LOWER
        delta = target - current_q
        clipped_delta = np.clip(delta, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        safe_target = current_q + clipped_delta
        return safe_target

    def control_loop(self, event):
        obs_t = self.build_observation()    # 收集当前状态
        if obs_t is None:
            return
        with torch.no_grad():
            action_t = self.policy(obs_t.to(self.device))   # 输入 obs → 输出 action
            if isinstance(action_t, tuple):
                action = action_t[0].cpu().numpy().squeeze()
            else:
                action = action_t.cpu().numpy().squeeze()
        if action.ndim == 0:    # 保证 action 是 1D 数组
            action = np.array([action], dtype=np.float32)

        with self.lock: # 使用锁保证线程安全
            js = self.latest_joint_state
        if js is None:
            return
        # ROS 的 joint_state 顺序 → 映射到 PPO 训练时的 joint order
        current_q = np.zeros(len(POLICY_JOINT_ORDER), dtype=np.float32)
        for i, jn in enumerate(POLICY_JOINT_ORDER):
            if jn in js.name:
                idx = js.name.index(jn)
                current_q[i] = float(js.position[idx]) if idx < len(js.position) else 0.0

        target_q = self.map_action_to_ros(action, current_q)

        joints_dict = {jn: float(target_q[i]) for i, jn in enumerate(POLICY_JOINT_ORDER)}
        try:
            self.hand_commander.plan_to_joint_value_target(joints_dict, angle_degrees=False)
            self.hand_commander.move_to_joint_value_target(joints_dict, wait=False, angle_degrees=False)
        except Exception as e:
            rospy.logerr(f"Failed to send hand command: {e}")

        self.prev_action = action.copy()

    def shutdown(self):
        rospy.loginfo("Shutting down deployer")
        self.control_timer.shutdown()
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    try:
        node = DeployerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


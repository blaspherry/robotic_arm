import matplotlib
matplotlib.use("TkAgg")

import numpy as np
from src.robotic_arm import RoboticArm
from src.visualizer import plot_arm

if __name__ == "__main__":
    arm = RoboticArm(6)
    arm.set_joint_angles([0.3, -0.6, 0.8, 0.2, -0.4, 0.5])
    T, joint_positions = arm.forward_kinematics()
    print("End effector pos:", T[:3, 3])
    plot_arm(joint_positions, title="FK Visualization")
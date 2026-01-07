import time
import numpy as np
from src.robotic_arm import RoboticArm


def benchmark_forward_kinematics(num_tests: int = 10000):
    arm = RoboticArm(num_joints=6)

    test_angles = np.random.uniform(-np.pi, np.pi, (num_tests, 6))

    start = time.time()
    for angles in test_angles:
        arm.forward_kinematics(angles)
    elapsed = time.time() - start

    ops_per_sec = num_tests / max(elapsed, 1e-9)

    print("正向运动学性能测试：")
    print(f"  测试次数: {num_tests}")
    print(f"  总耗时: {elapsed:.4f} sec")
    print(f"  性能: {ops_per_sec:.1f} ops/sec")
    print(f"  目标: >= 10,000 ops/sec")
    print (f" 达标: { ' ✓' if ops_per_sec >= 10000 else ' X '}")
    return ops_per_sec

def test_accuracy():
    arm = RoboticArm(num_joints=6)
    test_cases = [
        {
            'angles': [0, 0, 0, 0, 0, 0],
            'expected_pos': [ 0, 0, 0.95],
            'tolerance': 0.001
        }
    ]

    print("\n精度测试：")
    for i, case in enumerate(test_cases):
        arm.set_joint_angles(case['angles'])
        actual_pos = arm.get_end_effector_position()
        expected_pos = np.array(case['expected_pos'])
        error= np.linalg.norm(actual_pos - expected_pos)
        print(f"测试{i+1}: ")
        print(f"期望位置：{expected_pos}")
        print(f"实际位置：{actual_pos}")
        print(f"误差：{error*1000:.2f}mm")
        print(f"达标：{' ✓' if error < case['tolerance'] else 'X'}")


if __name__ == "__main__":
    benchmark_forward_kinematics()
    test_accuracy()
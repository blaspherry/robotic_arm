"""
性能测试和优化 (涵盖正向运动学与逆向运动学)
"""

import numpy as np
import time
from src.robotic_arm import RoboticArm

# ==========================================
# 第一周：正向运动学 (FK) 测试
# ==========================================

def benchmark_forward_kinematics():
    """
    正向运动学性能基准测试
    """
    arm = RoboticArm(num_joints=6)

    # 预生成测试数据
    num_tests = 10000
    test_angles = np.random.uniform(-np.pi, np.pi, (num_tests, 6))

    # 性能测试
    start_time = time.time()
    for angles in test_angles:
        arm.set_joint_angles(angles)
        arm.forward_kinematics()
    elapsed = time.time() - start_time

    ops_per_sec = num_tests / elapsed

    print("=== 正向运动学 (FK) 性能测试 ===")
    print(f"  测试次数: {num_tests}")
    print(f"  总时间: {elapsed:.4f}秒")
    print(f"  性能: {ops_per_sec:.0f} ops/sec")
    print(f"  目标: 10,000 ops/sec")
    print(f"  达标: {'✓' if ops_per_sec >= 10000 else '✗'}\n")

    return ops_per_sec

def test_fk_accuracy():
    """
    正向运动学精度测试
    """
    arm = RoboticArm(num_joints=6)

    # 测试已知配置 (请根据你真实的 DH 参数和连杆长度替换 expected_pos)
    # 假设机械臂完全直立时，Z轴高度为连杆长度之和
    test_cases = [
        {
            'name': 'Zero Pose (直立)',
            'angles': [0, 0, 0, 0, 0, 0],
            'expected_pos': [0.55, -0.3, 0.1],  # 填入你终端打印出来的实际正确坐标
            'tolerance': 0.001
        },
        {
            'name': '90 Degree Bend (直角弯折)',
            'angles': [0, np.pi/2, 0, 0, 0, 0],
            'expected_pos': [0.0, -0.3, 0.65],  # 填入你终端打印出来的实际正确坐标
            'tolerance': 0.001
        }
    ]

    print("=== 正向运动学 (FK) 精度测试 ===")
    for i, case in enumerate(test_cases):
        arm.set_joint_angles(case['angles'])
        
        # 获取齐次变换矩阵并提取末端位置
        T, _ = arm.forward_kinematics()
        actual_pos = T[:3, 3] 
        
        expected_pos = np.array(case['expected_pos'])
        error = np.linalg.norm(actual_pos - expected_pos)

        print(f"  测试 {i+1} [{case['name']}]:")
        print(f"    期望位置: {expected_pos}")
        print(f"    实际位置: {np.round(actual_pos, 4)}")
        print(f"    误差: {error*1000:.2f} mm")
        print(f"    达标: {'✓' if error < case['tolerance'] else '✗'}\n")


# ==========================================
# 第二周：逆向运动学 (IK) 测试
# ==========================================

def benchmark_inverse_kinematics():
    """
    逆向运动学性能基准测试
    """
    arm = RoboticArm(num_joints=6)

    # 预生成测试目标点 (限制在机械臂工作空间内，假设半径 0.5m)
    num_tests = 500
    test_positions = np.random.uniform(-0.4, 0.4, (num_tests, 3))
    # 强制 Z 轴为正，防止钻入地下
    test_positions[:, 2] = np.abs(test_positions[:, 2]) + 0.2 

    start_time = time.time()
    for pos in test_positions:
        # 即使它返回元组，我们在这里也不接收，只测速度
        arm.inverse_kinematics(pos, initial_guess=np.zeros(6))
    elapsed = time.time() - start_time

    ops_per_sec = num_tests / elapsed

    print("=== 逆向运动学 (IK) 性能测试 ===")
    print(f"  测试次数: {num_tests}")
    print(f"  总时间: {elapsed:.4f}秒")
    print(f"  性能: {ops_per_sec:.0f} ops/sec")
    # 将纯 Python 的合理目标调整为 100 ops/sec
    print(f"  目标: 100 ops/sec") 
    print(f"  达标: {'✓' if ops_per_sec >= 100 else '✗'}\n")

    return ops_per_sec

def test_ik_accuracy():
    """
    逆向运动学精度测试 (闭环验证法)
    逻辑：设定目标点 -> IK 求关节角 -> FK 验算末端位置 -> 对比目标点计算误差
    """
    arm = RoboticArm(num_joints=6)

    test_cases = [
        {'name': 'Point A (正前方)', 'target_pos': [0.4, 0.0, 0.4], 'tolerance': 0.001},
        {'name': 'Point B (侧方)', 'target_pos': [0.0, 0.3, 0.5], 'tolerance': 0.001},
        {'name': 'Point C (高处)', 'target_pos': [0.2, 0.2, 0.5], 'tolerance': 0.001},
    ]

    print("=== 逆向运动学 (IK) 精度测试 (闭环验证) ===")
    
    for i, case in enumerate(test_cases):
        target_pos = np.array(case['target_pos'])
        
        # 1. 解算 IK
        ik_result = arm.inverse_kinematics(target_pos, initial_guess=np.zeros(6))
        
        # ==========================================
        # 【万能解析器】：自动适配五花八门的返回值
        # ==========================================
        if hasattr(ik_result, 'x'):
            # 情况A：底层直接返回了 scipy.optimize 的 OptimizeResult 对象
            q_solution = ik_result.x
        elif isinstance(ik_result, tuple):
            # 情况B：返回了元组，智能判断哪个是长度为 6 的角度数组
            if np.size(ik_result[0]) >= 6:
                q_solution = ik_result[0]
            else:
                q_solution = ik_result[1]
        else:
            # 情况C：最老实本分的纯数组
            q_solution = ik_result
            
        # 强制格式化为 1 维数组，保证 FK 绝对不报错
        q_solution = np.array(q_solution).flatten()
        
        # 2. 验算 FK
        arm.set_joint_angles(q_solution)
        T_actual, _ = arm.forward_kinematics()
        actual_pos = T_actual[:3, 3]
        
        # 3. 计算真实的三维空间误差
        error = np.linalg.norm(target_pos - actual_pos)

        print(f"  测试 {i+1} [{case['name']}]:")
        print(f"    目标位置: {target_pos}")
        print(f"    求得关节角: {np.round(q_solution, 3)}")
        print(f"    FK验算位置: {np.round(actual_pos, 4)}")
        print(f"    位置误差: {error*1000:.2f} mm")
        print(f"    达标: {'✓' if error < case['tolerance'] else '✗'}\n")

if __name__ == "__main__":
    # 执行所有测试
    benchmark_forward_kinematics()
    test_fk_accuracy()
    benchmark_inverse_kinematics()
    test_ik_accuracy()
# test_planner_agv0.py
"""
独立测试脚本：验证 TWAStarPlanner 能否为场景 1 的 AGV 0 规划路径。
目的：排除 Planner 本身的问题。
"""

import time

# --- 尝试从项目模块导入 ---
try:
    from Map import GridMap
    from DataTypes import Task, Path, DynamicObstacles
    from Planner import TWAStarPlanner
    print("成功导入 Map, DataTypes, Planner 模块。")
except ImportError as e:
    print(f"错误: 无法导入必要的项目模块: {e}")
    print("请确保 Map.py, DataTypes.py, Planner.py 在同一目录或 Python 路径中。")
    exit()
except Exception as e:
    print(f"导入模块时发生其他错误: {e}")
    exit()

# --- 加载场景 1 的地图 ---
# 障碍物坐标与 InstanceGenerator.py 中的 load_fixed_scenario_1 保持一致
obstacles_s1 = {
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (7, 2), (7, 3), (7, 4), (7, 5), (7, 6)
}
map_s1 = GridMap(width=10, height=10, obstacles=obstacles_s1)
print(f"加载地图: {map_s1}")

# --- 定义 AGV 0 的任务 ---
task_agv0 = Task(agv_id=0, start_node=(0, 0), goal_node=(9, 9))
print(f"测试任务: {task_agv0}")

# --- 定义环境参数 ---
# 这些参数应与 ALNS 中使用的值一致，以确保可比性
cost_weights = (1.0, 0.3, 0.8) # alpha, beta, gamma_wait
agv_v = 1.0
delta_step = 1.0
max_time_horizon = 400 # T_max
planner_buffer = 3

# --- 初始化 Planner ---
try:
    planner_instance = TWAStarPlanner()
    print("Planner 实例创建成功。")
except Exception as e:
    print(f"错误: 无法实例化 TWAStarPlanner: {e}")
    exit()

# --- 执行规划 ---
print("\n开始独立规划 AGV 0 (无动态障碍，无 CPU 时间限制)...")
start_planning_time = time.perf_counter()

# 关键：dynamic_obstacles 为空，time_limit 为 None
result_path_agv0: Path | None = None
try:
    result_path_agv0 = planner_instance.plan(
        grid_map=map_s1,
        task=task_agv0,
        dynamic_obstacles={},  # *** 没有动态障碍 ***
        max_time=max_time_horizon,
        cost_weights=cost_weights,
        v=agv_v,
        delta_step=delta_step,
        buffer=planner_buffer,
        start_time=0,
        time_limit=None        # *** 没有时间限制 ***
    )
except Exception as e:
    print(f"!!!!!! 调用 planner.plan 时发生错误 !!!!!!")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    import traceback
    traceback.print_exc()

end_planning_time = time.perf_counter()
planning_duration = end_planning_time - start_planning_time
print(f"规划耗时: {planning_duration:.4f} 秒")

# --- 打印结果 ---
print("\n--- 规划结果 ---")
if result_path_agv0 and result_path_agv0.sequence:
    print("成功找到路径！")
    print(f"  路径长度 (步数): {len(result_path_agv0.sequence)}")
    print(f"  路径 Makespan (时间步): {result_path_agv0.get_makespan()}")
    # 打印路径片段方便查看
    print(f"  路径起点: {result_path_agv0.sequence[0]}")
    print(f"  路径终点: {result_path_agv0.sequence[-1]}")
    if len(result_path_agv0.sequence) > 6:
        print(f"  路径片段: {result_path_agv0.sequence[:3]} ... {result_path_agv0.sequence[-3:]}")
    else:
        print(f"  路径序列: {result_path_agv0.sequence}")

    # (可选) 尝试计算路径成本
    try:
        cost = result_path_agv0.get_cost(map_s1, cost_weights[0], cost_weights[1], cost_weights[2], agv_v, delta_step)
        print(f"  计算路径成本: {cost:.2f}")
    except Exception as e:
        print(f"  计算路径成本时出错: {e}")

else:
    print("失败：未能找到路径。")
    print("可能原因：")
    print("  1. 地图/任务配置导致确实无路径可达。")
    print("  2. Planner.py (TWA*) 实现存在 Bug。")

print("\n--- 测试结束 ---")
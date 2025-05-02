# Benchmark.py-v14 (支持 AGV 特定速度和优先级策略排序)
"""
实现基准算法：优先顺序规划器 (Prioritized Sequential Planner)。
该算法按固定优先级（基于 AGV ID 或速度）依次规划每个 AGV 的路径，
并将已规划的路径视为后续 AGV 的动态障碍。
用作与 ALNS 算法进行性能对比的基准。

与论文的关联:
- 对比基准: 作为论文实验部分用于对比 ALNS 性能的简单但常用的基准方法。
- 核心规划器: 内部依赖 TWA* (Planner.py v24+) 来规划单条路径。
- 目标函数评估: 通过调用 Path.get_cost (v11+) 或累加单路径成本来计算总成本。
                 **v14 修改**: 现在使用特定 AGV 速度计算成本。
- 约束满足: 通过序贯规划和动态障碍避免节点冲突。
- **v14 修改**: 支持基于 ID 或速度的静态优先级排序。
- **v14 修改**: 在规划和成本计算中使用 AGV 特定速度。

版本变更 (v13 -> v14):
- **新增**: `__init__` 增加 `agv_speeds` 和 `priority_strategy` 参数。
- **修改**: `__init__` 根据 `priority_strategy` 对任务列表 `self.tasks` 进行排序。
- **修改**: `plan` 方法现在查找并传递特定 AGV 速度给 `planner.plan`。
- **修改**: `plan` 方法在计算总成本时（通过调用内部 `_calculate_total_cost`），确保传递特定 AGV 速度。
- **修改**: 新增内部辅助方法 `_calculate_total_cost` 来统一处理成本计算，并接收 `agv_speeds`。
- **修改**: `if __name__ == '__main__':` 更新以演示新参数的使用。
- **依赖**: Map.py(v9+), DataTypes.py(v11+), Planner.py(v24+), InstanceGenerator.py(v15+)。
"""
import time
import random
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
import traceback
import sys # 用于退出

# --- 类型提示导入 ---
if TYPE_CHECKING:
    from Map import GridMap, Node # 依赖 v9+
    from DataTypes import Task, Path, TimeStep, State, DynamicObstacles, Solution, CostDict # 依赖 v11+
    from Planner import TWAStarPlanner # 依赖 v24+
# --- 从项目模块导入 ---
try:
    from Map import GridMap, Node # 依赖 v9+
    from DataTypes import Task, Path, TimeStep, State, DynamicObstacles, Solution, CostDict, calculate_tij # 依赖 v11+
    from Planner import TWAStarPlanner # 依赖 v24+
except ImportError as e:
    print(f"错误: 导入 Benchmark 依赖项失败: {e}")
    GridMap = type('GridMap', (object,), {})
    Task = type('Task', (object,), {})
    Path = type('Path', (object,), {})
    Solution = Dict; CostDict = Dict; DynamicObstacles = Dict
    TWAStarPlanner = type('TWAStarPlanner', (object,), {})
    Node = Tuple; TimeStep = int
    sys.exit(1)

# --- 优先顺序规划器类 (v14) ---
class PrioritizedPlanner:
    """实现简单的优先顺序规划基准算法。"""
    def __init__(self,
                 grid_map: 'GridMap',
                 tasks: List['Task'],
                 planner: 'TWAStarPlanner',
                 agv_speeds: Dict[int, float], # <--- 新增: 接收速度字典
                 delta_step: float,
                 priority_strategy: str = 'id'): # <--- 新增: 接收优先级策略
        """
        初始化优先规划器。

        Args:
            grid_map (GridMap): 地图对象。
            tasks (List[Task]): 任务列表。
            planner (TWAStarPlanner): 核心路径规划器实例。
            agv_speeds (Dict[int, float]): AGV 特定速度字典 {agv_id: speed}。
            delta_step (float): 每个时间步的时长 (秒)。
            priority_strategy (str): 任务规划顺序策略 ('id' 或 'speed')。
        """
        # --- 输入验证 ---
        if not isinstance(grid_map, GridMap): raise TypeError("grid_map 必须是 GridMap 类型")
        if not isinstance(tasks, list) or not all(isinstance(t, Task) for t in tasks): raise TypeError("tasks 必须是 Task 列表")
        if not isinstance(planner, TWAStarPlanner): raise TypeError("planner 必须是 TWAStarPlanner 类型")
        if not isinstance(agv_speeds, dict): raise TypeError("agv_speeds 必须是字典类型")
        if not isinstance(delta_step, (int, float)) or delta_step <= 0: raise ValueError("时间步长 delta_step 必须为正数。")
        if not all(isinstance(k, int) and isinstance(v, (int, float)) and v > 0 for k, v in agv_speeds.items()):
            raise ValueError("agv_speeds 字典的键必须是整数，值必须是正数。")

        # --- 基础属性 ---
        self.grid_map = grid_map
        self.planner = planner
        self.agv_speeds = agv_speeds # <--- 存储速度
        self.delta_step = float(delta_step)
        self.num_agvs = len(tasks)
        self.priority_strategy = priority_strategy.lower()
        if self.priority_strategy not in ['id', 'speed']:
            print(f"警告 (Benchmark): 无效的 priority_strategy '{priority_strategy}'，回退到 'id'。")
            self.priority_strategy = 'id'

        # --- 验证所有任务的 AGV 都有速度信息 ---
        missing_speeds = [task.agv_id for task in tasks if task.agv_id not in self.agv_speeds]
        if missing_speeds:
            raise ValueError(f"错误 (Benchmark): 以下 AGV 在 agv_speeds 字典中缺少速度信息: {missing_speeds}")

        # --- 根据优先级策略排序任务 ---
        if self.priority_strategy == 'speed':
            # 按速度降序，速度相同则 ID 升序
            self.tasks = sorted(tasks, key=lambda t: (-self.agv_speeds.get(t.agv_id, 0), t.agv_id))
            print("  Benchmark: 使用基于速度的规划顺序。")
        else: # 默认 ID 优先级
            self.tasks = sorted(tasks, key=lambda t: t.agv_id)
            print("  Benchmark: 使用基于 ID 的规划顺序。")
        # --- ------------------------- ---
        print(f"  Benchmark 初始化完成，规划顺序 ({self.priority_strategy}): {[t.agv_id for t in self.tasks]}")

    def plan(self, cost_weights: Tuple[float, float, float], max_time: TimeStep, time_limit_per_agent: Optional[float] = None) -> Tuple[Optional['Solution'], float, 'CostDict']:
        """
        执行优先顺序规划。

        Args:
            cost_weights (Tuple[float, float, float]): (alpha, beta, gamma_wait) 成本权重。
            max_time (TimeStep): 规划的最大时间步。
            time_limit_per_agent (Optional[float]): 每个 AGV 规划的 CPU 时间限制 (秒)。

        Returns:
            Tuple[Optional[Solution], float, CostDict]: (解决方案, 总耗时, 总成本字典)
        """
        start_plan_time = time.perf_counter()
        solution: Solution = {}
        dynamic_obstacles: DynamicObstacles = {} # 存储已规划路径的时空占用
        inf_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        all_success = True

        print(f"\n--- 开始基准规划 (优先序贯, 策略: {self.priority_strategy}) ---")
        if not self.tasks:
            print("警告: Benchmark 任务列表为空。")
            return None, 0.0, inf_dict

        # 按预先排好的顺序迭代任务
        for task in self.tasks:
            agv_id = task.agv_id
            t_start_call = time.perf_counter()
            # --- 获取特定 AGV 速度 ---
            try:
                agv_speed = self.agv_speeds[agv_id] # __init__ 已验证存在
            except KeyError: # 再次检查以防万一
                 print(f"    严重错误: 无法获取 AGV {agv_id} 的速度！Benchmark 终止。")
                 all_success = False; break
            # --- -------------------- ---
            print(f"  规划 AGV {agv_id} (速度: {agv_speed:.2f})...")

            try:
                # --- 调用修改后的 planner.plan (v24+)，传入特定速度 ---
                path: Optional[Path] = self.planner.plan(
                    self.grid_map, task, dynamic_obstacles, max_time, cost_weights,
                    agv_speed, # <--- 传递特定速度
                    self.delta_step, 0, time_limit_per_agent, None
                )
                # --- ----------------------------------------- ---
                # 验证返回类型
                if path is not None and not isinstance(path, Path):
                    print(f"    错误: Planner 为 AGV {agv_id} 返回了非 Path 类型: {type(path)}。")
                    all_success = False; break
            except Exception as e:
                print(f"    错误: 调用 Planner 为 AGV {agv_id} 规划时发生异常: {e}")
                traceback.print_exc()
                all_success = False; break
            call_dur = time.perf_counter() - t_start_call

            if path and path.sequence: # 处理成功结果
                solution[agv_id] = path
                # 更新动态障碍 (只添加节点占用，不考虑边)
                # 这是序贯规划的核心：将已规划路径视为后续 AGV 的障碍
                for node, t in path.sequence:
                    if t not in dynamic_obstacles: dynamic_obstacles[t] = set()
                    dynamic_obstacles[t].add(node)
                # (成本计算移到最后统一处理)
                print(f"    成功，耗时 {call_dur:.4f}s, 路径长度 {len(path)}, Makespan {path.get_makespan()}")
            else: # 处理失败结果
                print(f"  错误：AGV {agv_id} 规划失败！(耗时 {call_dur:.4f}s) Benchmark 终止。")
                all_success = False
                break # 只要有一个失败，整个序贯规划就失败

        total_duration = time.perf_counter() - start_plan_time
        print(f"--- 基准规划完成，总耗时: {total_duration:.4f}s ---")

        if all_success and len(solution) == self.num_agvs:
            # --- 统一计算最终成本 (使用修改后的辅助函数) ---
            final_cost_dict = self._calculate_total_cost(solution, cost_weights)
            if final_cost_dict.get('total', float('inf')) != float('inf'):
                return solution, total_duration, final_cost_dict
            else:
                print("  错误：基准规划成功但最终成本计算为 Inf！")
                return None, total_duration, inf_dict
        else:
            # 返回规划耗时，但解决方案和成本为失败状态
            return None, total_duration, inf_dict

    # --- 新增: 内部辅助函数，用于计算总成本 (v14) ---
    def _calculate_total_cost(self, solution: 'Solution', cost_weights: Tuple[float, float, float]) -> 'CostDict':
        """
        (内部辅助) 计算给定解决方案的总成本，考虑每个 AGV 的特定速度。
        """
        total_cost_dict: CostDict = {'total': 0.0, 'travel': 0.0, 'turn': 0.0, 'wait': 0.0}
        inf_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        if not solution or not isinstance(solution, dict): return inf_dict

        alpha, beta, gamma_wait = cost_weights
        valid_solution = True
        num_paths = 0
        for agv_id, path in solution.items():
            if not isinstance(path, Path) or not path.sequence:
                valid_solution = False; break
            num_paths += 1
            try:
                # --- 获取特定 AGV 速度 ---
                agv_speed = self.agv_speeds.get(agv_id)
                if agv_speed is None: # 再次检查以防万一
                    print(f"错误 (Benchmark cost calc): 无法获取 AGV {agv_id} 的速度。")
                    valid_solution = False; break
                # --- -------------------- ---
                # --- 调用修改后的 path.get_cost (v11+) ---
                cost_dict = path.get_cost(self.grid_map, alpha, beta, gamma_wait, agv_speed, self.delta_step)
                # --- ------------------------------------ ---
            except Exception as e:
                print(f"错误 (Benchmark cost calc): 计算 AGV {agv_id} 成本时出错: {e}")
                valid_solution = False; break

            if not isinstance(cost_dict, dict) or cost_dict.get('total', float('inf')) == float('inf'):
                valid_solution = False; break
            # 累加成本
            for key in total_cost_dict:
                cost_val = cost_dict.get(key, 0.0)
                if not isinstance(cost_val, (int, float)): cost_val = 0.0 # 处理可能的非数字值
                total_cost_dict[key] += cost_val

        # 确保所有 AGV 都计算了成本
        if valid_solution and num_paths == self.num_agvs:
            return total_cost_dict
        else:
            return inf_dict

# --- 示例用法 (v14 - 更新以测试新参数) ---
if __name__ == '__main__':
    print("--- Benchmark (Prioritized Planner) 测试 (v14 - 支持速度和优先级) ---")
    try:
        # 假设 InstanceGenerator v15 可用
        from InstanceGenerator import load_fixed_scenario_1
    except ImportError as e:
        print(f"错误: 导入 InstanceGenerator (v15+) 失败: {e}")
        sys.exit(1)

    print("加载固定算例场景 1 (需要 Map.json)...")
    # 调用 v15 函数，获取地图、任务和速度
    instance_data_s1 = load_fixed_scenario_1(json_map_file="Map.json", expansion_radius=0, assign_speeds=True)
    if not instance_data_s1:
        print("错误: 无法加载场景 1 数据。")
        sys.exit(1)
    test_map_s1, test_tasks_s1, test_speeds_s1 = instance_data_s1 # 解包
    if not test_tasks_s1:
        print("错误: 场景 1 任务列表为空。")
        sys.exit(1)
    print(f"加载了 {len(test_tasks_s1)} 个任务。")
    print(f"使用的 AGV 速度: {test_speeds_s1}")

    # --- 初始化 Planner (v24+) ---
    try:
        planner_instance_bm = TWAStarPlanner()
    except Exception as planner_e:
        print(f"错误: 初始化 TWAStarPlanner 失败: {planner_e}")
        sys.exit(1)
    # --------------------------

    # --- 测试场景 1: ID 优先级 ---
    print("\n[测试 1] 使用 ID 优先级策略运行 Benchmark...")
    try:
        # --- 修改 Benchmark 初始化，传入速度和策略 ---
        benchmark_planner_id = PrioritizedPlanner(
            test_map_s1, test_tasks_s1, planner_instance_bm,
            agv_speeds=test_speeds_s1, # <--- 传递速度
            delta_step=1.0,
            priority_strategy='id' # <--- 设置为 ID 优先级
        )
        # --- ------------------------------------- ---
    except Exception as init_e:
        print(f"错误: 初始化 PrioritizedPlanner (ID Prio) 失败: {init_e}")
        traceback.print_exc()
        sys.exit(1)

    cost_w_test = (1.0, 0.3, 0.8)
    max_t_horizon_test = 600 # 增加时间范围
    time_lim_test = 30.0 # 每个 agent 的时间限制

    try:
        # --- 运行 plan ---
        final_solution_id, duration_id, cost_dict_result_id = benchmark_planner_id.plan(
            cost_w_test, max_t_horizon_test, time_lim_test
        )
        # --- --------- ---
    except Exception as plan_e:
        print(f"错误: 执行 benchmark_planner_id.plan 时发生异常: {plan_e}")
        traceback.print_exc()
        sys.exit(1)

    if final_solution_id:
        print(f"  Benchmark (ID Prio) 规划成功！总耗时: {duration_id:.4f}s")
        if isinstance(cost_dict_result_id, dict):
            final_total_cost_id = cost_dict_result_id.get('total', float('inf'))
            final_cost_str_id = f"{final_total_cost_id:.2f}" if final_total_cost_id != float('inf') else "Inf"
            print(f"  最终解计算成本: Total={final_cost_str_id}")
            print(f"    Breakdown: Travel={cost_dict_result_id.get('travel', 'N/A'):.2f}, Turn={cost_dict_result_id.get('turn', 'N/A'):.2f}, Wait={cost_dict_result_id.get('wait', 'N/A'):.2f}")
            max_makespan_id = 0
            if isinstance(final_solution_id, dict):
                for agv_id, path in final_solution_id.items():
                    if isinstance(path, Path): max_makespan_id = max(max_makespan_id, path.get_makespan())
            print(f"  总 Makespan: {max_makespan_id}")
        else: print(f"错误: Benchmark.plan (ID Prio) 返回成本非字典 ({type(cost_dict_result_id)})。")
    else: print(f"  Benchmark (ID Prio) 规划失败。总耗时: {duration_id:.4f}s")

    # --- 测试场景 2: 速度优先级 ---
    print("\n[测试 2] 使用 Speed 优先级策略运行 Benchmark...")
    try:
        # --- 修改 Benchmark 初始化，传入速度和策略 ---
        benchmark_planner_speed = PrioritizedPlanner(
            test_map_s1, test_tasks_s1, planner_instance_bm,
            agv_speeds=test_speeds_s1, # <--- 传递速度
            delta_step=1.0,
            priority_strategy='speed' # <--- 设置为 Speed 优先级
        )
        # --- ------------------------------------- ---
    except Exception as init_e:
        print(f"错误: 初始化 PrioritizedPlanner (Speed Prio) 失败: {init_e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        # --- 运行 plan ---
        final_solution_speed, duration_speed, cost_dict_result_speed = benchmark_planner_speed.plan(
            cost_w_test, max_t_horizon_test, time_lim_test
        )
        # --- --------- ---
    except Exception as plan_e:
        print(f"错误: 执行 benchmark_planner_speed.plan 时发生异常: {plan_e}")
        traceback.print_exc()
        sys.exit(1)

    if final_solution_speed:
        print(f"  Benchmark (Speed Prio) 规划成功！总耗时: {duration_speed:.4f}s")
        if isinstance(cost_dict_result_speed, dict):
            final_total_cost_speed = cost_dict_result_speed.get('total', float('inf'))
            final_cost_str_speed = f"{final_total_cost_speed:.2f}" if final_total_cost_speed != float('inf') else "Inf"
            print(f"  最终解计算成本: Total={final_cost_str_speed}")
            print(f"    Breakdown: Travel={cost_dict_result_speed.get('travel', 'N/A'):.2f}, Turn={cost_dict_result_speed.get('turn', 'N/A'):.2f}, Wait={cost_dict_result_speed.get('wait', 'N/A'):.2f}")
            max_makespan_speed = 0
            if isinstance(final_solution_speed, dict):
                for agv_id, path in final_solution_speed.items():
                    if isinstance(path, Path): max_makespan_speed = max(max_makespan_speed, path.get_makespan())
            print(f"  总 Makespan: {max_makespan_speed}")
        else: print(f"错误: Benchmark.plan (Speed Prio) 返回成本非字典 ({type(cost_dict_result_speed)})。")
    else: print(f"  Benchmark (Speed Prio) 规划失败。总耗时: {duration_speed:.4f}s")
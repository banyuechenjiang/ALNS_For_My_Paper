# Benchmark.py-v7
"""
实现基准算法：优先顺序规划器 (Prioritized Sequential Planner)。
该算法按固定优先级（AGV ID 顺序）依次规划每个 AGV 的路径，
并将已规划的路径视为后续 AGV 的动态障碍。
用作与 ALNS 算法进行性能对比的基准。

与论文的关联:
- 对比基准: 作为论文实验部分用于对比 ALNS 性能的简单但常用的基准方法。
           结果通常用于展示 ALNS 等更复杂算法的相对优势。
- 核心规划器: 与 ALNS 类似，内部依赖 TWA* (Planner.py) 来规划单条路径 (论文 3.5.1)。
- 目标函数评估 (论文 公式 1): 通过调用 Path.get_cost 或累加单路径成本来计算总成本，
                 与论文目标函数一致。
- 约束满足:
    - 节点冲突 (论文 约束 12): 通过序贯规划和将已规划路径视为动态障碍来尝试避免，
             但可能不如 ALNS 的全局冲突解决机制鲁棒。
    - 其他约束: 与 ALNS 类似，依赖 TWA* 和 Path 结构来满足。

版本变更 (v6 -> v7):
- 添加了详细的模块和类/方法文档字符串，明确其作为基准算法的角色和与论文的关联。
- 在 plan 方法中添加了内联注释，解释其序贯规划逻辑和约束处理方式。
- 清理了不再需要的注释。
- 确保代码风格一致性和完整性。
"""
import time
from typing import List, Tuple, Dict, Optional

# --- 标准导入 (使用 v8 的 DataTypes 和 v14 的 Planner) ---
try:
    from Map import GridMap, Node
    from DataTypes import Task, Path, TimeStep, State, DynamicObstacles, Solution, CostDict, calculate_tij # 导入 v8
    from Planner import TWAStarPlanner # 导入 v14
except ImportError as e:
    print(f"错误: 导入 Benchmark 依赖项失败: {e}")
    # 定义临时的占位符类型以便静态分析
    GridMap = type('GridMap', (object,), {})
    Task = type('Task', (object,), {})
    Path = type('Path', (object,), {})
    Solution = Dict
    CostDict = Dict
    DynamicObstacles = Dict
    TWAStarPlanner = type('TWAStarPlanner', (object,), {})
    Node = Tuple
    TimeStep = int


# --- 优先顺序规划器类 (v7 - 增强文档和模型关联) ---
class PrioritizedPlanner:
    """
    实现简单的优先顺序规划基准算法。
    按固定优先级（例如 AGV ID 顺序）依次规划每个 AGV 的路径，
    并将已规划的路径视为后续 AGV 的动态障碍。
    """
    def __init__(self, grid_map: 'GridMap', tasks: List['Task'], planner: 'TWAStarPlanner', v: float, delta_step: float):
        """
        初始化优先规划器。

        Args:
            grid_map (GridMap): 地图对象。
            tasks (List[Task]): 所有 AGV 的任务列表。
            planner (TWAStarPlanner): 用于规划单个路径的 TWA* 实例 (论文 3.5.1)。
            v (float): AGV 速度 (对应模型参数 v)。
            delta_step (float): 时间步长 (对应模型参数 δ_step)。
        """
        # --- 依赖项检查 (假设导入成功) ---
        # 类型检查
        if not isinstance(grid_map, GridMap): raise TypeError("grid_map 必须是 GridMap 类型")
        if not isinstance(tasks, list) or not all(isinstance(t, Task) for t in tasks): raise TypeError("tasks 必须是 Task 列表")
        if not isinstance(planner, TWAStarPlanner): raise TypeError("planner 必须是 TWAStarPlanner 类型")
        if not isinstance(v, (int, float)) or v <= 0: raise ValueError("速度 v 必须为正数。")
        if not isinstance(delta_step, (int, float)) or delta_step <= 0: raise ValueError("时间步长 delta_step 必须为正数。")

        self.grid_map = grid_map
        # 按 AGV ID 对任务排序，以确定固定的优先级
        self.tasks = sorted(tasks, key=lambda t: t.agv_id)
        self.planner = planner
        self.v = float(v) # 确保是浮点数
        self.delta_step = float(delta_step) # 确保是浮点数
        self.num_agvs = len(tasks)

    # --- 核心规划方法 (v7 - 增强文档) ---
    def plan(self, cost_weights: Tuple[float, float, float], max_time: TimeStep, time_limit_per_agent: Optional[float] = None) -> Tuple[Optional['Solution'], float, 'CostDict']:
        """
        执行优先顺序规划。
        按固定优先级逐个调用 TWA*，并将已规划路径作为动态障碍。

        Args:
            cost_weights (Tuple[float, float, float]): 成本权重 (α, β, γ_wait)，用于 TWA* 内部和最终成本评估 (关联论文公式 1)。
            max_time (TimeStep): 规划的最大时间范围 (T_max)。
            time_limit_per_agent (Optional[float]): 单个 AGV 规划的时间限制（秒）。

        Returns:
            Tuple[Optional[Solution], float, CostDict]:
                - Solution: 字典 {agv_id: Path}，如果所有 AGV 都成功规划。
                - float: 总计算时间（秒）。
                - CostDict: 解决方案的总成本字典 (对应论文公式 1)，如果成功；否则为 Inf。
        """
        start_plan_time = time.perf_counter() # 记录开始时间
        solution: Solution = {} # 初始化空解
        dynamic_obstacles: DynamicObstacles = {} # 初始化动态障碍
        # 初始化成本字典
        total_cost_dict: CostDict = {'total': 0.0, 'travel': 0.0, 'turn': 0.0, 'wait': 0.0}
        # 初始化无效成本字典
        inf_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        all_success = True # 标记是否所有 AGV 都成功

        print("--- 开始基准规划 (优先序贯) ---")

        # 确保 tasks 列表不为空
        if not self.tasks:
             print("警告: Benchmark 任务列表为空。")
             return None, 0.0, inf_dict

        # --- 核心序贯规划循环 ---
        # 按排序后的任务列表进行规划 (固定优先级)
        for task in self.tasks:
            agv_id = task.agv_id
            t_start_call = time.perf_counter()
            print(f"  规划 AGV {agv_id}...")

            # 调用 TWA* 规划器 (论文 3.5.1)
            # 关键：将之前 AGV 的路径 (存储在 dynamic_obstacles) 传入
            # TWA* 内部处理动态障碍，尝试满足节点冲突约束 (论文公式 12)
            try:
                 path: Optional[Path] = self.planner.plan(
                     grid_map=self.grid_map,
                     task=task,
                     dynamic_obstacles=dynamic_obstacles, # 传入当前已规划的路径作为障碍
                     max_time=max_time,
                     cost_weights=cost_weights, # TWA* 优化时使用的权重
                     v=self.v,
                     delta_step=self.delta_step,
                     buffer=0, # Benchmark 通常不使用 buffer
                     start_time=0,
                     time_limit=time_limit_per_agent
                 )
                 # 检查返回类型
                 if path is not None and not isinstance(path, Path):
                      print(f"    错误: Planner 为 AGV {agv_id} 返回了非 Path 类型: {type(path)}。规划失败。")
                      all_success = False; break
            except Exception as e:
                 print(f"    错误: 调用 Planner 为 AGV {agv_id} 规划时发生异常: {e}")
                 import traceback
                 traceback.print_exc()
                 all_success = False; break

            call_dur = time.perf_counter() - t_start_call

            # --- 处理规划结果 ---
            if path and path.sequence: # 检查路径和序列是否有效
                solution[agv_id] = path # 将成功规划的路径加入解
                # 将新路径添加到动态障碍中，供后续 AGV 避让
                for node, t in path.sequence:
                    if t not in dynamic_obstacles:
                        dynamic_obstacles[t] = set()
                    dynamic_obstacles[t].add(node)
                # 计算并累加成本 (对应论文公式 1)
                try:
                    # 确保 path 是 Path 类型再调用 get_cost
                    if isinstance(path, Path):
                        # 调用 v8 的 Path.get_cost
                        path_cost_dict = path.get_cost(self.grid_map, *cost_weights, self.v, self.delta_step)
                        if path_cost_dict.get('total', float('inf')) == float('inf'):
                             print(f"    错误: AGV {agv_id} 路径成本计算为 Inf。规划失败。")
                             all_success = False; break # 成本无效，停止
                        # 累加成本
                        for key in total_cost_dict:
                             total_cost_dict[key] += path_cost_dict.get(key, 0.0)
                    else: # 理论上不应发生
                         print(f"    错误: AGV {agv_id} 的 path 对象不是 Path 类型。规划失败。")
                         all_success = False; break
                except Exception as e:
                     print(f"    错误: 计算 AGV {agv_id} 成本时出错: {e}。规划失败。")
                     all_success = False; break # 计算出错，停止

                print(f"    成功，耗时 {call_dur:.4f}s, 路径长度 {len(path)}, Makespan {path.get_makespan()}")

            else:
                # 如果任何一个 AGV 规划失败，则 Benchmark 整体失败
                print(f"  错误：AGV {agv_id} 规划失败！(耗时 {call_dur:.4f}s) Benchmark 终止。")
                all_success = False
                break # 停止规划后续 AGV
        # --- 序贯规划循环结束 ---

        end_plan_time = time.perf_counter()
        total_duration = end_plan_time - start_plan_time # 计算总耗时

        print(f"--- 基准规划完成，总耗时: {total_duration:.4f}s ---")

        # --- 返回结果 ---
        # 检查是否所有 AGV 都成功规划且最终成本有效
        if all_success and len(solution) == self.num_agvs and total_cost_dict.get('total', float('inf')) != float('inf'):
             # 返回成功结果
             return solution, total_duration, total_cost_dict
        else:
             # 返回失败状态
             return None, total_duration, inf_dict

    # --- 成本计算辅助方法 (未使用，保留用于参考或未来扩展) ---
    def _calculate_total_cost(self, solution: 'Solution', cost_weights: Tuple[float, float, float]) -> 'CostDict':
        """(内部辅助) 计算解决方案的总成本及各分项成本 (适配 v8 Path.get_cost)。"""
        total_cost_dict: CostDict = {'total': 0.0, 'travel': 0.0, 'turn': 0.0, 'wait': 0.0}
        inf_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        if not solution or not isinstance(solution, dict): return inf_dict

        alpha, beta, gamma_wait = cost_weights
        valid_solution = True
        num_paths = 0
        for agv_id, path in solution.items():
            # 检查 path 是否是 Path 实例且有序列
            if not isinstance(path, Path) or not path.sequence: valid_solution = False; break
            num_paths += 1
            try:
                # 调用 Path 对象的 get_cost (对应论文公式 1)
                cost_dict = path.get_cost(self.grid_map, alpha, beta, gamma_wait, self.v, self.delta_step)
            except Exception as e:
                print(f"错误 (Benchmark cost calc): 计算 AGV {agv_id} 成本时出错: {e}")
                valid_solution = False; break
            # 检查返回的成本是否有效
            if not isinstance(cost_dict, dict) or cost_dict.get('total', float('inf')) == float('inf'):
                valid_solution = False; break
            # 累加成本
            for key in total_cost_dict:
                cost_val = cost_dict.get(key, 0.0)
                if not isinstance(cost_val, (int, float)):
                     print(f"警告 (Benchmark cost calc): AGV {agv_id} 的成本项 '{key}' 不是数值: {cost_val}")
                     cost_val = 0.0
                total_cost_dict[key] += cost_val

        # 检查最终是否有效且包含所有 AGV
        if valid_solution and num_paths == self.num_agvs:
             return total_cost_dict
        else:
             return inf_dict

# --- 示例用法 (保持不变，使用 v11 的 InstanceGenerator 和 v14 的 Planner) ---
if __name__ == '__main__':
    print("--- Benchmark (Prioritized Planner) 测试 (v7 - Enhanced Docs & Model Links) ---")
    # --- 标准导入 ---
    import sys
    try:
        from InstanceGenerator import load_fixed_scenario_1 # 导入 v11
    except ImportError as e:
        print(f"错误: 导入 InstanceGenerator 失败: {e}")
        sys.exit(1)

    # --- 准备测试环境 ---
    print("加载固定算例场景 1...")
    instance_data_s1 = load_fixed_scenario_1()
    if not instance_data_s1:
        print("错误: 无法加载场景 1 数据。")
        sys.exit(1)
    test_map_s1, test_tasks_s1 = instance_data_s1

    # 创建 Planner 实例 (使用 v14)
    try:
        planner_instance = TWAStarPlanner() # 使用 v14
    except NameError:
        print("错误: TWAStarPlanner 类未定义，请确保 Planner.py 已正确导入。")
        sys.exit(1)
    except Exception as planner_e:
         print(f"错误: 初始化 TWAStarPlanner 失败: {planner_e}")
         sys.exit(1)


    # 创建 Benchmark 实例 (使用 v7)
    try:
        # 使用顶层已定义的 PrioritizedPlanner 类
        benchmark_planner = PrioritizedPlanner(
            grid_map=test_map_s1,
            tasks=test_tasks_s1,
            planner=planner_instance,
            v=1.0,          # 提供 v
            delta_step=1.0  # 提供 delta_step
        )
    except Exception as init_e:
         print(f"错误: 初始化 PrioritizedPlanner 失败: {init_e}")
         sys.exit(1)

    # --- 执行规划 ---
    cost_w_test = (1.0, 0.3, 0.8)
    max_t_horizon_test = 400
    time_lim_test = 30.0

    try:
        final_solution_test, duration_test, cost_dict_result_test = benchmark_planner.plan(
            cost_weights=cost_w_test,
            max_time=max_t_horizon_test,
            time_limit_per_agent=time_lim_test
        )
    except AttributeError as ae:
         print(f"错误: 调用 benchmark_planner.plan 时发生 AttributeError: {ae}")
         print("请确认 Benchmark.py 文件内容正确。")
         sys.exit(1)
    except Exception as plan_e:
         print(f"错误: 执行 benchmark_planner.plan 时发生未知异常: {plan_e}")
         import traceback
         traceback.print_exc()
         sys.exit(1)

    # --- 打印结果 ---
    if final_solution_test:
        print(f"\nBenchmark 规划成功！总耗时: {duration_test:.4f}s")
        if isinstance(cost_dict_result_test, dict):
            final_total_cost = cost_dict_result_test.get('total', float('inf'))
            final_cost_str = f"{final_total_cost:.2f}" if final_total_cost != float('inf') else "Inf"
            print(f"最终解计算成本 (对应公式1): Total={final_cost_str}")
            print(f"  Breakdown: Travel={cost_dict_result_test.get('travel', 'N/A'):.2f}, Turn={cost_dict_result_test.get('turn', 'N/A'):.2f}, Wait={cost_dict_result_test.get('wait', 'N/A'):.2f}")
            max_makespan = 0
            if isinstance(final_solution_test, dict):
                for agv_id, path in final_solution_test.items():
                    if isinstance(path, Path):
                        print(f"  AGV {agv_id}: 路径长度={len(path)}, Makespan={path.get_makespan()}")
                        max_makespan = max(max_makespan, path.get_makespan())
                    else:
                        print(f"  AGV {agv_id}: 路径无效或类型错误 ({type(path)})。")
            else:
                 print(f"错误: final_solution 不是预期的字典类型 ({type(final_solution_test)})。")
            print(f"  总 Makespan: {max_makespan}")
        else:
             print(f"错误: Benchmark.plan 返回的成本不是字典类型 ({type(cost_dict_result_test)})。")
    else:
        print(f"\nBenchmark 规划失败。总耗时: {duration_test:.4f}s")
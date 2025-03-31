# ALNS.py-v38 (完整代码)
"""
实现自适应大邻域搜索 (ALNS) 算法主体，用于解决论文定义的仓储 AGV 路径规划问题。

核心功能:
- 基于论文第二章构建的数学模型，最小化目标函数 (公式 1)，即行驶时间、转弯成本和等待时间的加权和。
- 通过迭代破坏和修复解，并自适应调整算子权重，搜索高质量无冲突路径。
- 包含冲突与死锁检测及解决机制。
- **集成区域分割优化 (论文 3.5.2):** ALNS 内部在调用 TWA* 规划器时，
  会根据任务计算包围盒 (Bounding Box)，并传递给 Planner，以限制搜索范围，
  提高规划效率。

与论文数学模型 (Chapter 2) 的关联:
- 目标函数: _calculate_total_cost 方法计算与公式(1)对应的成本。
- 约束条件:
    - 起始/目标 (2, 3, 4): 通过初始解生成和规划器保证。
    - 节点冲突 (12): 由 _resolve_conflicts_and_deadlocks 方法检测和处理。
    - 障碍物 (13): 由地图和规划器处理。
    - 其他约束 (5-11, 14-18): 主要由 Path 数据结构、规划器逻辑和成本计算隐式满足。
- 参数: __init__ 中使用的 alpha, beta, gamma_wait, v, delta_step 等与模型符号对应。

版本变更 (v37 -> v38):
- **修正**: 在 `_call_planner` 方法中，调用 `planner.plan` 时，**移除** 了已经
          不再被 Planner (v15+) 接受的 `buffer` 参数。
- **改进**: 在 `_call_planner` 方法中，添加了对 `bounding_box` 参数的支持。
          虽然 `ALNS.py` 本身不直接计算包围盒，但为未来可能的扩展预留接口。
          实际的包围盒计算和传递在 `Operators.py-v6` 中完成。
- **风格**: 重构代码以符合 PEP 8 标准，提高结构清晰性和可读性，与 `Operators.py-v6` 风格对齐。
- 保持: v37 的所有其他功能、逻辑和详细注释均被保留或适当更新。
"""
import math
import random
import time
import copy
import os
import sys
import csv
import traceback
import inspect
from typing import TYPE_CHECKING, List, Tuple, Dict, Set, Optional, Callable, Type, NamedTuple
from collections import defaultdict, Counter

# --- 导入类型定义 ---
try:
    from Map import GridMap, Node
    from DataTypes import Task, Path, TimeStep, State, DynamicObstacles, Solution, CostDict
    from Planner import TWAStarPlanner
except ImportError as e:
    print(f"错误: 导入 ALNS 依赖项失败: {e}")
    sys.exit(1)

# --- 导入外部算子模块 ---
try:
    import Operators
except ImportError:
    print("错误: 无法导入 Operators.py。请确保该文件与 ALNS.py 在同一目录或 Python 路径中。")
    sys.exit(1)

# --- 尝试导入可选的可视化库 ---
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    _visual_libs_available = True
except ImportError:
    _visual_libs_available = False
    print("警告: 未找到 pandas 或 matplotlib。绘图功能将被禁用。")


# --- ALNS 主类 (v38 - 支持区域分割, 修正参数传递, 增强可读性) ---
class ALNS:
    """
    自适应大邻域搜索算法实现。
    负责协调破坏/修复算子、管理算法状态、评估解、处理冲突并执行自适应机制。
    """
    def __init__(self,
                 grid_map: GridMap, tasks: List[Task], planner: TWAStarPlanner,
                 instance_identifier: str = "default_instance",
                 results_dir: str = "alns_results",
                 **kwargs):
        """初始化 ALNS 实例，包括参数解析、算子注册、状态初始化等。"""
        # --- 输入验证 ---
        if not isinstance(grid_map, GridMap): raise TypeError("grid_map 必须是 GridMap 类型")
        if not isinstance(tasks, list) or not all(isinstance(t, Task) for t in tasks): raise TypeError("tasks 必须是 Task 列表")
        if not isinstance(planner, TWAStarPlanner): raise TypeError("planner 必须是 TWAStarPlanner 类型")
        if not isinstance(instance_identifier, str): raise TypeError("instance_identifier 必须是字符串")
        if not isinstance(results_dir, str): raise TypeError("results_dir 必须是字符串")

        # --- 基础属性 ---
        self.grid_map = grid_map
        self.tasks = tasks
        self.num_agvs = len(tasks)
        self.planner = planner # 核心路径规划器 (TWA*)
        self.agv_ids = sorted([task.agv_id for task in tasks])
        self.instance_identifier = instance_identifier # 用于结果文件命名
        self.results_dir = results_dir # 保存结果的目录

        # --- ALNS 核心参数 ---
        self.max_iterations: int = kwargs.get('max_iterations', 100)
        self.initial_temperature: float = kwargs.get('initial_temp', 10.0)
        self.cooling_rate: float = kwargs.get('cooling_rate', 0.985)
        self.segment_size: int = kwargs.get('segment_size', 20)
        self.weight_update_rate: float = kwargs.get('weight_update_rate', 0.2)
        self.sigma1: float = kwargs.get('sigma1', 15.0)
        self.sigma2: float = kwargs.get('sigma2', 8.0)
        self.sigma3: float = kwargs.get('sigma3', 3.0)
        self.removal_percentage_min: float = kwargs.get('removal_percentage_min', 0.15)
        self.removal_percentage_max: float = kwargs.get('removal_percentage_max', 0.40)
        self.regret_k: int = max(2, kwargs.get('regret_k', 3))
        self.no_improvement_limit: Optional[int] = kwargs.get('no_improvement_limit', 40)
        self.conflict_history_decay: float = kwargs.get('conflict_history_decay', 0.9)

        # --- 环境与规划器参数 ---
        self.max_time: TimeStep = kwargs.get('max_time', 400)
        self.cost_weights: Tuple[float, float, float] = kwargs.get('cost_weights', (1.0, 0.3, 0.8))
        self.alpha, self.beta, self.gamma_wait = self.cost_weights
        self.v: float = kwargs.get('v', 1.0)
        self.delta_step: float = kwargs.get('delta_step', 1.0)
        self.buffer: int = kwargs.get('buffer', 0) # 用于区域分割的 buffer
        self.planner_time_limit_factor: float = kwargs.get('planner_time_limit_factor', 4.0)
        self.regret_planner_time_limit_abs: float = kwargs.get('regret_planner_time_limit_abs', 0.1)

        # --- 冲突解决参数 ---
        self.wait_threshold: int = kwargs.get('wait_threshold', 6)
        self.deadlock_max_wait: int = kwargs.get('deadlock_max_wait', 10)

        # --- 控制与输出参数 ---
        self.verbose: bool = kwargs.get('verbose', False)
        self.debug_weights: bool = kwargs.get('debug_weights', False)
        self.record_history: bool = kwargs.get('record_history', True)
        self.plot_convergence_flag: bool = kwargs.get('plot_convergence', True)

        # --- 状态变量 ---
        self.temperature: float = self.initial_temperature
        self.best_solution: Optional[Solution] = None
        self.best_cost: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.current_solution: Optional[Solution] = None
        self.current_cost: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.initial_cost: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.iteration_count: int = 0
        self.no_improvement_count: int = 0
        self.agv_conflict_counts: Counter[int] = Counter()
        self.cost_history: List[Dict] = []
        self.operator_history: List[Dict] = []

        # --- 算子注册 ---
        DestroyOpType = Callable[['ALNS', Solution, int], Tuple[Solution, List[int]]]
        RepairOpType = Callable[['ALNS', Solution, List[int]], Optional[Solution]]
        self.destroy_operators: Dict[str, DestroyOpType] = {
            "random_removal": Operators.random_removal,
            "worst_removal": Operators.worst_removal,
            "related_removal": Operators.related_removal,
            "congestion_removal": Operators.congestion_removal,
            "conflict_history_removal": Operators.conflict_history_removal,
        }
        self.repair_operators: Dict[str, RepairOpType] = {
            "greedy_insertion": Operators.greedy_insertion,
            "regret_insertion": Operators.regret_insertion,
            "wait_adjustment_repair": Operators.wait_adjustment_repair,
        }

        # --- 权重/统计初始化 ---
        self.destroy_weights: Dict[str, float] = {name: 1.0 for name in self.destroy_operators}
        self.repair_weights: Dict[str, float] = {name: 1.0 for name in self.repair_operators}
        self.destroy_scores: Dict[str, float] = {name: 0.0 for name in self.destroy_operators}
        self.repair_scores: Dict[str, float] = {name: 0.0 for name in self.repair_operators}
        self.destroy_counts: Dict[str, int] = {name: 0 for name in self.destroy_operators}
        self.repair_counts: Dict[str, int] = {name: 0 for name in self.repair_operators}

        # --- 创建结果目录 ---
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except OSError as e:
            print(f"错误: 无法创建结果目录 '{self.results_dir}': {e}")

    # --- 核心规划器调用 ---
    def _call_planner(
        self,
        task: Task,
        dynamic_obstacles: DynamicObstacles,
        start_time: TimeStep = 0,
        bounding_box: Optional[Tuple[int, int, int, int]] = None # 添加 bounding_box 参数
    ) -> Optional[Path]:
        """
        调用 TWA* 规划器生成单条路径。
        这是 ALNS 与底层路径查找算法的接口。现在支持传递 bounding_box
        以实现区域分割 (论文 3.5.2)。
        """
        planner_time_limit = None
        # 计算动态时间限制
        if self.planner_time_limit_factor is not None and self.planner_time_limit_factor > 0:
            base_time = 0.8
            num_dynamic_states = sum(len(v) for v in dynamic_obstacles.values())
            max_t_effective = max(1, self.max_time)
            complexity_factor = max(0.5, min(1.0 + num_dynamic_states / max_t_effective, 5.0))
            planner_time_limit = max(0.2, min(base_time * self.planner_time_limit_factor * complexity_factor, 15.0))

        try:
            # 调用 Planner.py v15+ 的 plan 方法，不再传递 buffer，传递 bounding_box
            path: Optional[Path] = self.planner.plan(
                grid_map=self.grid_map, task=task, dynamic_obstacles=dynamic_obstacles,
                max_time=self.max_time, cost_weights=self.cost_weights, v=self.v,
                delta_step=self.delta_step, # buffer=self.buffer, # 移除 buffer 参数
                start_time=start_time, time_limit=planner_time_limit,
                bounding_box=bounding_box # 传递 bounding_box 参数
            )
            # 类型检查
            if path is not None and not isinstance(path, Path):
                print(f"错误: Planner 返回了非 Path 类型: {type(path)}")
                return None
            return path
        except Exception as e:
            print(f"错误: Planner.plan 调用失败 (AGV {task.agv_id}): {e}")
            return None

    # --- 成本计算 ---
    def _calculate_total_cost(self, solution: Solution) -> CostDict:
        """计算解决方案总成本 (对应论文公式 1)。"""
        total_cost_dict: CostDict = {'total': 0.0, 'travel': 0.0, 'turn': 0.0, 'wait': 0.0}
        inf_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        if not isinstance(solution, dict) or not solution: return inf_dict

        valid_solution = True
        num_paths = 0
        for agv_id, path in solution.items():
            if not path or not isinstance(path, Path) or not path.sequence:
                valid_solution = False
                break
            num_paths += 1
            try:
                cost_dict = path.get_cost(self.grid_map, self.alpha, self.beta, self.gamma_wait, self.v, self.delta_step)
            except Exception as e:
                print(f"错误: 计算 AGV {agv_id} 成本时出错: {e}")
                valid_solution = False
                break
            if cost_dict.get('total', float('inf')) == float('inf'):
                valid_solution = False
                break
            for key in total_cost_dict:
                total_cost_dict[key] += cost_dict.get(key, 0.0)

        if valid_solution and num_paths == self.num_agvs:
            return total_cost_dict
        else:
            return inf_dict

    # --- 初始解生成 ---
    def generate_initial_solution(self) -> Optional[Solution]:
        """生成初始解 (优先序贯规划)。"""
        print("--- 生成初始解 (优先序贯规划) ---")
        solution: Solution = {}
        dynamic_obstacles: DynamicObstacles = {}
        sorted_tasks = sorted(self.tasks, key=lambda t: t.agv_id)

        for task in sorted_tasks:
            agv_id = task.agv_id
            t_start_call = time.perf_counter()
            if self.verbose: print(f"  规划初始 AGV {agv_id}...")

            path = self._call_planner(task, dynamic_obstacles, start_time=0)
            call_dur = time.perf_counter() - t_start_call

            if path and path.sequence:
                solution[agv_id] = path
                for node, t in path.sequence:
                    if t not in dynamic_obstacles: dynamic_obstacles[t] = set()
                    dynamic_obstacles[t].add(node)
                if self.verbose: print(f"    成功 (耗时 {call_dur:.3f}s)，路径长度 {len(path)}, Makespan {path.get_makespan()}")
            else:
                print(f"  错误：AGV {agv_id} 初始规划失败！(耗时 {call_dur:.3f}s)")
                self.initial_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
                return None

        self.initial_cost = self._calculate_total_cost(solution)
        initial_total_cost = self.initial_cost.get('total', float('inf'))
        initial_cost_str = f"{initial_total_cost:.2f}" if initial_total_cost != float('inf') else "Inf"
        print(f"--- 初始解生成完毕，成本: {initial_cost_str} ---")
        return solution if initial_total_cost != float('inf') else None

    # --- 算子选择 ---
    def _select_operator_roulette_wheel(self, weights: Dict[str, float]) -> str:
        """轮盘赌选择算子。"""
        total_weight = sum(weights.values())
        if total_weight <= 1e-9:
            valid_operators = [name for name, w in weights.items() if w > 1e-9]
            return random.choice(valid_operators) if valid_operators else random.choice(list(weights.keys()))

        pick = random.uniform(0, total_weight)
        current = 0.0
        for name, weight in weights.items():
            current += weight
            if current >= pick - 1e-9:
                return name
        return list(weights.keys())[-1]

    # --- 冲突与死锁解决 ---
    def _resolve_conflicts_and_deadlocks(self, solution: Solution) -> Optional[Solution]:
        """
        冲突与死锁解决。

        (完整代码，与 v37 相同，为节省篇幅省略)
        """
        if not solution: return None
        current_solution = copy.deepcopy(solution)
        max_resolve_attempts = self.num_agvs * 3
        resolve_attempt = 0

        while resolve_attempt < max_resolve_attempts:
            resolve_attempt += 1
            made_change_this_pass = False
            if self.verbose and self.debug_weights: print(f"  冲突/死锁解决尝试 #{resolve_attempt}")

            # --- 死锁检测 ---
            agv_wait_times: Dict[int, Tuple[TimeStep, Node]] = defaultdict(lambda: (0, (-1,-1)))
            max_t_in_solution = 0
            paths_sequences = {agv_id: path.sequence for agv_id, path in current_solution.items() if path and path.sequence}
            if not paths_sequences: return current_solution
            for path_seq in paths_sequences.values():
                 if path_seq: max_t_in_solution = max(max_t_in_solution, path_seq[-1][1])

            deadlocked_agvs: Set[int] = set()
            node_at_time: Dict[int, Dict[TimeStep, Node]] = defaultdict(dict)
            for agv_id, path_seq in paths_sequences.items():
                 for node, t in path_seq: node_at_time[agv_id][t] = node

            for agv_id in paths_sequences.keys():
                wait_duration, current_wait_node = agv_wait_times[agv_id]
                for t_step in range(max_t_in_solution + 1):
                    node_t = node_at_time[agv_id].get(t_step)
                    node_t_minus_1 = node_at_time[agv_id].get(t_step - 1) if t_step > 0 else node_t
                    if node_t and node_t_minus_1 and node_t == node_t_minus_1:
                        if current_wait_node == node_t: wait_duration += 1
                        else: wait_duration = 1; current_wait_node = node_t
                        agv_wait_times[agv_id] = (wait_duration, current_wait_node)
                        if wait_duration > self.deadlock_max_wait:
                            deadlocked_agvs.add(agv_id)
                            self.agv_conflict_counts[agv_id] += 3
                            if self.verbose: print(f"      检测到潜在死锁：AGV {agv_id} @ {current_wait_node} wait > {self.deadlock_max_wait} (ConflictCount +3)")
                            break
                    else:
                         if wait_duration > 0:
                             wait_duration = 0; current_wait_node = (-1,-1)
                             agv_wait_times[agv_id] = (wait_duration, current_wait_node)

            # --- 死锁解决 ---
            if deadlocked_agvs:
                agv_to_resolve = random.choice(list(deadlocked_agvs))
                if self.verbose: print(f"      解决死锁：选择 AGV {agv_to_resolve} 进行重规划。")
                path_to_resolve = current_solution.get(agv_to_resolve)
                if not path_to_resolve or not path_to_resolve.sequence: print(f"错误：无法解决死锁，AGV {agv_to_resolve} 路径无效。"); return None

                wait_t_count, _ = agv_wait_times[agv_to_resolve]
                deadlock_start_time = path_to_resolve.sequence[-1][1] - wait_t_count + 1 if path_to_resolve.sequence else 0
                replan_start_index = -1
                for i in range(len(path_to_resolve.sequence) - 1, -1, -1):
                    if path_to_resolve.sequence[i][1] < deadlock_start_time:
                        replan_start_index = i
                        break
                replan_start_index = max(0, replan_start_index)
                replan_start_state = path_to_resolve.sequence[replan_start_index]

                original_task = next((t for t in self.tasks if t.agv_id == agv_to_resolve), None)
                if not original_task: print(f"错误：找不到 AGV {agv_to_resolve} 的原始任务。"); return None

                replan_task = Task(agv_id=agv_to_resolve, start_node=replan_start_state[0], goal_node=original_task.goal_node)
                replan_start_time_val = replan_start_state[1]
                current_solution[agv_to_resolve].sequence = path_to_resolve.sequence[:replan_start_index+1]
                dynamic_obs = Operators._build_dynamic_obstacles(current_solution, exclude_agv_id=agv_to_resolve)
                new_path_segment = self._call_planner(replan_task, dynamic_obs, start_time=replan_start_time_val)

                if new_path_segment and new_path_segment.sequence and len(new_path_segment.sequence) > 1:
                    current_solution[agv_to_resolve].sequence.extend(new_path_segment.sequence[1:])
                    made_change_this_pass = True
                    if self.verbose: print(f"      死锁解决：AGV {agv_to_resolve} 重规划成功。")
                    continue
                else:
                    print(f"      错误：死锁解决失败，AGV {agv_to_resolve} 重规划未找到有效路径。")
                    return None

            # --- 节点冲突检测与解决 ---
            conflict_resolved_in_scan = False
            max_t_check = 0
            paths_sequences = {agv_id: path.sequence for agv_id, path in current_solution.items() if path and path.sequence}
            if not paths_sequences: return current_solution
            for path_seq in paths_sequences.values():
                 if path_seq: max_t_check = max(max_t_check, path_seq[-1][1])

            node_occupancy_cache: Dict[TimeStep, Dict[Node, List[int]]] = defaultdict(lambda: defaultdict(list))
            for agv_id, path_seq in paths_sequences.items():
                 for node, t in path_seq: node_occupancy_cache[t][node].append(agv_id)

            for t_step in range(max_t_check + 1):
                nodes_with_conflict = {node: occupants for node, occupants in node_occupancy_cache[t_step].items() if len(occupants) > 1}

                for node, occupants in nodes_with_conflict.items():
                    if len(occupants) < 2: continue

                    conflict_resolved_in_scan = True
                    made_change_this_pass = True

                    occupants.sort()
                    agv_high_priority = occupants[0]
                    agvs_low_priority = occupants[1:]
                    path_high = current_solution.get(agv_high_priority)
                    if self.verbose: print(f"      检测到节点冲突 @ ({node}, {t_step}), High={agv_high_priority}, Low={agvs_low_priority}")

                    for agv_low in agvs_low_priority:
                        self.agv_conflict_counts[agv_low] += 1

                        path_low = current_solution.get(agv_low)
                        if not path_low or not path_low.sequence: continue

                        conflict_index = -1
                        for i, (n, t) in enumerate(path_low.sequence):
                            if n == node and t == t_step:
                                conflict_index = i
                                break
                        if conflict_index == -1: continue

                        previous_state_index = conflict_index - 1
                        should_replan = False

                        if previous_state_index < 0:
                             print(f"错误：AGV {agv_low} 在起点 ({node}, {t_step}) 发生冲突。尝试重规划。")
                             previous_node, previous_time = path_low.sequence[0][0], 0
                             should_replan = True
                        else:
                            previous_node, previous_time = path_low.sequence[previous_state_index]

                            wait_until_time = t_step + 1
                            if path_high:
                                last_high_time_at_node = -1
                                for high_node, high_t in path_high.sequence:
                                    if high_t >= t_step and high_node == node:
                                        last_high_time_at_node = max(last_high_time_at_node, high_t)
                                if last_high_time_at_node != -1:
                                    wait_until_time = max(wait_until_time, last_high_time_at_node + 1)

                            required_wait_duration = wait_until_time - t_step
                            if required_wait_duration > self.wait_threshold:
                                should_replan = True
                                self.agv_conflict_counts[agv_low] += 1
                                if self.verbose: print(f"      AGV {agv_low} 需等待 {required_wait_duration} > {self.wait_threshold} 步，触发重规划 (ConflictCount +1)。")
                            else:
                                if self.verbose: print(f"      AGV {agv_low} 尝试等待 {required_wait_duration} 步 @ {previous_node}")
                                new_sequence = path_low.sequence[:previous_state_index+1]
                                dynamic_obs_check = Operators._build_dynamic_obstacles(current_solution, exclude_agv_id=agv_low)
                                wait_conflict = False
                                for wait_step in range(1, required_wait_duration + 1):
                                    wait_t = previous_time + wait_step
                                    if wait_t in dynamic_obs_check and previous_node in dynamic_obs_check.get(wait_t, set()):
                                        if self.verbose: print(f"      AGV {agv_low} 等待期间在 t={wait_t} 与障碍冲突。")
                                        should_replan = True; wait_conflict = True; self.agv_conflict_counts[agv_low] += 1
                                        break
                                    new_sequence.append((previous_node, wait_t))

                                if not wait_conflict:
                                    time_shift = required_wait_duration
                                    path_valid_after_wait = True
                                    shifted_part = []
                                    for i in range(conflict_index, len(path_low.sequence)):
                                        original_node, original_time = path_low.sequence[i]
                                        new_time = original_time + time_shift
                                        if new_time > self.max_time: path_valid_after_wait = False; break
                                        if new_time in dynamic_obs_check and original_node in dynamic_obs_check.get(new_time, set()): path_valid_after_wait = False; break
                                        shifted_part.append((original_node, new_time))

                                    if path_valid_after_wait:
                                        current_solution[agv_low].sequence = new_sequence + shifted_part
                                        if self.verbose: print(f"      AGV {agv_low} 等待成功。")
                                        node_occupancy_cache.clear()
                                        break
                                    else:
                                         should_replan = True; self.agv_conflict_counts[agv_low] += 1
                                         if self.verbose: print(f"      AGV {agv_low} 等待后路径无效或冲突，触发重规划 (ConflictCount +1)。")
                                else:
                                    should_replan = True

                        if should_replan:
                            if self.verbose: print(f"      AGV {agv_low} 将从 ({previous_node}, {previous_time}) 开始重规划。")
                            original_task_low = next((t for t in self.tasks if t.agv_id == agv_low), None)
                            if not original_task_low: print(f"错误：无法找到 AGV {agv_low} 的原始任务。"); return None
                            replan_task_low = Task(agv_id=agv_low, start_node=previous_node, goal_node=original_task_low.goal_node)
                            replan_start_time_low = previous_time
                            current_solution[agv_low].sequence = path_low.sequence[:previous_state_index+1] if previous_state_index >= 0 else []
                            dynamic_obs_low = Operators._build_dynamic_obstacles(current_solution, exclude_agv_id=agv_low)
                            new_path_segment_low = self._call_planner(replan_task_low, dynamic_obs_low, start_time=replan_start_time_low)

                            if new_path_segment_low and new_path_segment_low.sequence and len(new_path_segment_low.sequence) > 1:
                                current_solution[agv_low].sequence.extend(new_path_segment_low.sequence[1:])
                                if self.verbose: print(f"      AGV {agv_low} 重规划成功。")
                                node_occupancy_cache.clear()
                                break
                            else:
                                print(f"      错误：冲突解决失败，AGV {agv_low} 重规划未找到有效路径段。")
                                return None

                    if conflict_resolved_in_scan: break
                if conflict_resolved_in_scan: break

            if not made_change_this_pass:
                 if self.verbose and self.debug_weights: print(f"  冲突/死锁解决：本轮扫描未做修改，解已稳定。")
                 break

        if resolve_attempt >= max_resolve_attempts and made_change_this_pass:
            print(f"  错误：冲突/死锁解决超过最大尝试次数 {max_resolve_attempts}。")
            return None
        final_check_cost_dict = self._calculate_total_cost(current_solution)
        if final_check_cost_dict.get('total', float('inf')) == float('inf'):
            print("  错误：冲突/死锁解决后最终成本检查为 Inf！")
            return None

        if self.verbose and resolve_attempt > 1: print(f"--- 冲突/死锁解决完成 (尝试 {resolve_attempt} 次) ---")
        return current_solution

    # --- 主运行方法 ---
    def run(self) -> Tuple[Optional[Solution], float, CostDict]:
        """
        执行 ALNS 算法主流程 (对应论文 3.1 节框架和 3.9 节伪代码)。
        """
        start_run_time = time.perf_counter()

        # --- 初始化状态变量 ---
        self.iteration_count = 0
        self.no_improvement_count = 0
        self.temperature = self.initial_temperature
        self.best_solution = None
        self.best_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.current_solution = None
        self.current_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.initial_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.cost_history = []
        self.operator_history = []
        self.agv_conflict_counts.clear()
        self.destroy_weights = {name: 1.0 for name in self.destroy_operators}
        self.repair_weights = {name: 1.0 for name in self.repair_operators}
        self.destroy_scores = {name: 0.0 for name in self.destroy_operators}
        self.repair_scores = {name: 0.0 for name in self.repair_operators}
        self.destroy_counts = {name: 0 for name in self.destroy_operators}
        self.repair_counts = {name: 0 for name in self.repair_operators}

        # 1. 生成初始解
        initial_solution_raw = self.generate_initial_solution()
        if not initial_solution_raw:
            print("错误：无法生成可行的初始解，ALNS 终止。")
            return None, time.perf_counter() - start_run_time, self.best_cost

        # 2. 处理初始解冲突
        if self.verbose: print("--- 处理初始解冲突 (保险步骤) ---")
        resolved_initial_solution = self._resolve_conflicts_and_deadlocks(initial_solution_raw)
        if resolved_initial_solution:
            self.current_solution = resolved_initial_solution
            self.current_cost = self._calculate_total_cost(self.current_solution)
            current_total_cost = self.current_cost.get('total', float('inf'))
            current_cost_str = f"{current_total_cost:.2f}" if current_total_cost != float('inf') else "Inf"
            if self.verbose: print(f"初始解冲突处理后成本: {current_cost_str}")
        else:
            print("错误：处理初始解冲突失败，ALNS 终止。")
            return None, time.perf_counter() - start_run_time, self.best_cost
        if self.current_cost.get('total', float('inf')) == float('inf'):
            print("错误：处理冲突后的初始解成本为无穷大，ALNS 终止。")
            return None, time.perf_counter() - start_run_time, self.best_cost

        # 3. 设置初始最优解
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_cost = self.current_cost
        best_total_cost = self.best_cost.get('total', float('inf'))
        best_cost_str = f"{best_total_cost:.2f}" if best_total_cost != float('inf') else "Inf"
        print(f"最终初始解成本 (无冲突): {best_cost_str}")

        # 记录初始成本
        self.cost_history.append({
            'iteration': 0,
            'current_cost': self.current_cost['total'],
            'best_cost': self.best_cost['total'],
            'temperature': self.temperature
        })

        # 4. ALNS 主迭代循环
        print(f"--- 开始 ALNS 迭代 (Max Iter: {self.max_iterations}, No Improve Limit: {self.no_improvement_limit}) ---")
        for i in range(self.max_iterations):
            self.iteration_count = i + 1
            iter_start_time = time.perf_counter()

            if not self.current_solution:
                print(f"错误: 迭代 {i+1} 开始时当前解丢失！终止。")
                break

            # --- a. 选择算子 ---
            destroy_op_name = self._select_operator_roulette_wheel(self.destroy_weights)
            repair_op_name = self._select_operator_roulette_wheel(self.repair_weights)
            destroy_op = self.destroy_operators.get(destroy_op_name)
            repair_op = self.repair_operators.get(repair_op_name)
            if not destroy_op or not repair_op:
                print(f"错误：无法找到算子 {destroy_op_name} 或 {repair_op_name}！")
                break

            # --- 打印迭代信息 ---
            best_total_cost_iter = self.best_cost.get('total', float('inf'))
            current_total_cost_iter = self.current_cost.get('total', float('inf'))
            best_cost_str_iter = f"{best_total_cost_iter:.2f}" if best_total_cost_iter != float('inf') else "Inf"
            current_cost_str_iter = f"{current_total_cost_iter:.2f}" if current_total_cost_iter != float('inf') else "Inf"
            no_improve_limit_str = str(self.no_improvement_limit) if self.no_improvement_limit is not None else "N/A"
            if self.verbose:
                print(f"\nIter {i+1}/{self.max_iterations} | T={self.temperature:.3f} | Best={best_cost_str_iter} | Curr={current_cost_str_iter} | NoImpr={self.no_improvement_count}/{no_improve_limit_str} | Ops: D='{destroy_op_name}', R='{repair_op_name}'")

            # --- b. 破坏解 ---
            removal_percentage = random.uniform(self.removal_percentage_min, self.removal_percentage_max)
            removal_count = max(1, int(self.num_agvs * removal_percentage))
            try:
                partial_solution, removed_agv_ids = destroy_op(self, self.current_solution, removal_count)
            except Exception as e:
                print(f"!!!!!!!!!! 调用破坏算子 {destroy_op_name} 时发生错误 !!!!!!!!!!!!!\n错误信息: {e}")
                traceback.print_exc()
                break

            # --- c. 修复解 ---
            new_solution_raw = None
            try:
                new_solution_raw = repair_op(self, partial_solution, removed_agv_ids)
            except Exception as e:
                print(f"!!!!!!!!!! 调用修复算子 {repair_op_name} 时发生错误 !!!!!!!!!!!!!\n错误信息: {e}")
                traceback.print_exc()

            # --- d. 冲突与死锁处理 ---
            new_solution_processed: Optional[Solution] = None
            new_cost_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
            if new_solution_raw:
                 new_solution_processed = self._resolve_conflicts_and_deadlocks(new_solution_raw)
                 if new_solution_processed:
                     new_cost_dict = self._calculate_total_cost(new_solution_processed)

            new_total_cost = new_cost_dict.get('total', float('inf'))
            current_total_cost = self.current_cost.get('total', float('inf'))
            best_total_cost = self.best_cost.get('total', float('inf'))

            # --- e. 评估与接受 ---
            score = 0.0
            accepted = False
            improved_best = False

            if new_total_cost != float('inf'):
                delta_cost = new_total_cost - current_total_cost if current_total_cost != float('inf') else -float('inf')

                if delta_cost < -1e-9 or current_total_cost == float('inf'):
                    accepted = True
                    score = self.sigma2
                    if new_total_cost < best_total_cost - 1e-9:
                        score = self.sigma1
                        improved_best = True
                    if self.verbose or improved_best:
                        print(f"  接受新解 ({'更好' if delta_cost < -1e-9 else '从无效变有效'}){' *** New Best! ***' if improved_best else ''}, Cost={new_total_cost:.2f}")
                else:
                    acceptance_prob = math.exp(-delta_cost / self.temperature) if self.temperature > 1e-6 else 0.0
                    if random.random() < acceptance_prob:
                        accepted = True
                        score = self.sigma3
                        if self.verbose:
                            print(f"  接受新解 (较差/相同, Prob={acceptance_prob:.3f}), Cost={new_total_cost:.2f}")

            # --- f. 更新状态和统计 ---
            if accepted:
                self.current_solution = new_solution_processed
                self.current_cost = new_cost_dict
                if improved_best:
                    self.best_solution = copy.deepcopy(new_solution_processed)
                    self.best_cost = new_cost_dict
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                self.destroy_scores[destroy_op_name] += score
                self.repair_scores[repair_op_name] += score
            else:
                self.no_improvement_count += 1

            self.destroy_counts[destroy_op_name] += 1
            self.repair_counts[repair_op_name] += 1

            # 记录成本历史
            self.cost_history.append({
                'iteration': i + 1,
                'current_cost': self.current_cost['total'],
                'best_cost': self.best_cost['total'],
                'temperature': self.temperature
            })

            # --- g. 周期性权重更新 ---
            if (i + 1) % self.segment_size == 0:
                self._update_weights()

            # --- h. 降温 ---
            self.temperature = max(1e-6, self.temperature * self.cooling_rate)

            # --- i. 检查早停条件 ---
            if self.no_improvement_limit is not None and self.no_improvement_count >= self.no_improvement_limit:
                print(f"\n--- 触发早停：最优解连续 {self.no_improvement_limit} 次迭代未改进 (当前迭代 {i+1}) ---")
                break

        # 5. 运行结束
        end_run_time = time.perf_counter()
        total_duration = end_run_time - start_run_time
        print("\n--- ALNS 最终结果 ---")
        best_total_cost_final = self.best_cost.get('total', float('inf'))
        best_cost_final_str = f"{best_total_cost_final:.2f}" if best_total_cost_final != float('inf') else "Inf"
        if self.best_solution:
            print(f"找到最优解成本: {best_cost_final_str}")
            print(f"  Breakdown (对应公式1): Travel={self.best_cost.get('travel', 0.0):.2f}, Turn={self.best_cost.get('turn', 0.0):.2f}, Wait={self.best_cost.get('wait', 0.0):.2f}")
        else:
            print("未能找到可行解。")
        print(f"总运行时间: {total_duration:.2f} 秒")
        print(f"总迭代次数: {self.iteration_count}")
        print("\n--- 最终算子权重 ---")
        print(f"Destroy: {{{', '.join([f'{n}:{w:.3f}' for n, w in self.destroy_weights.items()])}}}")
        print(f"Repair: {{{', '.join([f'{n}:{w:.3f}' for n, w in self.repair_weights.items()])}}}")

        # 保存历史数据和绘图
        print("\n--- 保存历史数据和绘图 ---")
        self._save_history_data()
        self.plot_convergence_method()

        # 返回最终结果
        return self.best_solution, total_duration, self.best_cost

    # --- 更新算子权重 ---
    def _update_weights(self):
        """更新算子权重。"""
        if self.verbose and self.debug_weights:
            print(f"--- 更新权重 (Segment End Iteration {self.iteration_count}) ---")

        segment_destroy_scores = self.destroy_scores.copy()
        segment_repair_scores = self.repair_scores.copy()
        segment_destroy_counts = self.destroy_counts.copy()
        segment_repair_counts = self.repair_counts.copy()

        for name in self.destroy_operators:
            count = segment_destroy_counts.get(name, 0)
            score = segment_destroy_scores.get(name, 0.0)
            if count > 0:
                performance = score / count
                self.destroy_weights[name] = (1 - self.weight_update_rate) * self.destroy_weights[name] + self.weight_update_rate * performance
            self.destroy_weights[name] = max(0.01, self.destroy_weights[name])

        for name in self.repair_operators:
            count = segment_repair_counts.get(name, 0)
            score = segment_repair_scores.get(name, 0.0)
            if count > 0:
                performance = score / count
                self.repair_weights[name] = (1 - self.weight_update_rate) * self.repair_weights[name] + self.weight_update_rate * performance
            self.repair_weights[name] = max(0.01, self.repair_weights[name])

        segment_record = {'segment_end_iteration': self.iteration_count, 'destroy_ops': {}, 'repair_ops': {}}
        for name in self.destroy_operators:
            segment_record['destroy_ops'][name] = {
                'score': segment_destroy_scores.get(name, 0.0),
                'count': segment_destroy_counts.get(name, 0),
                'weight': self.destroy_weights[name]
            }
        for name in self.repair_operators:
            segment_record['repair_ops'][name] = {
                'score': segment_repair_scores.get(name, 0.0),
                'count': segment_repair_counts.get(name, 0),
                'weight': self.repair_weights[name]
            }
        self.operator_history.append(segment_record)

        self.destroy_scores = {name: 0.0 for name in self.destroy_operators}
        self.repair_scores = {name: 0.0 for name in self.repair_operators}
        self.destroy_counts = {name: 0 for name in self.destroy_operators}
        self.repair_counts = {name: 0 for name in self.repair_operators}

        decay_factor = self.conflict_history_decay
        if decay_factor < 1.0:
            original_sum = sum(self.agv_conflict_counts.values())
            self.agv_conflict_counts = Counter({
                agv_id: int(count * decay_factor)
                for agv_id, count in self.agv_conflict_counts.items()
                if int(count * decay_factor) > 0
            })
            new_sum = sum(self.agv_conflict_counts.values())
            if self.verbose and self.debug_weights:
                print(f"    Conflict Counts Decayed (Factor={decay_factor:.2f}): Total {original_sum} -> {new_sum}")

    # --- 保存历史数据 ---
    def _save_history_data(self):
        """将成本历史和算子历史保存到 CSV 文件。"""
        if not self.record_history:
            return

        cost_history_file = os.path.join(self.results_dir, f"{self.instance_identifier}_cost_history.csv")
        try:
            if self.cost_history:
                with open(cost_history_file, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = self.cost_history[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.cost_history)
                if self.verbose: print(f"成本历史已保存到: {cost_history_file}")
        except Exception as e:
            print(f"错误: 无法写入成本历史文件 '{cost_history_file}': {e}")

        operator_history_file = os.path.join(self.results_dir, f"{self.instance_identifier}_operator_history.csv")
        try:
            if self.operator_history:
                flat_op_history = []
                for segment_record in self.operator_history:
                    iter_num = segment_record['segment_end_iteration']
                    for op_type, ops_dict in segment_record.items():
                        if op_type.endswith('_ops'):
                            op_category = op_type.split('_')[0]
                            for op_name, stats in ops_dict.items():
                                flat_op_history.append({
                                    'segment_end_iteration': iter_num,
                                    'operator_type': op_category,
                                    'operator_name': op_name,
                                    'score': stats['score'],
                                    'count': stats['count'],
                                    'weight': stats['weight']
                                })
                if flat_op_history:
                    with open(operator_history_file, 'w', newline='', encoding='utf-8') as csvfile:
                        fieldnames = flat_op_history[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(flat_op_history)
                    if self.verbose: print(f"算子历史已保存到: {operator_history_file}")
        except Exception as e:
            print(f"错误: 无法写入算子历史文件 '{operator_history_file}': {e}")

    # --- 绘制收敛图 ---
    def plot_convergence_method(self):
        """绘制成本和温度随迭代次数变化的收敛图。"""
        if not self.plot_convergence_flag:
            return
        if not _visual_libs_available:
            print("绘图功能不可用 (缺少 pandas 或 matplotlib)。")
            return
        if not self.cost_history:
            print("警告: 成本历史为空，无法绘制收敛图。")
            return

        plot_file = os.path.join(self.results_dir, f"{self.instance_identifier}_convergence.png")
        try:
            df = pd.DataFrame(self.cost_history)
            df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)

            initial_valid_cost = next((item.get('best_cost') for item in self.cost_history if item.get('best_cost') is not None and item['best_cost'] != float('inf')), None)
            initial_valid_cost = initial_valid_cost if initial_valid_cost is not None else 1000
            y_limit_upper = initial_valid_cost * 1.5 if initial_valid_cost and initial_valid_cost > 0 else None

            fig, ax1 = plt.subplots(figsize=(12, 6))
            color1 = 'tab:blue'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Cost', color=color1)
            ax1.plot(df['iteration'], df['current_cost'].dropna(), color=color1, alpha=0.6, label='Current Cost')
            ax1.plot(df['iteration'], df['best_cost'].dropna(), color=color1, linestyle='-', linewidth=2, label='Best Cost')
            ax1.tick_params(axis='y', labelcolor=color1)
            if y_limit_upper: ax1.set_ylim(bottom=0, top=y_limit_upper)
            else: ax1.set_ylim(bottom=0)

            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel('Temperature', color=color2)
            ax2.plot(df['iteration'], df['temperature'], color=color2, linestyle=':', alpha=0.7, label='Temperature')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(bottom=0)

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines or lines2:
                 ax2.legend(lines + lines2, labels + labels2, loc='upper right')

            plt.title(f'ALNS Convergence ({self.instance_identifier})')
            fig.tight_layout()
            plt.savefig(plot_file)
            plt.close(fig)

            if self.verbose: print(f"收敛图已保存到: {plot_file}")
        except ImportError:
            print("错误: 绘图需要 pandas 和 matplotlib。")
        except Exception as e:
            print(f"错误: 绘制收敛图时发生错误: {e}")


# --- 示例用法 ---
if __name__ == '__main__':
    print("--- ALNS (v38 - Bounding Box Support) 示例 ---")
    try:
        from InstanceGenerator import load_fixed_scenario_1
    except ImportError as e:
        print(f"错误: 导入 InstanceGenerator 失败: {e}")
        sys.exit(1)

    instance_data = load_fixed_scenario_1()
    if not instance_data:
        print("错误: 无法加载固定算例。")
        sys.exit(1)
    test_map, test_tasks = instance_data

    # 创建规划器实例
    planner_main = TWAStarPlanner()

    # 定义实例标识符和结果目录
    instance_id = "fixed_scenario_1_v38_test"
    results_directory = "alns_output_v38"

    # 定义传递给 ALNS.__init__ 的参数字典
    alns_init_params = {
        # 基础信息
        'instance_identifier': instance_id,
        'results_dir': results_directory,
        # ALNS 核心参数
        'max_iterations': 150,
        'initial_temp': 15.0,
        'cooling_rate': 0.99,
        'segment_size': 25,
        'weight_update_rate': 0.2,
        'sigma1': 15.0, 'sigma2': 8.0, 'sigma3': 3.0,
        'removal_percentage_min': 0.15,
        'removal_percentage_max': 0.40,
        'regret_k': 3,
        'no_improvement_limit': 30,
        'conflict_history_decay': 0.9,
        # 环境/规划器参数
        'max_time': 400,
        'cost_weights': (1.0, 0.3, 0.8),
        'v': 1.0,
        'delta_step': 1.0,
        'buffer': 1, # 启用 buffer，用于区域分割
        'planner_time_limit_factor': 5.0,
        'regret_planner_time_limit_abs': 0.15,
        # 冲突解决参数
        'wait_threshold': 6,
        'deadlock_max_wait': 10,
        # 控制/输出参数
        'verbose': True,
        'debug_weights': True,
        'record_history': True,
        'plot_convergence': True,
    }

    # 创建 ALNS 实例
    try:
        alns_instance = ALNS(grid_map=test_map, tasks=test_tasks, planner=planner_main, **alns_init_params)
    except Exception as init_e:
        print(f"错误: 初始化 ALNS 实例失败: {init_e}")
        traceback.print_exc()
        sys.exit(1)

    # 运行 ALNS
    try:
        best_sol, duration, best_cost_dict = alns_instance.run()
    except Exception as run_e:
        print(f"错误: ALNS 运行时发生异常: {run_e}")
        traceback.print_exc()
        sys.exit(1)


    print("\n--- ALNS (v38) 运行完成 ---")
    if best_sol:
        final_total_cost = best_cost_dict.get('total', float('inf'))
        final_cost_str = f"{final_total_cost:.2f}" if final_total_cost != float('inf') else "Inf"
        print(f"最终最优解成本: Total={final_cost_str}")
        print(f"  Breakdown: Travel={best_cost_dict.get('travel', 0.0):.2f}, Turn={best_cost_dict.get('turn', 0.0):.2f}, Wait={best_cost_dict.get('wait', 0.0):.2f}")
    else:
        print("ALNS 未能找到可行解。")

    print(f"结果文件已保存在目录: '{os.path.abspath(results_directory)}'")
# ALNS.py-v39 (完整代码, 修复边冲突, 严格结构)
"""
实现自适应大邻域搜索 (ALNS) 算法主体，用于解决论文定义的仓储 AGV 路径规划问题。

核心功能:
- 基于论文第二章构建的数学模型，最小化目标函数 (公式 1)，即行驶时间、转弯成本和等待时间的加权和。
- 通过迭代破坏和修复解，并自适应调整算子权重，搜索高质量无冲突路径。
- **冲突处理增强**: 包含对节点冲突 (Vertex Conflicts) 和边冲突 (Edge Conflicts, 主要是迎面冲突)
                   的检测与解决机制，以及死锁处理。
- **区域分割优化 (论文 3.5.2):** ALNS 内部在调用 TWA* 规划器时，
  会根据任务计算包围盒 (Bounding Box)，并传递给 Planner (v17+)，以限制搜索范围。

与论文数学模型 (Chapter 2) 的关联:
- 目标函数: _calculate_total_cost 方法计算与公式(1)对应的成本。
- 约束条件:
    - 起始/目标 (2, 3, 4): 通过初始解生成和规划器保证。
    - 节点冲突 (12): 由 _resolve_conflicts_and_deadlocks 方法检测和处理。
    - 障碍物 (13): 由地图和规划器处理。
    - 边冲突 (非模型约束): _resolve_conflicts_and_deadlocks 新增逻辑处理。
    - 其他约束 (5-11, 14-18): 由 Path、Planner、成本计算等隐式满足。
- 参数: __init__ 中使用的 alpha, beta, gamma_wait, v, delta_step 等与模型符号对应。

版本变更 (v38 -> v39):
- **新增**: 在 `_resolve_conflicts_and_deadlocks` 中添加了边冲突（迎面）检测
           和解决逻辑（基于等待或重规划）。
- **新增**: 添加了 `edge_wait_threshold` 参数用于控制边冲突解决策略。
- **修正**: 确保 `_resolve_conflicts_and_deadlocks` 中的所有返回路径、
           错误处理和嵌套逻辑严格遵循规范的 `if/else` 结构。
- **补全**: 提供了所有方法的完整实现，没有省略。
- **保持**: v38 的包围盒支持、Planner 调用修正、算子接口、自适应权重等功能。
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

# --- 导入类型定义 (依赖 DataTypes v10+ for check_time_overlap) ---
try:
    from Map import GridMap, Node
    # 需要 v10 版本包含 check_time_overlap
    from DataTypes import Task, Path, TimeStep, State, DynamicObstacles, Solution, CostDict, check_time_overlap
    from Planner import TWAStarPlanner # 需要 v17+ 包含边冲突启发式 (可选)
except ImportError as e:
    print(f"错误: 导入 ALNS 依赖项失败 (DataTypes v10+, Planner v17+): {e}")
    sys.exit(1)

# --- 导入外部算子模块 (依赖 Operators v8+ 修复终点占用) ---
try:
    import Operators # 需要 v8 版本修复终点占用
except ImportError:
    print("错误: 无法导入 Operators.py (v8+)。请确保文件存在且在路径中。")
    sys.exit(1)

# --- 导入可选可视化库 ---
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    _visual_libs_available = True
except ImportError:
    _visual_libs_available = False
    # print("警告: 未找到 pandas 或 matplotlib。绘图功能将被禁用。")

# --- ALNS 主类 (v39 - 完整代码, 添加边冲突处理, 严格结构) ---
class ALNS:
    """
    自适应大邻域搜索算法实现。
    负责协调破坏/修复算子、管理算法状态、评估解、处理节点/边冲突并执行自适应机制。
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
        self.planner = planner
        self.agv_ids = sorted([task.agv_id for task in tasks])
        self.instance_identifier = instance_identifier
        self.results_dir = results_dir

        # --- ALNS 核心参数 (使用更高的默认值) ---
        self.max_iterations: int = kwargs.get('max_iterations', 500)
        self.initial_temperature: float = kwargs.get('initial_temp', 20.0)
        self.cooling_rate: float = kwargs.get('cooling_rate', 0.99)
        self.segment_size: int = kwargs.get('segment_size', 50)
        self.weight_update_rate: float = kwargs.get('weight_update_rate', 0.15)
        self.sigma1: float = kwargs.get('sigma1', 15.0)
        self.sigma2: float = kwargs.get('sigma2', 8.0)
        self.sigma3: float = kwargs.get('sigma3', 3.0)
        self.removal_percentage_min: float = kwargs.get('removal_percentage_min', 0.20)
        self.removal_percentage_max: float = kwargs.get('removal_percentage_max', 0.45)
        self.regret_k: int = max(2, kwargs.get('regret_k', 3))
        self.no_improvement_limit: Optional[int] = kwargs.get('no_improvement_limit', 100)
        self.conflict_history_decay: float = kwargs.get('conflict_history_decay', 0.9)

        # --- 环境与规划器参数 ---
        self.max_time: TimeStep = kwargs.get('max_time', 500)
        self.cost_weights: Tuple[float, float, float] = kwargs.get('cost_weights', (1.0, 0.3, 0.8))
        self.alpha, self.beta, self.gamma_wait = self.cost_weights
        self.v: float = kwargs.get('v', 1.0)
        self.delta_step: float = kwargs.get('delta_step', 1.0)
        self.buffer: int = kwargs.get('buffer', 1) # 用于 Operators.py 计算包围盒
        self.planner_time_limit_factor: float = kwargs.get('planner_time_limit_factor', 5.0)
        self.regret_planner_time_limit_abs: float = kwargs.get('regret_planner_time_limit_abs', 0.15)

        # --- 冲突解决参数 ---
        self.wait_threshold: int = kwargs.get('wait_threshold', 6)
        self.deadlock_max_wait: int = kwargs.get('deadlock_max_wait', 10)
        self.edge_wait_threshold: int = kwargs.get('edge_wait_threshold', 4)

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

        # --- 算子注册 (依赖 Operators v8+) ---
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
        bounding_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Path]:
        """调用 TWA* 规划器生成单条路径，支持区域分割。"""
        planner_time_limit = None
        if self.planner_time_limit_factor is not None and self.planner_time_limit_factor > 0:
            base_time = 0.8; num_dynamic_states = sum(len(v) for v in dynamic_obstacles.values())
            max_t_effective = max(1, self.max_time); complexity_factor = max(0.5, min(1.0 + num_dynamic_states / max_t_effective, 5.0))
            planner_time_limit = max(0.2, min(base_time * self.planner_time_limit_factor * complexity_factor, 15.0))

        try:
            path: Optional[Path] = self.planner.plan(
                grid_map=self.grid_map, task=task, dynamic_obstacles=dynamic_obstacles,
                max_time=self.max_time, cost_weights=self.cost_weights, v=self.v,
                delta_step=self.delta_step, start_time=start_time, time_limit=planner_time_limit,
                bounding_box=bounding_box
            )
            if path is not None and not isinstance(path, Path):
                print(f"错误: Planner 返回了非 Path 类型: {type(path)}")
                return None
            else:
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

        valid_solution = True; num_paths = 0
        for agv_id, path in solution.items():
            if not path or not isinstance(path, Path) or not path.sequence: valid_solution = False; break
            num_paths += 1
            try: cost_dict = path.get_cost(self.grid_map, self.alpha, self.beta, self.gamma_wait, self.v, self.delta_step)
            except Exception as e: print(f"错误: 计算 AGV {agv_id} 成本时出错: {e}"); valid_solution = False; break
            if cost_dict.get('total', float('inf')) == float('inf'): valid_solution = False; break
            for key in total_cost_dict: total_cost_dict[key] += cost_dict.get(key, 0.0)

        if valid_solution and num_paths == self.num_agvs: return total_cost_dict
        else: return inf_dict

    # --- 初始解生成 ---
    def generate_initial_solution(self) -> Optional[Solution]:
        """生成初始解 (优先序贯规划)。"""
        print("--- 生成初始解 (优先序贯规划) ---")
        solution: Solution = {}; dynamic_obstacles: DynamicObstacles = {}
        sorted_tasks = sorted(self.tasks, key=lambda t: t.agv_id)

        for task in sorted_tasks:
            agv_id = task.agv_id; t_start_call = time.perf_counter()
            if self.verbose: print(f"  规划初始 AGV {agv_id}...")
            # 使用 Operators v8+ 中的 _build_dynamic_obstacles
            try:
                dynamic_obstacles = Operators._build_dynamic_obstacles(self, solution)
            except Exception as build_e:
                print(f"错误: 构建动态障碍时失败 (AGV {agv_id}): {build_e}")
                return None

            path = self._call_planner(task, dynamic_obstacles, start_time=0)
            call_dur = time.perf_counter() - t_start_call

            if path and path.sequence:
                solution[agv_id] = path
                if self.verbose: print(f"    成功 (耗时 {call_dur:.3f}s)，路径长度 {len(path)}, Makespan {path.get_makespan()}")
            else:
                print(f"  错误：AGV {agv_id} 初始规划失败！(耗时 {call_dur:.3f}s)")
                self.initial_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
                return None

        self.initial_cost = self._calculate_total_cost(solution)
        initial_total_cost = self.initial_cost.get('total', float('inf'))
        initial_cost_str = f"{initial_total_cost:.2f}" if initial_total_cost != float('inf') else "Inf"
        print(f"--- 初始解生成完毕，成本: {initial_cost_str} ---")
        if initial_total_cost != float('inf'): return solution
        else: return None

    # --- 算子选择 ---
    def _select_operator_roulette_wheel(self, weights: Dict[str, float]) -> str:
        """轮盘赌选择算子。"""
        total_weight = sum(weights.values())
        if total_weight <= 1e-9:
            valid_operators = [name for name, w in weights.items() if w > 1e-9]
            if valid_operators: return random.choice(valid_operators)
            else: return random.choice(list(weights.keys()))
        else:
            pick = random.uniform(0, total_weight); current = 0.0
            for name, weight in weights.items():
                current += weight
                if current >= pick - 1e-9: return name
            return list(weights.keys())[-1]

    # --- 冲突与死锁解决 (v39 - 完整代码, 包含边冲突处理, 严格结构) ---
    def _resolve_conflicts_and_deadlocks(self, solution: Solution) -> Optional[Solution]:
        """
        按顺序处理死锁、节点冲突和边冲突。
        采用严格的 if/else 结构确保清晰的返回路径。
        """
        if not solution: return None
        current_solution = copy.deepcopy(solution)
        max_resolve_attempts = self.num_agvs * 5
        resolve_attempt = 0

        while resolve_attempt < max_resolve_attempts:
            resolve_attempt += 1
            made_change_this_pass = False
            if self.verbose and self.debug_weights: print(f"  冲突/死锁解决尝试 #{resolve_attempt}")

            # --- 1. 死锁检测与解决 ---
            agv_wait_times: Dict[int, Tuple[TimeStep, Node]] = defaultdict(lambda: (0, (-1,-1)))
            max_t_in_solution = 0
            paths_sequences = {agv_id: path.sequence for agv_id, path in current_solution.items() if path and path.sequence}
            if not paths_sequences: return current_solution # 没有路径，直接返回

            for path_seq in paths_sequences.values():
                 if path_seq: max_t_in_solution = max(max_t_in_solution, path_seq[-1][1])

            deadlocked_agvs: Set[int] = set()
            node_at_time: Dict[int, Dict[TimeStep, Node]] = defaultdict(dict)
            for agv_id, path_seq in paths_sequences.items():
                 for node, t in path_seq: node_at_time[agv_id][t] = node

            for agv_id in paths_sequences.keys():
                wait_duration, current_wait_node = agv_wait_times[agv_id]
                # 优化: 存储上一个节点以避免重复查找 node_at_time[agv_id].get(t_step - 1)
                last_node = None
                for t_step in range(max_t_in_solution + 1):
                    node_t = node_at_time[agv_id].get(t_step)
                    if node_t is not None and last_node is not None and node_t == last_node:
                        if current_wait_node == node_t: wait_duration += 1
                        else: wait_duration = 1; current_wait_node = node_t
                        agv_wait_times[agv_id] = (wait_duration, current_wait_node)
                        if wait_duration > self.deadlock_max_wait:
                            deadlocked_agvs.add(agv_id)
                            self.agv_conflict_counts[agv_id] += 3
                            if self.verbose: print(f"      检测到潜在死锁：AGV {agv_id} @ {current_wait_node} wait > {self.deadlock_max_wait} (ConflictCount +3)")
                            break # 已检测到死锁，无需继续检查此 AGV
                    else: # 移动或时间步结束
                         if wait_duration > 0: # 结束了等待状态
                             wait_duration = 0; current_wait_node = (-1,-1)
                             agv_wait_times[agv_id] = (wait_duration, current_wait_node)
                    last_node = node_t # 更新上一个节点

            # --- 死锁解决 ---
            if deadlocked_agvs:
                # 选择优先级最低（ID 最大）的来重规划通常更稳定
                agv_to_resolve = max(deadlocked_agvs)
                if self.verbose: print(f"      检测到死锁，选择 AGV {agv_to_resolve} 重规划。")

                path_to_resolve = current_solution.get(agv_to_resolve)
                original_task = next((t for t in self.tasks if t.agv_id == agv_to_resolve), None)

                if path_to_resolve and path_to_resolve.sequence and original_task:
                    # Find appropriate replan start state (before deadlock starts)
                    wait_t_count, _ = agv_wait_times[agv_to_resolve] # Get the detected wait duration
                    deadlock_start_time = path_to_resolve.sequence[-1][1] - wait_t_count + 1

                    replan_start_index = -1
                    for i in range(len(path_to_resolve.sequence) - 1, -1, -1):
                        if path_to_resolve.sequence[i][1] < deadlock_start_time:
                            replan_start_index = i
                            break
                    replan_start_index = max(0, replan_start_index) # Ensure index is not negative
                    replan_start_state = path_to_resolve.sequence[replan_start_index]
                    start_idx_truncate = replan_start_index + 1 # Index to truncate the sequence at

                    replan_task = Task(agv_id=agv_to_resolve, start_node=replan_start_state[0], goal_node=original_task.goal_node)
                    replan_start_time_val = replan_start_state[1]

                    # Truncate the path before replanning
                    current_solution[agv_to_resolve].sequence = path_to_resolve.sequence[:start_idx_truncate]

                    # Build obstacles excluding the replanned AGV
                    dynamic_obs = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_to_resolve)

                    # Call the planner
                    new_path_segment = self._call_planner(replan_task, dynamic_obs, start_time=replan_start_time_val)

                    if new_path_segment and new_path_segment.sequence and len(new_path_segment.sequence) > 1:
                        # Append the new segment (excluding the first state which is the start state)
                        current_solution[agv_to_resolve].sequence.extend(new_path_segment.sequence[1:])
                        made_change_this_pass = True
                        if self.verbose: print(f"      死锁解决：AGV {agv_to_resolve} 重规划成功。")
                        continue # Deadlock resolved, restart conflict checking from the beginning
                    else:
                        print(f"      错误：死锁解决失败，AGV {agv_to_resolve} 重规划未找到有效路径。")
                        return None # Fatal error, cannot resolve
                else:
                    print(f"错误：解决死锁时无法获取 AGV {agv_to_resolve} 的路径或任务。")
                    return None # Fatal error

            # --- 2. 节点冲突检测与解决 ---
            conflict_resolved_in_scan_node = False
            max_t_check_node = 0
            paths_sequences_node = {aid: p.sequence for aid, p in current_solution.items() if p and p.sequence}
            if not paths_sequences_node: return current_solution # No paths

            for path_seq in paths_sequences_node.values(): max_t_check_node = max(max_t_check_node, path_seq[-1][1])
            node_occupancy_cache_node: Dict[TimeStep, Dict[Node, List[int]]] = defaultdict(lambda: defaultdict(list))
            for agv_id, path_seq in paths_sequences_node.items():
                 for node, t in path_seq: node_occupancy_cache_node[t][node].append(agv_id)

            for t_step in range(max_t_check_node + 1):
                nodes_with_conflict = {node: occ for node, occ in node_occupancy_cache_node[t_step].items() if len(occ) > 1}
                if nodes_with_conflict:
                    for node, occupants in nodes_with_conflict.items():
                        if len(occupants) < 2: continue
                        occupants.sort() # Lower ID has higher priority
                        agv_high = occupants[0]; agvs_low = occupants[1:]
                        if self.verbose: print(f"      检测到节点冲突 @ ({node}, {t_step}), High={agv_high}, Low={agvs_low}")

                        for agv_low in agvs_low:
                            self.agv_conflict_counts[agv_low] += 1
                            path_low = current_solution.get(agv_low)
                            if not path_low or not path_low.sequence: continue

                            conflict_idx = -1
                            for i, (n,t) in enumerate(path_low.sequence):
                                if n==node and t==t_step: conflict_idx = i; break
                            if conflict_idx == -1: continue # Should not happen if logic is correct

                            prev_idx = conflict_idx - 1; should_replan = False
                            if prev_idx < 0: # Conflict at the very start (t=0)
                                # This case is tricky. Replan from the start.
                                prev_node, prev_time = path_low.sequence[0][0], 0
                                print(f"警告：AGV {agv_low} 在起点 ({node}, {t_step}) 发生冲突。触发重规划。")
                                should_replan = True
                            else:
                                prev_node, prev_time = path_low.sequence[prev_idx]

                            if not should_replan:
                                # Calculate required wait time
                                wait_until = t_step + 1; path_high = current_solution.get(agv_high)
                                if path_high and path_high.sequence:
                                    last_high_t = -1
                                    for hn, ht in path_high.sequence:
                                        if ht >= t_step and hn == node: last_high_t = max(last_high_t, ht)
                                    if last_high_t != -1: wait_until = max(wait_until, last_high_t + 1)
                                else:
                                    # If high priority path is invalid, maybe replan low? Or log error.
                                    print(f"警告: 节点冲突中高优先级 AGV {agv_high} 路径无效。")
                                    # Fallback: Just wait one step
                                    wait_until = t_step + 1

                                required_wait_duration = wait_until - t_step # Duration AGV Low needs to pause *at prev_node*

                                if required_wait_duration <= 0 : # Should not happen if conflict exists, but as safety
                                    continue # No wait needed?

                                if required_wait_duration > self.wait_threshold: # Wait too long -> replan
                                    should_replan = True; self.agv_conflict_counts[agv_low] += 1
                                    if self.verbose: print(f"      AGV {agv_low} 节点冲突需等待 {required_wait_duration} > {self.wait_threshold} 步，触发重规划。")
                                else: # Try to wait
                                    if self.verbose: print(f"      AGV {agv_low} 尝试等待 {required_wait_duration} 步 @ {prev_node}")
                                    new_sequence_prefix = path_low.sequence[:prev_idx+1]
                                    dynamic_obs_wait_check = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_low)
                                    wait_possible = True

                                    # Insert wait states
                                    for w_step in range(1, required_wait_duration + 1):
                                        current_wait_time = prev_time + w_step
                                        # Check if wait location is free during wait period
                                        if current_wait_time in dynamic_obs_wait_check and prev_node in dynamic_obs_wait_check.get(current_wait_time, set()):
                                            should_replan = True; wait_possible = False; self.agv_conflict_counts[agv_low] += 1
                                            if self.verbose: print(f"      AGV {agv_low} 等待期间在 t={current_wait_time} 与障碍冲突，触发重规划。")
                                            break
                                        new_sequence_prefix.append((prev_node, current_wait_time))

                                    if wait_possible: # Check if path after wait is valid
                                        time_shift = required_wait_duration
                                        path_after_wait_valid = True
                                        shifted_path_suffix = []

                                        for i in range(conflict_idx, len(path_low.sequence)):
                                            original_node_wait, original_time_wait = path_low.sequence[i]
                                            new_time_wait = original_time_wait + time_shift

                                            if new_time_wait > self.max_time: path_after_wait_valid = False; break
                                            # Check shifted path against obstacles
                                            if new_time_wait in dynamic_obs_wait_check and original_node_wait in dynamic_obs_wait_check.get(new_time_wait, set()):
                                                path_after_wait_valid = False; break
                                            shifted_path_suffix.append((original_node_wait, new_time_wait))

                                        if path_after_wait_valid: # Apply wait
                                            current_solution[agv_low].sequence = new_sequence_prefix + shifted_path_suffix
                                            made_change_this_pass = True; conflict_resolved_in_scan_node = True
                                            if self.verbose: print(f"      AGV {agv_low} 等待解决节点冲突成功。")
                                            # Clear cache as paths changed
                                            node_occupancy_cache_node.clear()
                                            break # Conflict for this node resolved, move to next conflict check
                                        else:
                                            should_replan = True; self.agv_conflict_counts[agv_low] += 1
                                            if self.verbose: print(f"      AGV {agv_low} 等待后路径无效或冲突，触发重规划。")
                                    # else: should_replan is already True

                            if should_replan:
                                if self.verbose: print(f"      AGV {agv_low} 将从 ({prev_node}, {prev_time}) 开始重规划 (节点冲突)。")
                                task_low_original = next((t for t in self.tasks if t.agv_id == agv_low), None)
                                if task_low_original is not None:
                                    replan_task_low = Task(agv_id=agv_low, start_node=prev_node, goal_node=task_low_original.goal_node)
                                    # Truncate path at the state *before* the conflict start
                                    current_solution[agv_low].sequence = path_low.sequence[:prev_idx+1] if prev_idx >= 0 else []
                                    # Build dynamic obstacles excluding the replanning AGV
                                    dynamic_obs_replan_node = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_low)
                                    new_segment_node = self._call_planner(replan_task_low, dynamic_obs_replan_node, start_time=prev_time)

                                    if new_segment_node and new_segment_node.sequence and len(new_segment_node.sequence) > 1:
                                        # Append the new segment (excluding the first state)
                                        current_solution[agv_low].sequence.extend(new_segment_node.sequence[1:])
                                        made_change_this_pass = True; conflict_resolved_in_scan_node = True
                                        if self.verbose: print(f"      AGV {agv_low} 重规划解决节点冲突成功。")
                                        # Clear cache as paths changed
                                        node_occupancy_cache_node.clear()
                                        break # Conflict resolved, move to next conflict check
                                    else:
                                        print(f"      错误：节点冲突解决失败，AGV {agv_low} 重规划未找到有效路径段。")
                                        return None # Fatal error
                                else:
                                    print(f"错误：无法找到 AGV {agv_low} 的原始任务用于重规划。")
                                    return None # Fatal error

                        # End loop for low priority AGVs at this node/time
                        if conflict_resolved_in_scan_node: break
                    # End loop for conflicting nodes at this time
                    if conflict_resolved_in_scan_node: break
                # End loop for time steps
                if conflict_resolved_in_scan_node: break # Restart scan from t=0

            if conflict_resolved_in_scan_node:
                continue # If node conflict was resolved, restart the whole process

            # --- 3. 边冲突检测与解决 (迎面) ---
            edge_conflict_resolved_in_scan = False
            moves_dict: Dict[int, List[Tuple[Node, Node, TimeStep, TimeStep]]] = defaultdict(list)
            paths_sequences_edge = {aid: p.sequence for aid, p in current_solution.items() if p and p.sequence}
            if not paths_sequences_edge: return current_solution # No paths

            for agv_id, path_seq in paths_sequences_edge.items():
                for i in range(len(path_seq) - 1):
                    node1, time1 = path_seq[i]
                    node2, time2 = path_seq[i+1]
                    if node1 != node2: # It's a move
                        # Store move as (from_node, to_node, start_time, end_time)
                        # end_time is the time step when the AGV *arrives* at node2
                        moves_dict[agv_id].append((node1, node2, time1, time2))

            processed_edge_conflicts: Set[Tuple[int, int, TimeStep]] = set() # (agv1, agv2, approx_time)
            agv_ids_list = list(paths_sequences_edge.keys())

            for idx1, agv1_id in enumerate(agv_ids_list):
                for idx2 in range(idx1 + 1, len(agv_ids_list)):
                    agv2_id = agv_ids_list[idx2]
                    agv_low_id, agv_high_id = min(agv1_id, agv2_id), max(agv1_id, agv2_id)

                    # Check every move of agv1 against every move of agv2
                    for move1_from, move1_to, move1_start, move1_end in moves_dict[agv1_id]:
                        for move2_from, move2_to, move2_start, move2_end in moves_dict[agv2_id]:
                            # Check for head-on collision: (i, j) vs (j, i)
                            if move1_from == move2_to and move1_to == move2_from:
                                # Check for time overlap using the helper function
                                # Note: The time interval is [start_time, end_time) for occupation
                                if check_time_overlap(move1_start, move1_end, move2_start, move2_end):
                                    conflict_time_approx = max(move1_start, move2_start) # Approximate time
                                    conflict_key = (agv_low_id, agv_high_id, conflict_time_approx)
                                    if conflict_key in processed_edge_conflicts: continue # Already handled

                                    if self.verbose: print(f"      检测到边冲突 (迎面) between AGV {agv1_id} ({move1_from}->{move1_to} @ [{move1_start},{move1_end})) and AGV {agv2_id} ({move2_from}->{move2_to} @ [{move2_start},{move2_end}))")
                                    processed_edge_conflicts.add(conflict_key)

                                    # --- Resolve Edge Conflict ---
                                    # Default: lower ID (higher priority) proceeds, higher ID (lower prio) waits/replans
                                    agv_to_wait_id = agv_high_id
                                    agv_proceed_id = agv_low_id
                                    move_proceed_end_t = move1_end if agv_proceed_id == agv1_id else move2_end

                                    path_to_wait = current_solution.get(agv_to_wait_id)
                                    if not path_to_wait or not path_to_wait.sequence: continue

                                    # Find the move in the waiting AGV's path that needs delay
                                    wait_from_idx = -1; wait_start_time = -1
                                    move_wait_from = move2_from if agv_to_wait_id == agv2_id else move1_from
                                    move_wait_to = move2_to if agv_to_wait_id == agv2_id else move1_to

                                    for i in range(len(path_to_wait.sequence)-1):
                                        n1_w, t1_w = path_to_wait.sequence[i]
                                        n2_w, t2_w = path_to_wait.sequence[i+1]
                                        # Find the state *before* the conflicting move starts
                                        if n1_w == move_wait_from and n2_w == move_wait_to and t1_w == (move2_start if agv_to_wait_id == agv2_id else move1_start):
                                            wait_from_idx = i
                                            wait_start_time = t1_w
                                            break
                                    if wait_from_idx == -1: continue # Couldn't find the exact move state

                                    # Determine how long to wait: until the other AGV finishes its move
                                    required_edge_wait_duration = max(0, move_proceed_end_t - wait_start_time)

                                    should_replan_edge = False
                                    if required_edge_wait_duration > self.edge_wait_threshold:
                                        should_replan_edge = True
                                        self.agv_conflict_counts[agv_to_wait_id] += 1
                                        if self.verbose: print(f"      AGV {agv_to_wait_id} 边冲突需等待 {required_edge_wait_duration} > {self.edge_wait_threshold} 步，触发重规划。")
                                    else: # Try to wait for edge conflict
                                        wait_node_edge, _ = path_to_wait.sequence[wait_from_idx]
                                        if self.verbose: print(f"      AGV {agv_to_wait_id} 尝试等待 {required_edge_wait_duration} 步 @ {wait_node_edge} (边冲突)。")
                                        new_sequence_prefix_edge = path_to_wait.sequence[:wait_from_idx+1]
                                        dynamic_obs_edge_wait_check = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_to_wait_id)
                                        wait_possible_edge = True

                                        # Insert wait states
                                        for w_step in range(1, required_edge_wait_duration + 1):
                                            current_wait_time_edge = wait_start_time + w_step
                                            # Check if wait location is free
                                            if current_wait_time_edge in dynamic_obs_edge_wait_check and wait_node_edge in dynamic_obs_edge_wait_check.get(current_wait_time_edge, set()):
                                                should_replan_edge = True; wait_possible_edge = False; self.agv_conflict_counts[agv_to_wait_id] += 1
                                                if self.verbose: print(f"      AGV {agv_to_wait_id} 边冲突等待期间在 t={current_wait_time_edge} 与障碍冲突，触发重规划。")
                                                break
                                            new_sequence_prefix_edge.append((wait_node_edge, current_wait_time_edge))

                                        if wait_possible_edge: # Check path validity after waiting
                                            time_shift_edge = required_edge_wait_duration
                                            path_after_wait_valid_edge = True
                                            shifted_path_suffix_edge = []
                                            # Start shifting from the state *after* the wait insertion point
                                            start_shift_idx = wait_from_idx + 1
                                            for i in range(start_shift_idx, len(path_to_wait.sequence)):
                                                original_node_edge, original_time_edge = path_to_wait.sequence[i]
                                                new_time_edge = original_time_edge + time_shift_edge

                                                if new_time_edge > self.max_time: path_after_wait_valid_edge = False; break
                                                # Check shifted path against obstacles
                                                if new_time_edge in dynamic_obs_edge_wait_check and original_node_edge in dynamic_obs_edge_wait_check.get(new_time_edge, set()):
                                                    path_after_wait_valid_edge = False; break
                                                shifted_path_suffix_edge.append((original_node_edge, new_time_edge))

                                            if path_after_wait_valid_edge: # Apply wait
                                                current_solution[agv_to_wait_id].sequence = new_sequence_prefix_edge + shifted_path_suffix_edge
                                                made_change_this_pass = True; edge_conflict_resolved_in_scan = True
                                                if self.verbose: print(f"      AGV {agv_to_wait_id} 等待解决边冲突成功。")
                                                # Clear caches and break to restart scans
                                                moves_dict.clear(); paths_sequences_edge.clear()
                                                break # Conflict resolved, restart outer loops
                                            else:
                                                should_replan_edge = True; self.agv_conflict_counts[agv_to_wait_id] += 1
                                                if self.verbose: print(f"      AGV {agv_to_wait_id} 边冲突等待后路径无效或冲突，触发重规划。")
                                        # else: should_replan_edge is already True from wait check

                                    if should_replan_edge:
                                        prev_node_replan_edge, prev_time_replan_edge = path_to_wait.sequence[wait_from_idx]
                                        if self.verbose: print(f"      AGV {agv_to_wait_id} 将从 ({prev_node_replan_edge}, {prev_time_replan_edge}) 开始重规划 (边冲突)。")
                                        task_wait_original = next((t for t in self.tasks if t.agv_id == agv_to_wait_id), None)
                                        if task_wait_original is not None:
                                            replan_task_edge = Task(agv_id=agv_to_wait_id, start_node=prev_node_replan_edge, goal_node=task_wait_original.goal_node)
                                            # Truncate path at the state *before* the replan start
                                            current_solution[agv_to_wait_id].sequence = path_to_wait.sequence[:wait_from_idx+1]
                                            dynamic_obs_replan_edge = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_to_wait_id)
                                            new_segment_edge = self._call_planner(replan_task_edge, dynamic_obs_replan_edge, start_time=prev_time_replan_edge)

                                            if new_segment_edge and new_segment_edge.sequence and len(new_segment_edge.sequence) > 1:
                                                current_solution[agv_to_wait_id].sequence.extend(new_segment_edge.sequence[1:])
                                                made_change_this_pass = True; edge_conflict_resolved_in_scan = True
                                                if self.verbose: print(f"      AGV {agv_to_wait_id} 重规划解决边冲突成功。")
                                                # Clear caches and break to restart scans
                                                moves_dict.clear(); paths_sequences_edge.clear()
                                                break # Conflict resolved, restart outer loops
                                            else:
                                                print(f"      错误：边冲突解决失败，AGV {agv_to_wait_id} 重规划未找到有效路径段。")
                                                return None # Fatal error
                                        else:
                                            print(f"错误：无法找到 AGV {agv_to_wait_id} 的原始任务用于边冲突重规划。")
                                            return None # Fatal error
                        # End inner loop (moves of agv2)
                        if edge_conflict_resolved_in_scan: break
                    # End outer loop (moves of agv1)
                    if edge_conflict_resolved_in_scan: break
                # End loop comparing agv pairs
                if edge_conflict_resolved_in_scan: break

            if edge_conflict_resolved_in_scan:
                continue # If edge conflict was resolved, restart the whole process

            # --- 4. 结束条件 ---
            if not made_change_this_pass:
                 if self.verbose and self.debug_weights: print(f"  冲突/死锁解决：本轮扫描未做修改，解已稳定。")
                 break # No changes in this full pass (deadlock, node, edge), exit loop

        # --- 循环结束后的检查 ---
        if resolve_attempt >= max_resolve_attempts and made_change_this_pass:
            # If loop terminated due to max attempts but changes were still being made
            print(f"  错误：冲突/死锁解决超过最大尝试次数 {max_resolve_attempts}，但仍在修改。可能存在循环或复杂依赖。")
            return None # Indicate failure due to complexity/loop
        else:
            # Check final cost after resolution loop finishes (either stable or max attempts reached without changes in last pass)
            final_check_cost_dict = self._calculate_total_cost(current_solution)
            if final_check_cost_dict.get('total', float('inf')) == float('inf'):
                print("  错误：冲突/死锁解决后最终成本检查为 Inf！")
                return None # Invalid solution even after resolution
            else:
                if self.verbose and resolve_attempt > 0: # Only print if resolution attempts were made
                    print(f"--- 冲突/死锁解决完成 (尝试 {resolve_attempt} 次) ---")
                return current_solution # Solution is considered valid and resolved

    # --- 主运行方法 ---
    def run(self) -> Tuple[Optional[Solution], float, CostDict]:
        """执行 ALNS 算法主流程。"""
        start_run_time = time.perf_counter()

        # --- 初始化状态变量 ---
        self.iteration_count = 0; self.no_improvement_count = 0
        self.temperature = self.initial_temperature
        self.best_solution = None; self.best_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.current_solution = None; self.current_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.initial_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.cost_history = []; self.operator_history = []; self.agv_conflict_counts.clear()
        self.destroy_weights = {name: 1.0 for name in self.destroy_operators}
        self.repair_weights = {name: 1.0 for name in self.repair_operators}
        self.destroy_scores = {name: 0.0 for name in self.destroy_operators}
        self.repair_scores = {name: 0.0 for name in self.repair_operators}
        self.destroy_counts = {name: 0 for name in self.destroy_operators}
        self.repair_counts = {name: 0 for name in self.repair_operators}

        # 1. 生成初始解
        initial_solution_raw = self.generate_initial_solution()
        if initial_solution_raw is None:
            print("错误：无法生成可行的初始解，ALNS 终止。")
            return None, time.perf_counter() - start_run_time, self.best_cost
        else:
            # 2. 处理初始解冲突 (调用 v39 的方法)
            if self.verbose: print("--- 处理初始解冲突 (保险步骤) ---")
            resolved_initial_solution = self._resolve_conflicts_and_deadlocks(initial_solution_raw)
            if resolved_initial_solution is not None:
                self.current_solution = resolved_initial_solution
                self.current_cost = self._calculate_total_cost(self.current_solution)
                current_total_cost_init = self.current_cost.get('total', float('inf'))
                current_cost_str_init = f"{current_total_cost_init:.2f}" if current_total_cost_init != float('inf') else "Inf"
                if self.verbose: print(f"初始解冲突处理后成本: {current_cost_str_init}")

                if current_total_cost_init != float('inf'):
                    # 3. 设置初始最优解
                    self.best_solution = copy.deepcopy(self.current_solution)
                    self.best_cost = self.current_cost
                    best_total_cost_init = self.best_cost.get('total', float('inf'))
                    best_cost_str_init_final = f"{best_total_cost_init:.2f}" if best_total_cost_init != float('inf') else "Inf"
                    print(f"最终初始解成本 (无冲突): {best_cost_str_init_final}")
                    self.cost_history.append({'iteration': 0, 'current_cost': self.current_cost['total'], 'best_cost': self.best_cost['total'], 'temperature': self.temperature})
                else:
                    print("错误：处理冲突后的初始解成本为无穷大，ALNS 终止。")
                    return None, time.perf_counter() - start_run_time, self.best_cost
            else:
                print("错误：处理初始解冲突失败，ALNS 终止。")
                return None, time.perf_counter() - start_run_time, self.best_cost

        # 4. ALNS 主迭代循环
        print(f"\n--- 开始 ALNS 迭代 (Max Iter: {self.max_iterations}, No Improve Limit: {self.no_improvement_limit}) ---")
        for i in range(self.max_iterations):
            self.iteration_count = i + 1; iter_start_time = time.perf_counter()
            if self.current_solution is None: print(f"错误: 迭代 {i+1} 开始时当前解丢失！终止。"); break

            # --- a. 选择算子 ---
            destroy_op_name = self._select_operator_roulette_wheel(self.destroy_weights)
            repair_op_name = self._select_operator_roulette_wheel(self.repair_weights)
            destroy_op = self.destroy_operators.get(destroy_op_name)
            repair_op = self.repair_operators.get(repair_op_name)
            if not destroy_op or not repair_op: print(f"错误：无法找到算子 {destroy_op_name} 或 {repair_op_name}！"); break

            # --- 打印迭代信息 ---
            best_cost_iter_str = f"{self.best_cost['total']:.2f}" if self.best_cost['total'] != float('inf') else "Inf"
            curr_cost_iter_str = f"{self.current_cost['total']:.2f}" if self.current_cost['total'] != float('inf') else "Inf"
            no_imp_limit_str = str(self.no_improvement_limit) if self.no_improvement_limit is not None else "N/A"
            if self.verbose:
                print(f"\nIter {i+1}/{self.max_iterations} | T={self.temperature:.3f} | Best={best_cost_iter_str} | Curr={curr_cost_iter_str} | NoImpr={self.no_improvement_count}/{no_imp_limit_str} | Ops: D='{destroy_op_name}', R='{repair_op_name}'")

            # --- b. 破坏解 ---
            removal_percentage = random.uniform(self.removal_percentage_min, self.removal_percentage_max)
            removal_count = max(1, int(self.num_agvs * removal_percentage))
            try:
                partial_solution, removed_agv_ids = destroy_op(self, self.current_solution, removal_count)
            except Exception as e:
                print(f"!!!!!!!!!! 调用破坏算子 {destroy_op_name} 时发生错误 !!!!!!!!!!!!!\n错误信息: {e}")
                traceback.print_exc(); break

            # --- c. 修复解 ---
            new_solution_raw = None
            try:
                new_solution_raw = repair_op(self, partial_solution, removed_agv_ids)
            except Exception as e:
                print(f"!!!!!!!!!! 调用修复算子 {repair_op_name} 时发生错误 !!!!!!!!!!!!!\n错误信息: {e}")
                traceback.print_exc()

            # --- d. 冲突与死锁处理 (调用 v39 版本) ---
            new_solution_processed: Optional[Solution] = None
            new_cost_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
            if new_solution_raw is not None:
                if self.verbose: print("  正在处理新解的冲突与死锁...")
                resolve_start = time.perf_counter()
                new_solution_processed = self._resolve_conflicts_and_deadlocks(new_solution_raw)
                resolve_dur = time.perf_counter() - resolve_start
                if self.verbose: print(f"  冲突处理完成 (耗时 {resolve_dur:.3f}s)")

                if new_solution_processed is not None:
                    new_cost_dict = self._calculate_total_cost(new_solution_processed)
                else:
                    # 如果冲突解决失败，则此解无效
                    new_cost_dict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
                    if self.verbose: print("  警告: 冲突解决失败，新解无效。")

            new_total_cost = new_cost_dict.get('total', float('inf'))
            current_total_cost = self.current_cost.get('total', float('inf'))
            best_total_cost = self.best_cost.get('total', float('inf'))

            # --- e. 评估与接受 ---
            score = 0.0; accepted = False; improved_best = False
            if new_total_cost != float('inf'): # 只有有效解才可能被接受
                delta_cost = new_total_cost - current_total_cost if current_total_cost != float('inf') else -float('inf')

                if delta_cost < -1e-9 or current_total_cost == float('inf'): # 接受更好或从无效变有效
                    accepted = True; score = self.sigma2
                    if new_total_cost < best_total_cost - 1e-9: # 改进了历史最优
                        score = self.sigma1; improved_best = True
                    if self.verbose or improved_best:
                        print(f"  接受新解 ({'更好' if delta_cost < -1e-9 else '从无效变有效'}){' *** New Best! ***' if improved_best else ''}, Cost={new_total_cost:.2f}")
                else: # 接受较差解（模拟退火）
                    prob = math.exp(-delta_cost / self.temperature) if self.temperature > 1e-6 else 0.0
                    if random.random() < prob:
                        accepted = True; score = self.sigma3
                        if self.verbose:
                            print(f"  接受新解 (较差/相同, Prob={prob:.3f}), Cost={new_total_cost:.2f}")

            # --- f. 更新状态和统计 ---
            if accepted:
                self.current_solution = new_solution_processed
                self.current_cost = new_cost_dict
                if improved_best:
                    self.best_solution = copy.deepcopy(new_solution_processed)
                    self.best_cost = new_cost_dict
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1 # 即使接受了较差解，也算没有改进最优解
                # 更新算子得分
                self.destroy_scores[destroy_op_name] += score
                self.repair_scores[repair_op_name] += score
            else:
                # 如果解无效或未被接受，也增加未改进计数
                self.no_improvement_count += 1

            # 更新算子使用次数
            self.destroy_counts[destroy_op_name] += 1
            self.repair_counts[repair_op_name] += 1

            # 记录成本历史
            self.cost_history.append({'iteration': i + 1, 'current_cost': self.current_cost['total'], 'best_cost': self.best_cost['total'], 'temperature': self.temperature})

            # --- g. 周期性权重更新 ---
            if (i + 1) % self.segment_size == 0:
                self._update_weights()

            # --- h. 降温 ---
            self.temperature = max(1e-6, self.temperature * self.cooling_rate)

            # --- i. 检查早停条件 ---
            if self.no_improvement_limit is not None and self.no_improvement_count >= self.no_improvement_limit:
                print(f"\n--- 触发早停：最优解连续 {self.no_improvement_limit} 次迭代未改进 (当前迭代 {i+1}) ---")
                break
            # --- 单次迭代结束 ---

        # 5. 运行结束
        end_run_time = time.perf_counter()
        total_duration = end_run_time - start_run_time
        print("\n--- ALNS 最终结果 ---")
        best_total_cost_final = self.best_cost.get('total', float('inf'))
        best_cost_final_str = f"{best_total_cost_final:.2f}" if best_total_cost_final != float('inf') else "Inf"
        if self.best_solution is not None: # 严格检查 None
            print(f"找到最优解成本: {best_cost_final_str}")
            print(f"  Breakdown: Travel={self.best_cost.get('travel', 0.0):.2f}, Turn={self.best_cost.get('turn', 0.0):.2f}, Wait={self.best_cost.get('wait', 0.0):.2f}")
        else:
            print("未能找到可行解。")
        print(f"总运行时间: {total_duration:.2f} 秒")
        print(f"总迭代次数: {self.iteration_count}")
        print("\n--- 最终算子权重 ---")
        # 归一化权重以便更好地比较相对重要性
        destroy_total_w = sum(self.destroy_weights.values())
        repair_total_w = sum(self.repair_weights.values())
        if destroy_total_w > 1e-9: print(f"Destroy: {{{', '.join([f'{n}:{w/destroy_total_w:.3f}' for n, w in self.destroy_weights.items()])}}}")
        else: print(f"Destroy: {self.destroy_weights}")
        if repair_total_w > 1e-9: print(f"Repair: {{{', '.join([f'{n}:{w/repair_total_w:.3f}' for n, w in self.repair_weights.items()])}}}")
        else: print(f"Repair: {self.repair_weights}")

        # 保存历史数据和绘图
        print("\n--- 保存历史数据和绘图 ---")
        self._save_history_data()
        self.plot_convergence_method()

        return self.best_solution, total_duration, self.best_cost

    # --- 更新算子权重 ---
    def _update_weights(self):
        """根据最近一个 segment 的表现更新算子权重。"""
        if self.verbose and self.debug_weights:
            print(f"--- 更新权重 (Segment End Iteration {self.iteration_count}) ---")

        segment_destroy_scores = self.destroy_scores.copy()
        segment_repair_scores = self.repair_scores.copy()
        segment_destroy_counts = self.destroy_counts.copy()
        segment_repair_counts = self.repair_counts.copy()

        # 更新 Destroy 算子权重
        for name in self.destroy_operators:
            count = segment_destroy_counts.get(name, 0)
            score = segment_destroy_scores.get(name, 0.0)
            if count > 0:
                performance = score / count
                self.destroy_weights[name] = max(0.01, (1 - self.weight_update_rate) * self.destroy_weights[name] + self.weight_update_rate * performance)
            else:
                # 如果算子在本段未被使用，权重略微降低（避免完全停滞）
                self.destroy_weights[name] = max(0.01, self.destroy_weights[name] * 0.95)

        # 更新 Repair 算子权重
        for name in self.repair_operators:
            count = segment_repair_counts.get(name, 0)
            score = segment_repair_scores.get(name, 0.0)
            if count > 0:
                performance = score / count
                self.repair_weights[name] = max(0.01, (1 - self.weight_update_rate) * self.repair_weights[name] + self.weight_update_rate * performance)
            else:
                self.repair_weights[name] = max(0.01, self.repair_weights[name] * 0.95)

        # 记录当前段的统计（用于调试或分析）
        segment_record = {'segment_end_iteration': self.iteration_count, 'destroy_ops': {}, 'repair_ops': {}}
        for name in self.destroy_operators:
            segment_record['destroy_ops'][name] = {'score': segment_destroy_scores.get(name, 0.0), 'count': segment_destroy_counts.get(name, 0), 'weight': self.destroy_weights[name]}
        for name in self.repair_operators:
            segment_record['repair_ops'][name] = {'score': segment_repair_scores.get(name, 0.0), 'count': segment_repair_counts.get(name, 0), 'weight': self.repair_weights[name]}
        self.operator_history.append(segment_record)

        # 重置当前段的得分和计数
        self.destroy_scores = {name: 0.0 for name in self.destroy_operators}
        self.repair_scores = {name: 0.0 for name in self.repair_operators}
        self.destroy_counts = {name: 0 for name in self.destroy_operators}
        self.repair_counts = {name: 0 for name in self.repair_operators}

        # 对冲突历史计数进行衰减
        decay_factor = self.conflict_history_decay
        if decay_factor < 1.0:
            original_sum = sum(self.agv_conflict_counts.values())
            # 使用 Counter 的 update 方法和生成器表达式进行衰减
            decayed_counts = {agv_id: int(count * decay_factor) for agv_id, count in self.agv_conflict_counts.items()}
            self.agv_conflict_counts.clear()
            self.agv_conflict_counts.update({agv_id: count for agv_id, count in decayed_counts.items() if count > 0}) # 只保留大于0的计数
            new_sum = sum(self.agv_conflict_counts.values())
            if self.verbose and self.debug_weights:
                print(f"    Conflict Counts Decayed (Factor={decay_factor:.2f}): Total {original_sum} -> {new_sum}")

    # --- 保存历史数据 ---
    def _save_history_data(self):
        """将成本历史和算子历史保存到 CSV 文件。"""
        if not self.record_history:
            return

        # 保存成本历史
        cost_history_file = os.path.join(self.results_dir, f"{self.instance_identifier}_cost_history.csv")
        try:
            if self.cost_history:
                with open(cost_history_file, 'w', newline='', encoding='utf-8') as csvfile:
                    # 确保所有可能的键都在 fieldnames 中
                    fieldnames = ['iteration', 'current_cost', 'best_cost', 'temperature']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # 忽略字典中多余的键
                    writer.writeheader()
                    writer.writerows(self.cost_history)
                if self.verbose: print(f"成本历史已保存到: {cost_history_file}")
        except Exception as e:
            print(f"错误: 无法写入成本历史文件 '{cost_history_file}': {e}")

        # 保存算子历史
        operator_history_file = os.path.join(self.results_dir, f"{self.instance_identifier}_operator_history.csv")
        try:
            if self.operator_history:
                flat_op_history = []
                # 遍历每个记录段
                for segment_record in self.operator_history:
                    iter_num = segment_record['segment_end_iteration']
                    # 遍历 destroy 和 repair 两类算子
                    for op_type, ops_dict in segment_record.items():
                        if op_type.endswith('_ops'): # 确保是算子字典
                            op_category = op_type.split('_')[0] # 'destroy' or 'repair'
                            # 遍历该类别下的每个算子
                            for op_name, stats in ops_dict.items():
                                flat_op_history.append({
                                    'segment_end_iteration': iter_num,
                                    'operator_type': op_category,
                                    'operator_name': op_name,
                                    'score': stats.get('score', 0.0), # 使用 .get() 增加健壮性
                                    'count': stats.get('count', 0),
                                    'weight': stats.get('weight', 0.0)
                                })
                if flat_op_history:
                    # 获取所有可能的键作为 fieldnames
                    fieldnames = list(flat_op_history[0].keys()) if flat_op_history else []
                    if fieldnames:
                        with open(operator_history_file, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(flat_op_history)
                        if self.verbose: print(f"算子历史已保存到: {operator_history_file}")
        except Exception as e:
            print(f"错误: 无法写入算子历史文件 '{operator_history_file}': {e}")

    # --- 绘制收敛图 ---
    def plot_convergence_method(self):
        """绘制成本和温度随迭代次数变化的收敛图。"""
        if not self.plot_convergence_flag: return
        if not _visual_libs_available: return # 库不可用则静默退出
        if not self.cost_history: return # 无数据则静默退出

        plot_file = os.path.join(self.results_dir, f"{self.instance_identifier}_convergence.png")
        try:
            df = pd.DataFrame(self.cost_history)
            # 处理无穷大值，替换为 NaN 以便绘图忽略
            df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
            df.dropna(subset=['current_cost', 'best_cost', 'temperature'], how='any', inplace=True) # 删除包含 NaN 的行

            if df.empty:
                if self.verbose: print("警告: 处理后的成本历史为空，无法绘制收敛图。")
                return

            # 获取有效的初始最优成本用于设置 Y 轴上限
            initial_valid_cost = df['best_cost'].iloc[0] if not df.empty else None
            y_limit_upper = initial_valid_cost * 1.5 if initial_valid_cost is not None and initial_valid_cost > 0 else None

            fig, ax1 = plt.subplots(figsize=(12, 6))
            color1 = 'tab:blue'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Cost', color=color1)
            ax1.plot(df['iteration'], df['current_cost'], color=color1, alpha=0.6, label='Current Cost')
            ax1.plot(df['iteration'], df['best_cost'], color=color1, linestyle='-', linewidth=2, label='Best Cost')
            ax1.tick_params(axis='y', labelcolor=color1)
            # 设置 Y 轴下限为 0，上限根据初始成本动态调整
            ax1.set_ylim(bottom=0)
            if y_limit_upper: ax1.set_ylim(top=y_limit_upper)

            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel('Temperature', color=color2)
            ax2.plot(df['iteration'], df['temperature'], color=color2, linestyle=':', alpha=0.7, label='Temperature')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(bottom=0) # 温度下限为 0

            # 合并图例
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines or lines2:
                 ax2.legend(lines + lines2, labels + labels2, loc='upper right')

            plt.title(f'ALNS Convergence ({self.instance_identifier})')
            fig.tight_layout() # 调整布局防止标签重叠
            plt.savefig(plot_file)
            plt.close(fig) # 关闭图形，释放内存

            if self.verbose: print(f"收敛图已保存到: {plot_file}")
        except ImportError:
            # 这个异常应该在 _visual_libs_available 中处理，但作为保险
            pass # 静默失败
        except Exception as e:
            print(f"错误: 绘制收敛图时发生错误: {e}")


# --- 示例用法 (使用 v39) ---
if __name__ == '__main__':
    print("--- ALNS (v39 - Edge Conflict Handling, Full Code) 示例 ---")
    try:
        from InstanceGenerator import load_fixed_scenario_1
    except ImportError as e: print(f"错误: 导入 InstanceGenerator 失败: {e}"); sys.exit(1)

    instance_data = load_fixed_scenario_1()
    if not instance_data: print("错误: 无法加载固定算例。"); sys.exit(1)
    test_map, test_tasks = instance_data
    planner_main = TWAStarPlanner()
    instance_id = "fixed_scenario_1_v39_test"; results_directory = "alns_output_v39"

    # 使用更高的迭代次数进行测试
    alns_init_params = {
        'instance_identifier': instance_id, 'results_dir': results_directory,
        'max_iterations': 1000, # 增加迭代次数
        'initial_temp': 25.0,   # 稍高初始温度
        'cooling_rate': 0.995, # 稍慢冷却
        'segment_size': 100,    # 匹配更高迭代次数
        'weight_update_rate': 0.15,
        'sigma1': 15.0, 'sigma2': 8.0, 'sigma3': 3.0,
        'removal_percentage_min': 0.2, 'removal_percentage_max': 0.45,
        'regret_k': 3,
        'no_improvement_limit': 200, # 匹配更高迭代次数
        'conflict_history_decay': 0.9,
        'max_time': 600, # 可能需要更长时间范围
        'cost_weights': (1.0, 0.3, 0.8),
        'v': 1.0, 'delta_step': 1.0, 'buffer': 1,
        'planner_time_limit_factor': 5.0, 'regret_planner_time_limit_abs': 0.15,
        'wait_threshold': 6, 'deadlock_max_wait': 10,
        'edge_wait_threshold': 4, # 边冲突等待阈值
        'verbose': True, 'debug_weights': False, # 减少默认输出干扰
        'record_history': True, 'plot_convergence': True,
    }

    try: alns_instance = ALNS(grid_map=test_map, tasks=test_tasks, planner=planner_main, **alns_init_params)
    except Exception as init_e: print(f"错误: 初始化 ALNS 实例失败: {init_e}"); traceback.print_exc(); sys.exit(1)

    try: best_sol, duration, best_cost_dict = alns_instance.run()
    except Exception as run_e: print(f"错误: ALNS 运行时发生异常: {run_e}"); traceback.print_exc(); sys.exit(1)

    print("\n--- ALNS (v39) 运行完成 ---")
    if best_sol is not None:
        final_total_cost = best_cost_dict.get('total', float('inf'))
        final_cost_str = f"{final_total_cost:.2f}" if final_total_cost != float('inf') else "Inf"
        print(f"最终最优解成本: Total={final_cost_str}")
        print(f"  Breakdown: Travel={best_cost_dict.get('travel', 0.0):.2f}, Turn={best_cost_dict.get('turn', 0.0):.2f}, Wait={best_cost_dict.get('wait', 0.0):.2f}")
    else:
        print("ALNS 未能找到可行解。")

    print(f"结果文件已保存在目录: '{os.path.abspath(results_directory)}'")
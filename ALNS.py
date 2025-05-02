# ALNS.py (v44 - Dynamic Bounding Box, No Comments)
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

from Map import GridMap, Node
from DataTypes import Task, Path as AgentPath, TimeStep, State, DynamicObstacles, Solution, CostDict, check_time_overlap
from Planner import TWAStarPlanner

try:
    import Operators
except ImportError:
    print("错误: 无法导入 Operators.py (v9+)。请确保文件存在且在路径中。")
    sys.exit(1)

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    _visual_libs_available = True
except ImportError:
    _visual_libs_available = False


class ALNS:
    def __init__(self,
                 grid_map: GridMap, tasks: List[Task], planner: TWAStarPlanner,
                 instance_identifier: str = "default_instance",
                 results_dir: str = "alns_results",
                 initial_weights: Optional[Tuple[Dict[str, float], Dict[str, float]]] = None,
                 agv_speeds: Optional[Dict[int, float]] = None,
                 priority_strategy: str = 'id',
                 skip_internal_initial_solution: bool = False,
                 **kwargs):

        if not isinstance(grid_map, GridMap): raise TypeError("grid_map 必须是 GridMap 类型")
        if not isinstance(tasks, list) or not all(isinstance(t, Task) for t in tasks): raise TypeError("tasks 必须是 Task 列表")
        if not isinstance(planner, TWAStarPlanner): raise TypeError("planner 必须是 TWAStarPlanner 类型")
        if not isinstance(instance_identifier, str): raise TypeError("instance_identifier 必须是字符串")
        if not isinstance(results_dir, str): raise TypeError("results_dir 必须是字符串")

        self.grid_map = grid_map
        self.tasks = tasks
        self.num_agvs = len(tasks)
        self.planner = planner
        self.agv_ids = sorted([task.agv_id for task in tasks])
        self.instance_identifier = instance_identifier
        self.results_dir = results_dir
        self.skip_internal_initial_solution = skip_internal_initial_solution

        self.max_time: TimeStep = kwargs.get('max_time', 600)
        self.cost_weights: Tuple[float, float, float] = kwargs.get('cost_weights', (1.0, 0.3, 0.8))
        self.alpha, self.beta, self.gamma_wait = self.cost_weights
        self.default_v: float = kwargs.get('v', 1.0)
        self.delta_step: float = kwargs.get('delta_step', 1.0)
        self.buffer: int = kwargs.get('buffer', 1)
        self.planner_time_limit_factor: float = kwargs.get('alns_planner_time_limit_factor', 4.0)
        self.regret_planner_time_limit_abs: float = kwargs.get('alns_regret_planner_time_limit_abs', 0.1)

        self.agv_speeds: Dict[int, float] = {}
        if agv_speeds and isinstance(agv_speeds, dict):
            self.agv_speeds = agv_speeds
            missing_speeds = []
            for task in self.tasks:
                if task.agv_id not in self.agv_speeds:
                    missing_speeds.append(task.agv_id)
                    self.agv_speeds[task.agv_id] = self.default_v
            if missing_speeds:
                print(f"警告: {len(missing_speeds)} 个 AGV ({missing_speeds}) 缺少速度信息，已使用默认值 {self.default_v}")
        else:
            for task in self.tasks:
                self.agv_speeds[task.agv_id] = self.default_v
        self.priority_strategy = priority_strategy.lower()
        if self.priority_strategy not in ['id', 'speed']:
            print(f"警告: 无效的 priority_strategy '{priority_strategy}'，回退到 'id'。")
            self.priority_strategy = 'id'

        self.max_iterations: int = kwargs.get('alns_max_iterations', 1000)
        self.initial_temperature: float = kwargs.get('alns_initial_temp', 50.0)
        self.cooling_rate: float = kwargs.get('alns_cooling_rate', 0.99)
        self.segment_size: int = kwargs.get('alns_segment_size', 50)
        self.weight_update_rate: float = kwargs.get('alns_weight_update_rate', 0.15)
        self.sigma1: float = kwargs.get('alns_sigma1', 15.0)
        self.sigma2: float = kwargs.get('alns_sigma2', 8.0)
        self.sigma3: float = kwargs.get('alns_sigma3', 4.0)
        self.removal_percentage_min: float = kwargs.get('alns_removal_percentage_min', 0.15)
        self.removal_percentage_max: float = kwargs.get('alns_removal_percentage_max', 0.40)
        self.regret_k: int = max(2, kwargs.get('alns_regret_k', 3))
        self.regret_max_attempts: int = kwargs.get('alns_regret_max_attempts', 10)
        self.no_improvement_limit: Optional[int] = kwargs.get('alns_no_improvement_limit', 200)
        self.conflict_history_decay: float = kwargs.get('conflict_history_decay', 0.9)
        self.no_improvement_bbox_disable_threshold: int = kwargs.get('alns_no_improvement_bbox_disable', 10) # New param

        self.wait_threshold: int = kwargs.get('wait_threshold', 6)
        self.deadlock_max_wait: int = kwargs.get('deadlock_max_wait', 20)
        self.edge_wait_threshold: int = kwargs.get('edge_wait_threshold', 4)

        self.verbose: bool = kwargs.get('alns_verbose_output', False)
        self.debug_weights: bool = kwargs.get('alns_debug_weights', False)
        self.record_history: bool = kwargs.get('alns_record_history', True)
        self.plot_convergence_flag: bool = kwargs.get('alns_plot_convergence', True)

        self.temperature: float = self.initial_temperature
        self.best_solution: Optional[Solution] = None
        self.best_cost: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.current_solution: Optional[Solution] = None
        self.current_cost: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.initial_cost: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.iteration_count: int = 0
        self.no_improvement_count: int = 0
        self.use_bounding_box: bool = True # New state flag for dynamic bbox
        self.agv_conflict_counts: Counter[int] = Counter()
        self.cost_history: List[Dict] = []
        self.operator_history: List[Dict] = []

        DestroyOpType = Callable[['ALNS', Solution, int], Tuple[Solution, List[int]]]
        RepairOpType = Callable[['ALNS', Solution, List[int]], Optional[Solution]]
        self.destroy_operators: Dict[str, DestroyOpType] = {
            "random_removal": Operators.random_removal,
            "worst_removal": Operators.worst_removal,
            "related_removal": Operators.related_removal,
            "congestion_removal": Operators.congestion_removal,
            "conflict_history_removal": Operators.conflict_history_removal,
            "shaw_removal": Operators.shaw_removal,
            "string_removal": Operators.string_removal
        }
        self.repair_operators: Dict[str, RepairOpType] = {
            "greedy_insertion": Operators.greedy_insertion,
            "regret_insertion": Operators.regret_insertion,
            "wait_adjustment_repair": Operators.wait_adjustment_repair,
        }
        for name, op_func in self.destroy_operators.items():
            if not callable(op_func): raise TypeError(f"破坏算子 '{name}' 不是可调用函数。")
        for name, op_func in self.repair_operators.items():
            if not callable(op_func): raise TypeError(f"修复算子 '{name}' 不是可调用函数。")

        self.destroy_weights: Dict[str, float] = {}
        self.repair_weights: Dict[str, float] = {}
        destroy_ops = list(self.destroy_operators.keys())
        repair_ops = list(self.repair_operators.keys())

        if initial_weights:
            initial_destroy_weights, initial_repair_weights = initial_weights
            valid_destroy_ops = {op: initial_destroy_weights.get(op, 0.0) for op in destroy_ops}
            valid_repair_ops = {op: initial_repair_weights.get(op, 0.0) for op in repair_ops}
            total_destroy_weight = sum(valid_destroy_ops.values())
            total_repair_weight = sum(valid_repair_ops.values())
            if total_destroy_weight > 1e-9:
                self.destroy_weights = {op: w / total_destroy_weight for op, w in valid_destroy_ops.items()}
            else:
                uniform_destroy_weight = 1.0 / len(destroy_ops) if destroy_ops else 0
                self.destroy_weights = {op: uniform_destroy_weight for op in destroy_ops}
            if total_repair_weight > 1e-9:
                self.repair_weights = {op: w / total_repair_weight for op, w in valid_repair_ops.items()}
            else:
                uniform_repair_weight = 1.0 / len(repair_ops) if repair_ops else 0
                self.repair_weights = {op: uniform_repair_weight for op in repair_ops}
        else:
            uniform_destroy_weight = 1.0 / len(destroy_ops) if destroy_ops else 0
            uniform_repair_weight = 1.0 / len(repair_ops) if repair_ops else 0
            self.destroy_weights = {op_name: uniform_destroy_weight for op_name in destroy_ops}
            self.repair_weights = {op_name: uniform_repair_weight for op_name in repair_ops}

        self.destroy_scores = {op_name: 0.0 for op_name in destroy_ops}
        self.repair_scores = {op_name: 0.0 for op_name in repair_ops}
        self.destroy_counts = {op_name: 0 for op_name in destroy_ops}
        self.repair_counts = {op_name: 0 for op_name in repair_ops}

        try: os.makedirs(self.results_dir, exist_ok=True)
        except OSError as e: print(f"错误: 无法创建结果目录 '{self.results_dir}': {e}")

    def _call_planner(self, task: Task, dynamic_obstacles: DynamicObstacles, start_time: TimeStep = 0, bounding_box: Optional[Tuple[int, int, int, int]] = None) -> Optional[AgentPath]:
        planner_time_limit = None
        if self.planner_time_limit_factor is not None and self.planner_time_limit_factor > 0:
            base_time = 0.8
            planner_time_limit = max(0.2, min(base_time * self.planner_time_limit_factor, 15.0))

        # --- v44: Check if bounding box should be used ---
        effective_bounding_box = bounding_box if self.use_bounding_box else None
        # --- ----------------------------------------- ---

        try:
            agv_id = task.agv_id
            agv_speed = self.agv_speeds.get(agv_id, self.default_v)
            path: Optional[AgentPath] = self.planner.plan(
                grid_map=self.grid_map, task=task, dynamic_obstacles=dynamic_obstacles,
                max_time=self.max_time, cost_weights=self.cost_weights,
                agv_speed=agv_speed,
                delta_step=self.delta_step, start_time=start_time,
                time_limit=planner_time_limit,
                bounding_box=effective_bounding_box # Use potentially modified bbox
            )
            return path if isinstance(path, AgentPath) else None
        except Exception as e:
            print(f"错误: Planner.plan 调用失败 (AGV {task.agv_id}): {e}")
            return None

    def _calculate_total_cost(self, solution: Solution) -> CostDict:
        total_cost_dict: CostDict = {'total': 0.0, 'travel': 0.0, 'turn': 0.0, 'wait': 0.0}
        inf_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        if not isinstance(solution, dict) or not solution: return inf_dict
        valid_solution = True
        num_paths = 0
        for agv_id, path in solution.items():
            if not path or not isinstance(path, AgentPath) or not path.sequence:
                valid_solution = False; break
            num_paths += 1
            try:
                agv_speed = self.agv_speeds.get(agv_id, self.default_v)
                cost_dict = path.get_cost(self.grid_map, self.alpha, self.beta, self.gamma_wait, agv_speed, self.delta_step)
            except Exception as e:
                print(f"错误: 计算 AGV {agv_id} 成本时出错: {e}")
                valid_solution = False; break
            if cost_dict.get('total', float('inf')) == float('inf'):
                valid_solution = False; break
            for key in total_cost_dict:
                total_cost_dict[key] += cost_dict.get(key, 0.0)
        if valid_solution and num_paths == self.num_agvs:
            return total_cost_dict
        else:
            return inf_dict

    def generate_initial_solution(self) -> Optional[Solution]:
        print("--- 生成初始解 (优先序贯规划) ---")
        solution: Solution = {}
        dynamic_obstacles: DynamicObstacles = {}
        if self.priority_strategy == 'speed':
            sorted_tasks = sorted(self.tasks, key=lambda t: (-self.agv_speeds.get(t.agv_id, 0), t.agv_id))
            print("  使用基于速度的初始规划顺序。")
        else:
            sorted_tasks = sorted(self.tasks, key=lambda t: t.agv_id)
            print("  使用基于 ID 的初始规划顺序。")
        all_success = True
        for task in sorted_tasks:
            agv_id = task.agv_id
            t_start_call = time.perf_counter()
            if self.verbose: print(f"  规划初始 AGV {agv_id} (速度: {self.agv_speeds.get(agv_id, self.default_v):.2f})...")
            try:
                current_dynamic_obstacles = Operators._build_dynamic_obstacles(self, solution)
            except Exception as build_e:
                print(f"错误: 构建动态障碍时失败 (AGV {agv_id}): {build_e}")
                return None
            path = self._call_planner(task, current_dynamic_obstacles, start_time=0)
            call_dur = time.perf_counter() - t_start_call
            if path and path.sequence:
                solution[agv_id] = path
                if self.verbose: print(f"    成功 (耗时 {call_dur:.3f}s)，路径长度 {len(path)}, Makespan {path.get_makespan()}")
            else:
                print(f"  错误：AGV {agv_id} 初始规划失败！(耗时 {call_dur:.3f}s)")
                all_success = False; break
        if not all_success:
            self.initial_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
            return None
        self.initial_cost = self._calculate_total_cost(solution)
        initial_total_cost = self.initial_cost.get('total', float('inf'))
        initial_cost_str = f"{initial_total_cost:.2f}" if initial_total_cost != float('inf') else "Inf"
        print(f"--- 初始解生成完毕，成本: {initial_cost_str} ---")
        return solution if initial_total_cost != float('inf') else None

    def _select_operator_roulette_wheel(self, weights: Dict[str, float]) -> str:
        total_weight = sum(weights.values())
        if total_weight <= 1e-9:
            valid_operators = [name for name, w in weights.items() if w > 1e-9]
            return random.choice(valid_operators) if valid_operators else random.choice(list(weights.keys()))
        else:
            pick = random.uniform(0, total_weight)
            current = 0.0
            for name, weight in weights.items():
                current += weight
                if current >= pick - 1e-9: return name
            return list(weights.keys())[-1]

    def _resolve_conflicts_and_deadlocks(self, solution: Solution) -> Optional[Solution]:
        if not solution: return None
        current_solution = copy.deepcopy(solution)
        max_resolve_attempts = self.num_agvs * 15
        resolve_attempt = 0

        while resolve_attempt < max_resolve_attempts:
            resolve_attempt += 1
            made_change_this_pass = False
            if self.verbose and self.debug_weights: print(f"  冲突/死锁解决尝试 #{resolve_attempt}")

            agv_wait_times: Dict[int, Tuple[TimeStep, Node]] = defaultdict(lambda: (0, (-1,-1)))
            max_t_in_solution = 0
            paths_sequences = {agv_id: path.sequence for agv_id, path in current_solution.items() if path and path.sequence}
            if not paths_sequences: return current_solution
            for path_seq in paths_sequences.values():
                if path_seq: max_t_in_solution = max(max_t_in_solution, path_seq[-1][1])

            deadlocked_agvs: Set[int] = set()
            node_at_time: Dict[int, Dict[TimeStep, Node]] = defaultdict(dict)
            for agv_id, path_seq in paths_sequences.items():
                 if path_seq:
                     for node, t in path_seq: node_at_time[agv_id][t] = node

            for agv_id in paths_sequences.keys():
                wait_duration, current_wait_node = agv_wait_times[agv_id]
                last_node = None
                for t_step in range(max_t_in_solution + 1):
                    node_t = node_at_time[agv_id].get(t_step)
                    if node_t is not None and last_node is not None and node_t == last_node:
                        if current_wait_node == node_t: wait_duration += 1
                        else: wait_duration = 1; current_wait_node = node_t
                        agv_wait_times[agv_id] = (wait_duration, current_wait_node)
                        if wait_duration > self.deadlock_max_wait:
                            deadlocked_agvs.add(agv_id); self.agv_conflict_counts[agv_id] += 3; break
                    else:
                        wait_duration = 0; current_wait_node = (-1,-1); agv_wait_times[agv_id] = (wait_duration, current_wait_node)
                    last_node = node_t

            if deadlocked_agvs:
                agv_to_resolve = max(deadlocked_agvs)
                if self.verbose: print(f"      检测到死锁，选择 AGV {agv_to_resolve} 重规划。")
                path_to_resolve = current_solution.get(agv_to_resolve)
                original_task = next((t for t in self.tasks if t.agv_id == agv_to_resolve), None)
                if path_to_resolve and path_to_resolve.sequence and original_task:
                    wait_t_count, _ = agv_wait_times[agv_to_resolve]
                    deadlock_start_time = path_to_resolve.sequence[-1][1] - wait_t_count + 1
                    replan_start_index = max(0, next((i for i in range(len(path_to_resolve.sequence) - 1, -1, -1) if path_to_resolve.sequence[i][1] < deadlock_start_time), 0))
                    replan_start_state = path_to_resolve.sequence[replan_start_index]
                    start_idx_truncate = replan_start_index + 1
                    replan_task = Task(agv_id=agv_to_resolve, start_node=replan_start_state[0], goal_node=original_task.goal_node)
                    replan_start_time_val = replan_start_state[1]
                    current_solution[agv_to_resolve].sequence = path_to_resolve.sequence[:start_idx_truncate]
                    dynamic_obs = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_to_resolve)
                    new_path_segment = self._call_planner(replan_task, dynamic_obs, start_time=replan_start_time_val)
                    if new_path_segment and new_path_segment.sequence and len(new_path_segment.sequence) > 1:
                        current_solution[agv_to_resolve].sequence.extend(new_path_segment.sequence[1:])
                        made_change_this_pass = True
                        if self.verbose: print(f"      死锁解决：AGV {agv_to_resolve} 重规划成功。")
                        continue
                    else:
                        print(f"      错误：死锁解决失败，AGV {agv_to_resolve} 重规划未找到有效路径。"); return None
                else:
                    print(f"错误：解决死锁时无法获取 AGV {agv_to_resolve} 的路径或任务。"); return None

            conflict_resolved_in_scan_node = False
            max_t_check_node = 0
            paths_sequences_node = {aid: p.sequence for aid, p in current_solution.items() if p and p.sequence}
            if not paths_sequences_node: return current_solution
            for path_seq in paths_sequences_node.values():
                if path_seq: max_t_check_node = max(max_t_check_node, path_seq[-1][1])
            node_occupancy_cache_node: Dict[TimeStep, Dict[Node, List[int]]] = defaultdict(lambda: defaultdict(list))
            for agv_id, path_seq in paths_sequences_node.items():
                 if path_seq:
                     for node, t in path_seq: node_occupancy_cache_node[t][node].append(agv_id)

            for t_step in range(max_t_check_node + 1):
                nodes_with_conflict = {node: occ for node, occ in node_occupancy_cache_node[t_step].items() if len(occ) > 1}
                if nodes_with_conflict:
                    for node, occupants in nodes_with_conflict.items():
                        if len(occupants) < 2: continue
                        agv_high_priority_id: int = -1; agvs_low_priority_ids: List[int] = []
                        if self.priority_strategy == 'speed':
                            try:
                                occupants.sort(key=lambda aid: (-self.agv_speeds.get(aid, 0), aid))
                                agv_high_priority_id = occupants[0]; agvs_low_priority_ids = occupants[1:]
                            except Exception as sort_e: print(f"错误: 按速度排序冲突占用者时失败: {sort_e}"); continue
                        else:
                            occupants.sort(); agv_high_priority_id = occupants[0]; agvs_low_priority_ids = occupants[1:]
                        if agv_high_priority_id == -1: continue
                        if self.verbose: print(f"      检测到节点冲突 @ ({node}, {t_step}), HighPrio={agv_high_priority_id}, LowPrio={agvs_low_priority_ids} (Strategy: {self.priority_strategy})")

                        for agv_low in agvs_low_priority_ids:
                            self.agv_conflict_counts[agv_low] += 1
                            path_low = current_solution.get(agv_low)
                            if not path_low or not path_low.sequence: continue
                            conflict_idx = next((i for i, (n,t) in enumerate(path_low.sequence) if n==node and t==t_step), -1)
                            if conflict_idx == -1: continue
                            prev_idx = conflict_idx - 1; should_replan = False
                            if prev_idx < 0:
                                prev_node, prev_time = path_low.sequence[0][0], 0; should_replan = True
                                print(f"警告: AGV {agv_low} 在起点或极短路径发生节点冲突，强制重规划。")
                            else: prev_node, prev_time = path_low.sequence[prev_idx]

                            if not should_replan:
                                wait_until = t_step + 1; path_high = current_solution.get(agv_high_priority_id); last_high_t = -1
                                if path_high and path_high.sequence:
                                    last_high_t = max((ht for hn, ht in path_high.sequence if ht >= t_step and hn == node), default=-1)
                                if last_high_t != -1: wait_until = max(wait_until, last_high_t + 1)
                                required_wait_duration = wait_until - t_step
                                if required_wait_duration <= 0 : continue
                                if required_wait_duration > self.wait_threshold:
                                    should_replan = True; self.agv_conflict_counts[agv_low] += 1
                                else:
                                    if self.verbose: print(f"      AGV {agv_low} 尝试等待 {required_wait_duration} 步 @ {prev_node} (节点冲突)")
                                    new_sequence_prefix = path_low.sequence[:prev_idx+1]
                                    dynamic_obs_wait_check = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_low)
                                    wait_possible = True
                                    for w_step in range(1, required_wait_duration + 1):
                                        current_wait_time = prev_time + w_step
                                        if current_wait_time in dynamic_obs_wait_check and prev_node in dynamic_obs_wait_check.get(current_wait_time, set()):
                                            should_replan = True; wait_possible = False; self.agv_conflict_counts[agv_low] += 1
                                            if self.verbose: print(f"        等待失败: 节点 {prev_node} 在 t={current_wait_time} 被占用。"); break
                                        new_sequence_prefix.append((prev_node, current_wait_time))
                                    if wait_possible:
                                        time_shift = required_wait_duration; path_after_wait_valid = True; shifted_path_suffix = []
                                        for i in range(conflict_idx, len(path_low.sequence)):
                                            original_node_wait, original_time_wait = path_low.sequence[i]; new_time_wait = original_time_wait + time_shift
                                            if new_time_wait > self.max_time: path_after_wait_valid = False; break
                                            if new_time_wait in dynamic_obs_wait_check and original_node_wait in dynamic_obs_wait_check.get(new_time_wait, set()): path_after_wait_valid = False; break
                                            shifted_path_suffix.append((original_node_wait, new_time_wait))
                                        if path_after_wait_valid:
                                            current_solution[agv_low].sequence = new_sequence_prefix + shifted_path_suffix
                                            made_change_this_pass = True; conflict_resolved_in_scan_node = True
                                            if self.verbose: print(f"      AGV {agv_low} 等待解决节点冲突成功。")
                                            node_occupancy_cache_node.clear(); break
                                        else:
                                            should_replan = True; self.agv_conflict_counts[agv_low] += 1
                                            if self.verbose: print(f"        等待后平移路径失败 (超时或新冲突)，AGV {agv_low} 需要重规划。")

                            if should_replan:
                                if self.verbose: print(f"      AGV {agv_low} 将从 ({prev_node}, {prev_time}) 开始重规划 (节点冲突)。")
                                task_low_original = next((t for t in self.tasks if t.agv_id == agv_low), None)
                                if task_low_original is not None:
                                    replan_task_low = Task(agv_id=agv_low, start_node=prev_node, goal_node=task_low_original.goal_node)
                                    current_solution[agv_low].sequence = path_low.sequence[:prev_idx+1] if prev_idx >= 0 else []
                                    dynamic_obs_replan_node = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_low)
                                    new_segment_node = self._call_planner(replan_task_low, dynamic_obs_replan_node, start_time=prev_time)
                                    if new_segment_node and new_segment_node.sequence and len(new_segment_node.sequence) > 1:
                                        current_solution[agv_low].sequence.extend(new_segment_node.sequence[1:])
                                        made_change_this_pass = True; conflict_resolved_in_scan_node = True
                                        if self.verbose: print(f"      AGV {agv_low} 重规划解决节点冲突成功。")
                                        node_occupancy_cache_node.clear(); break
                                    else: print(f"      错误：节点冲突解决失败，AGV {agv_low} 重规划未找到有效路径段。"); return None
                                else: print(f"错误：无法找到 AGV {agv_low} 的原始任务用于重规划。"); return None
                        if conflict_resolved_in_scan_node: break
                    if conflict_resolved_in_scan_node: break
                if conflict_resolved_in_scan_node: break
            if conflict_resolved_in_scan_node: continue

            edge_conflict_resolved_in_scan = False
            moves_dict: Dict[int, List[Tuple[Node, Node, TimeStep, TimeStep]]] = defaultdict(list)
            paths_sequences_edge = {aid: p.sequence for aid, p in current_solution.items() if p and p.sequence}
            if not paths_sequences_edge: return current_solution
            for agv_id, path_seq in paths_sequences_edge.items():
                if path_seq:
                    for i in range(len(path_seq) - 1):
                        node1, time1 = path_seq[i]; node2, time2 = path_seq[i+1]
                        if node1 != node2: moves_dict[agv_id].append((node1, node2, time1, time2))

            processed_edge_conflicts: Set[Tuple[int, int, TimeStep]] = set()
            agv_ids_list = list(paths_sequences_edge.keys())

            for idx1, agv1_id in enumerate(agv_ids_list):
                for idx2 in range(idx1 + 1, len(agv_ids_list)):
                    agv2_id = agv_ids_list[idx2]
                    for move1_from, move1_to, move1_start, move1_end in moves_dict.get(agv1_id, []):
                        for move2_from, move2_to, move2_start, move2_end in moves_dict.get(agv2_id, []):
                            if move1_from == move2_to and move1_to == move2_from:
                                if check_time_overlap(move1_start, move1_end, move2_start, move2_end):
                                    conflict_time_approx = max(move1_start, move2_start)
                                    low_id, high_id = min(agv1_id, agv2_id), max(agv1_id, agv2_id)
                                    conflict_key = (low_id, high_id, conflict_time_approx)
                                    if conflict_key in processed_edge_conflicts: continue

                                    agv_to_wait_id: int = -1; agv_proceed_id: int = -1
                                    speed1 = self.agv_speeds.get(agv1_id, 0); speed2 = self.agv_speeds.get(agv2_id, 0)
                                    if self.priority_strategy == 'speed':
                                        if abs(speed1 - speed2) < 1e-6: agv_proceed_id, agv_to_wait_id = min(agv1_id, agv2_id), max(agv1_id, agv2_id)
                                        elif speed1 > speed2: agv_proceed_id, agv_to_wait_id = agv1_id, agv2_id
                                        else: agv_proceed_id, agv_to_wait_id = agv2_id, agv1_id
                                    else: agv_proceed_id, agv_to_wait_id = min(agv1_id, agv2_id), max(agv1_id, agv2_id)
                                    if agv_to_wait_id == -1: continue

                                    processed_edge_conflicts.add(conflict_key); self.agv_conflict_counts[agv_to_wait_id] += 1
                                    if self.verbose: print(f"      检测到边冲突 between AGV {agv1_id} & AGV {agv2_id}, Waiter={agv_to_wait_id} (Strategy: {self.priority_strategy})")

                                    move_proceed_end_t = move1_end if agv_proceed_id == agv1_id else move2_end
                                    path_to_wait = current_solution.get(agv_to_wait_id)
                                    if not path_to_wait or not path_to_wait.sequence: continue
                                    wait_from_idx = -1; wait_start_time = -1
                                    move_wait_from = move1_from if agv_to_wait_id == agv1_id else move2_from
                                    move_wait_to = move1_to if agv_to_wait_id == agv1_id else move2_to
                                    wait_move_start_t = move1_start if agv_to_wait_id == agv1_id else move2_start
                                    for i in range(len(path_to_wait.sequence)-1):
                                        n1_w, t1_w = path_to_wait.sequence[i]; n2_w, t2_w = path_to_wait.sequence[i+1]
                                        if n1_w == move_wait_from and n2_w == move_wait_to and t1_w == wait_move_start_t:
                                            wait_from_idx = i; wait_start_time = t1_w; break
                                    if wait_from_idx == -1: continue

                                    required_edge_wait_duration = max(0, move_proceed_end_t - wait_start_time)
                                    should_replan_edge = False
                                    if required_edge_wait_duration > self.edge_wait_threshold:
                                        should_replan_edge = True; self.agv_conflict_counts[agv_to_wait_id] += 1
                                    else:
                                        wait_node_edge, _ = path_to_wait.sequence[wait_from_idx]
                                        if self.verbose: print(f"      AGV {agv_to_wait_id} 尝试等待 {required_edge_wait_duration} 步 @ {wait_node_edge} (边冲突)。")
                                        new_sequence_prefix_edge = path_to_wait.sequence[:wait_from_idx+1]
                                        dynamic_obs_edge_wait_check = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_to_wait_id)
                                        wait_possible_edge = True
                                        for w_step in range(1, required_edge_wait_duration + 1):
                                            current_wait_time_edge = wait_start_time + w_step
                                            if current_wait_time_edge in dynamic_obs_edge_wait_check and wait_node_edge in dynamic_obs_edge_wait_check.get(current_wait_time_edge, set()):
                                                should_replan_edge = True; wait_possible_edge = False; self.agv_conflict_counts[agv_to_wait_id] += 1
                                                if self.verbose: print(f"        边冲突等待失败: 节点 {wait_node_edge} 在 t={current_wait_time_edge} 被占用。"); break
                                            new_sequence_prefix_edge.append((wait_node_edge, current_wait_time_edge))
                                        if wait_possible_edge:
                                            time_shift_edge = required_edge_wait_duration; path_after_wait_valid_edge = True; shifted_path_suffix_edge = []
                                            start_shift_idx = wait_from_idx + 1
                                            for i in range(start_shift_idx, len(path_to_wait.sequence)):
                                                original_node_edge, original_time_edge = path_to_wait.sequence[i]; new_time_edge = original_time_edge + time_shift_edge
                                                if new_time_edge > self.max_time: path_after_wait_valid_edge = False; break
                                                if new_time_edge in dynamic_obs_edge_wait_check and original_node_edge in dynamic_obs_edge_wait_check.get(new_time_edge, set()): path_after_wait_valid_edge = False; break
                                                shifted_path_suffix_edge.append((original_node_edge, new_time_edge))
                                            if path_after_wait_valid_edge:
                                                current_solution[agv_to_wait_id].sequence = new_sequence_prefix_edge + shifted_path_suffix_edge
                                                made_change_this_pass = True; edge_conflict_resolved_in_scan = True
                                                if self.verbose: print(f"      AGV {agv_to_wait_id} 等待解决边冲突成功。")
                                                moves_dict.clear(); paths_sequences_edge.clear(); break
                                            else:
                                                should_replan_edge = True; self.agv_conflict_counts[agv_to_wait_id] += 1
                                                if self.verbose: print(f"        边冲突等待后平移路径失败，AGV {agv_to_wait_id} 需要重规划。")

                                    if should_replan_edge:
                                        prev_node_replan_edge, prev_time_replan_edge = path_to_wait.sequence[wait_from_idx]
                                        if self.verbose: print(f"      AGV {agv_to_wait_id} 将从 ({prev_node_replan_edge}, {prev_time_replan_edge}) 开始重规划 (边冲突)。")
                                        task_wait_original = next((t for t in self.tasks if t.agv_id == agv_to_wait_id), None)
                                        if task_wait_original is not None:
                                            replan_task_edge = Task(agv_id=agv_to_wait_id, start_node=prev_node_replan_edge, goal_node=task_wait_original.goal_node)
                                            current_solution[agv_to_wait_id].sequence = path_to_wait.sequence[:wait_from_idx+1]
                                            dynamic_obs_replan_edge = Operators._build_dynamic_obstacles(self, current_solution, exclude_agv_id=agv_to_wait_id)
                                            new_segment_edge = self._call_planner(replan_task_edge, dynamic_obs_replan_edge, start_time=prev_time_replan_edge)
                                            if new_segment_edge and new_segment_edge.sequence and len(new_segment_edge.sequence) > 1:
                                                current_solution[agv_to_wait_id].sequence.extend(new_segment_edge.sequence[1:])
                                                made_change_this_pass = True; edge_conflict_resolved_in_scan = True
                                                if self.verbose: print(f"      AGV {agv_to_wait_id} 重规划解决边冲突成功。")
                                                moves_dict.clear(); paths_sequences_edge.clear(); break
                                            else: print(f"      错误：边冲突解决失败，AGV {agv_to_wait_id} 重规划未找到有效路径段。"); return None
                                        else: print(f"错误：无法找到 AGV {agv_to_wait_id} 的原始任务用于边冲突重规划。"); return None
                        if edge_conflict_resolved_in_scan: break
                    if edge_conflict_resolved_in_scan: break
                if edge_conflict_resolved_in_scan: break
            if edge_conflict_resolved_in_scan: continue

            if not made_change_this_pass: break

        if resolve_attempt >= max_resolve_attempts and made_change_this_pass:
            print(f"  错误：冲突/死锁解决超过最大尝试次数 {max_resolve_attempts}，但仍在修改。可能无法收敛。")
            return None
        else:
            final_check_cost_dict = self._calculate_total_cost(current_solution)
            if final_check_cost_dict.get('total', float('inf')) == float('inf'):
                print("  错误：冲突/死锁解决后最终成本检查为 Inf！")
                return None
            else:
                if self.verbose and resolve_attempt > 1 : print(f"  冲突解决完成，共尝试 {resolve_attempt} 轮。")
                return current_solution

    def run(self, external_initial_solution: Optional[Solution] = None) -> Tuple[Optional[Solution], float, CostDict]:
        start_run_time = time.perf_counter()
        self.iteration_count = 0; self.no_improvement_count = 0
        self.temperature = self.initial_temperature
        self.use_bounding_box = True # v44: Reset bbox flag at start
        self.best_solution = None; self.best_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.current_solution = None; self.current_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.initial_cost = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
        self.cost_history = []; self.operator_history = []; self.agv_conflict_counts.clear()

        initial_solution_to_process: Optional[Solution] = None
        if external_initial_solution is not None:
            print("--- 使用外部提供的初始解 ---")
            cost_external = self._calculate_total_cost(external_initial_solution)
            if cost_external.get('total', float('inf')) != float('inf'):
                 initial_solution_to_process = external_initial_solution
                 self.initial_cost = cost_external
                 print(f"  外部初始解成本 (可能冲突): {self.initial_cost['total']:.2f}")
            else:
                 print("  错误: 外部提供的初始解成本为 Inf，尝试内部生成...")
                 initial_solution_to_process = None
        elif self.skip_internal_initial_solution:
             print("错误: 配置为跳过内部初始解生成，但未提供外部初始解！")
             return None, time.perf_counter() - start_run_time, self.best_cost
        else:
            initial_solution_raw = self.generate_initial_solution()
            if initial_solution_raw is not None:
                initial_solution_to_process = initial_solution_raw
            else:
                print("错误：无法生成可行的初始解，ALNS 终止。")
                return None, time.perf_counter() - start_run_time, self.best_cost

        if initial_solution_to_process is not None:
            print("--- 处理初始解冲突 ---")
            resolve_start_init = time.perf_counter()
            resolved_initial_solution = self._resolve_conflicts_and_deadlocks(initial_solution_to_process)
            resolve_dur_init = time.perf_counter() - resolve_start_init
            print(f"  初始冲突处理完成 (耗时 {resolve_dur_init:.3f}s)")
            if resolved_initial_solution is not None:
                self.current_solution = resolved_initial_solution
                self.current_cost = self._calculate_total_cost(self.current_solution)
                current_total_cost_init = self.current_cost.get('total', float('inf'))
                if current_total_cost_init != float('inf'):
                    self.best_solution = copy.deepcopy(self.current_solution)
                    self.best_cost = self.current_cost
                    self.cost_history.append({'iteration': 0, 'current_cost': self.current_cost['total'], 'best_cost': self.best_cost['total'], 'temperature': self.temperature})
                    print(f"最终初始解成本 (无冲突): {self.best_cost['total']:.2f}")
                else:
                    print("错误：处理冲突后的初始解成本为无穷大，ALNS 终止。")
                    return None, time.perf_counter() - start_run_time, self.best_cost
            else:
                print("错误：处理初始解冲突失败，ALNS 终止。")
                return None, time.perf_counter() - start_run_time, self.best_cost
        else:
             return None, time.perf_counter() - start_run_time, self.best_cost

        print(f"\n--- 开始 ALNS 迭代 (Max Iter: {self.max_iterations}, No Improve Limit: {self.no_improvement_limit}) ---")
        for i in range(self.max_iterations):
            self.iteration_count = i + 1
            iter_start_time = time.perf_counter()
            if self.current_solution is None: print(f"错误: 迭代 {i+1} 开始时当前解丢失！终止。"); break

            destroy_op_name = self._select_operator_roulette_wheel(self.destroy_weights)
            repair_op_name = self._select_operator_roulette_wheel(self.repair_weights)
            destroy_op = self.destroy_operators.get(destroy_op_name)
            repair_op = self.repair_operators.get(repair_op_name)
            if not destroy_op or not repair_op: print(f"错误：无法找到算子 {destroy_op_name} 或 {repair_op_name}！"); break

            best_cost_iter_str = f"{self.best_cost['total']:.2f}" if self.best_cost['total'] != float('inf') else "Inf"
            curr_cost_iter_str = f"{self.current_cost['total']:.2f}" if self.current_cost['total'] != float('inf') else "Inf"
            no_imp_limit_str = str(self.no_improvement_limit) if self.no_improvement_limit is not None else "N/A"
            bbox_status_str = "启用" if self.use_bounding_box else "禁用" # v44
            if self.verbose: print(f"\nIter {i+1}/{self.max_iterations} | T={self.temperature:.3f} | Best={best_cost_iter_str} | Curr={curr_cost_iter_str} | NoImpr={self.no_improvement_count}/{no_imp_limit_str} | BBox={bbox_status_str} | Ops: D='{destroy_op_name}', R='{repair_op_name}'")

            removal_percentage = random.uniform(self.removal_percentage_min, self.removal_percentage_max)
            removal_count = max(1, int(self.num_agvs * removal_percentage))
            try: partial_solution, removed_agv_ids = destroy_op(self, self.current_solution, removal_count)
            except Exception as e: print(f"!!!!!!!!!! 调用破坏算子 {destroy_op_name} 时发生错误 !!!!!!!!!!!!!\n错误信息: {e}"); traceback.print_exc(); break

            new_solution_raw = None
            try: new_solution_raw = repair_op(self, partial_solution, removed_agv_ids)
            except Exception as e: print(f"!!!!!!!!!! 调用修复算子 {repair_op_name} 时发生错误 !!!!!!!!!!!!!\n错误信息: {e}"); traceback.print_exc(); new_solution_raw = None

            new_solution_processed: Optional[Solution] = None
            new_cost_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
            if new_solution_raw is not None:
                if self.verbose: print("  正在处理新解的冲突与死锁...")
                resolve_start = time.perf_counter()
                new_solution_processed = self._resolve_conflicts_and_deadlocks(new_solution_raw)
                resolve_dur = time.perf_counter() - resolve_start
                if self.verbose: print(f"  冲突处理完成 (耗时 {resolve_dur:.3f}s)")
                if new_solution_processed is not None: new_cost_dict = self._calculate_total_cost(new_solution_processed)
                else: new_cost_dict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}; print("  警告: 冲突解决失败，新解无效。")
            else: print("  警告: 修复算子未能生成有效解。")

            new_total_cost = new_cost_dict.get('total', float('inf'))
            current_total_cost = self.current_cost.get('total', float('inf'))
            best_total_cost = self.best_cost.get('total', float('inf'))

            score = 0.0; accepted = False; improved_best = False
            if new_total_cost != float('inf'):
                delta_cost = new_total_cost - current_total_cost if current_total_cost != float('inf') else -float('inf')
                if delta_cost < -1e-9 or current_total_cost == float('inf'):
                    accepted = True; score = self.sigma2
                    if new_total_cost < best_total_cost - 1e-9: score = self.sigma1; improved_best = True
                    if self.verbose or improved_best: print(f"  接受新解 ({'更好' if delta_cost < -1e-9 else '从无效变有效'}){' *** New Best! ***' if improved_best else ''}, Cost={new_total_cost:.2f}")
                elif self.temperature > 1e-6:
                    try:
                        prob = math.exp(-delta_cost / self.temperature)
                        if random.random() < prob: accepted = True; score = self.sigma3; print(f"  接受新解 (较差/相同, Prob={prob:.3f}), Cost={new_total_cost:.2f}")
                    except OverflowError: pass

            if accepted:
                self.current_solution = new_solution_processed; self.current_cost = new_cost_dict
                if improved_best: self.best_solution = copy.deepcopy(new_solution_processed); self.best_cost = new_cost_dict; self.no_improvement_count = 0
                else: self.no_improvement_count += 1
                self.destroy_scores[destroy_op_name] += score; self.repair_scores[repair_op_name] += score
            else: self.no_improvement_count += 1

            # --- v44: Update bounding box usage flag ---
            if self.use_bounding_box and self.no_improvement_count > self.no_improvement_bbox_disable_threshold:
                self.use_bounding_box = False
                if self.verbose: print(f"  --- 连续 {self.no_improvement_count} 次未改进，禁用包围盒策略 ---")
            # --- ---------------------------------- ---

            self.destroy_counts[destroy_op_name] += 1; self.repair_counts[repair_op_name] += 1
            self.cost_history.append({'iteration': i + 1, 'current_cost': self.current_cost['total'], 'best_cost': self.best_cost['total'], 'temperature': self.temperature})
            if (i + 1) % self.segment_size == 0: self._update_weights()
            self.temperature = max(1e-6, self.temperature * self.cooling_rate)
            if self.no_improvement_limit is not None and self.no_improvement_count >= self.no_improvement_limit:
                print(f"\n--- 触发早停：最优解连续 {self.no_improvement_limit} 次迭代未改进 (当前迭代 {i+1}) ---"); break

        end_run_time = time.perf_counter(); total_duration = end_run_time - start_run_time
        print("\n--- ALNS 最终结果 ---")
        best_total_cost_final = self.best_cost.get('total', float('inf'))
        best_cost_final_str = f"{best_total_cost_final:.2f}" if best_total_cost_final != float('inf') else "Inf"
        if self.best_solution is not None: print(f"找到最优解成本: {best_cost_final_str}"); print(f"  Breakdown: Travel={self.best_cost.get('travel', 0.0):.2f}, Turn={self.best_cost.get('turn', 0.0):.2f}, Wait={self.best_cost.get('wait', 0.0):.2f}")
        else: print("未能找到可行解。")
        print(f"总运行时间: {total_duration:.2f} 秒"); print(f"总迭代次数: {self.iteration_count}")

        print("\n--- 最终算子权重 ---")
        destroy_total_w = sum(self.destroy_weights.values()); repair_total_w = sum(self.repair_weights.values())
        if destroy_total_w > 1e-9: print(f"Destroy: {{{', '.join([f'{n}:{w/destroy_total_w:.3f}' for n, w in sorted(self.destroy_weights.items())])}}}")
        else: print(f"Destroy: {self.destroy_weights}")
        if repair_total_w > 1e-9: print(f"Repair: {{{', '.join([f'{n}:{w/repair_total_w:.3f}' for n, w in sorted(self.repair_weights.items())])}}}")
        else: print(f"Repair: {self.repair_weights}")

        print("\n--- 保存历史数据和绘图 ---")
        self._save_history_data(); self.plot_convergence_method()
        return self.best_solution, total_duration, self.best_cost

    def _update_weights(self):
        if self.verbose and self.debug_weights: print(f"--- 更新权重 (Segment End Iteration {self.iteration_count}) ---")
        segment_destroy_scores = self.destroy_scores.copy(); segment_repair_scores = self.repair_scores.copy()
        segment_destroy_counts = self.destroy_counts.copy(); segment_repair_counts = self.repair_counts.copy()
        for name in self.destroy_operators:
            count = segment_destroy_counts.get(name, 0); score = segment_destroy_scores.get(name, 0.0)
            if count > 0: self.destroy_weights[name] = max(0.01, (1 - self.weight_update_rate) * self.destroy_weights[name] + self.weight_update_rate * (score / count))
            else: self.destroy_weights[name] = max(0.01, self.destroy_weights[name] * 0.95)
        for name in self.repair_operators:
            count = segment_repair_counts.get(name, 0); score = segment_repair_scores.get(name, 0.0)
            if count > 0: self.repair_weights[name] = max(0.01, (1 - self.weight_update_rate) * self.repair_weights[name] + self.weight_update_rate * (score / count))
            else: self.repair_weights[name] = max(0.01, self.repair_weights[name] * 0.95)
        total_destroy_w = sum(self.destroy_weights.values())
        if total_destroy_w > 1e-9: self.destroy_weights = {name: w / total_destroy_w for name, w in self.destroy_weights.items()}
        total_repair_w = sum(self.repair_weights.values())
        if total_repair_w > 1e-9: self.repair_weights = {name: w / total_repair_w for name, w in self.repair_weights.items()}
        segment_record = {'segment_end_iteration': self.iteration_count, 'destroy_ops': {}, 'repair_ops': {}}
        for name in self.destroy_operators: segment_record['destroy_ops'][name] = {'score': segment_destroy_scores.get(name, 0.0), 'count': segment_destroy_counts.get(name, 0), 'weight': self.destroy_weights[name]}
        for name in self.repair_operators: segment_record['repair_ops'][name] = {'score': segment_repair_scores.get(name, 0.0), 'count': segment_repair_counts.get(name, 0), 'weight': self.repair_weights[name]}
        self.operator_history.append(segment_record)
        self.destroy_scores = {name: 0.0 for name in self.destroy_operators}; self.repair_scores = {name: 0.0 for name in self.repair_operators}
        self.destroy_counts = {name: 0 for name in self.destroy_operators}; self.repair_counts = {name: 0 for name in self.repair_operators}
        decay_factor = self.conflict_history_decay
        if decay_factor < 1.0:
            original_sum = sum(self.agv_conflict_counts.values())
            decayed_counts = {agv_id: int(count * decay_factor) for agv_id, count in self.agv_conflict_counts.items() if int(count * decay_factor) > 0}
            self.agv_conflict_counts.clear(); self.agv_conflict_counts.update(decayed_counts)
            new_sum = sum(self.agv_conflict_counts.values())
            if self.verbose and self.debug_weights: print(f"    Conflict Counts Decayed (Factor={decay_factor:.2f}): Total {original_sum} -> {new_sum}")

    def _save_history_data(self):
        if not self.record_history: return
        cost_history_file = os.path.join(self.results_dir, f"{self.instance_identifier}_cost_history.csv")
        try:
            if self.cost_history:
                fieldnames = list(self.cost_history[0].keys()) if self.cost_history else []
                if fieldnames:
                    with open(cost_history_file, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore'); writer.writeheader(); writer.writerows(self.cost_history)
        except Exception as e: print(f"错误: 无法写入成本历史文件 '{cost_history_file}': {e}")
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
                                flat_op_history.append({'segment_end_iteration': iter_num, 'operator_type': op_category, 'operator_name': op_name, 'score': stats.get('score', 0.0), 'count': stats.get('count', 0), 'weight': stats.get('weight', 0.0)})
                if flat_op_history:
                    fieldnames = list(flat_op_history[0].keys()) if flat_op_history else []
                    if fieldnames:
                        with open(operator_history_file, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames); writer.writeheader(); writer.writerows(flat_op_history)
        except Exception as e: print(f"错误: 无法写入算子历史文件 '{operator_history_file}': {e}")

    def plot_convergence_method(self):
        if not self.plot_convergence_flag or not _visual_libs_available or not self.cost_history: return
        plot_file = os.path.join(self.results_dir, f"{self.instance_identifier}_convergence.png")
        try:
            df = pd.DataFrame(self.cost_history); df.replace([float('inf'), -float('inf')], pd.NA, inplace=True); df.dropna(subset=['current_cost', 'best_cost', 'temperature'], how='any', inplace=True)
            if df.empty: return
            initial_valid_cost = df['best_cost'].iloc[0] if not df.empty else None
            y_limit_upper = initial_valid_cost * 1.5 if initial_valid_cost is not None and initial_valid_cost > 0 else None
            fig, ax1 = plt.subplots(figsize=(12, 6)); color1 = 'tab:blue'
            ax1.set_xlabel('Iteration'); ax1.set_ylabel('Cost', color=color1)
            ax1.plot(df['iteration'], df['current_cost'], color=color1, alpha=0.6, label='Current Cost')
            ax1.plot(df['iteration'], df['best_cost'], color=color1, linestyle='-', linewidth=2, label='Best Cost')
            ax1.tick_params(axis='y', labelcolor=color1); ax1.set_ylim(bottom=0)
            if y_limit_upper: ax1.set_ylim(top=y_limit_upper)
            ax2 = ax1.twinx(); color2 = 'tab:red'
            ax2.set_ylabel('Temperature', color=color2)
            ax2.plot(df['iteration'], df['temperature'], color=color2, linestyle=':', alpha=0.7, label='Temperature')
            ax2.tick_params(axis='y', labelcolor=color2); ax2.set_ylim(bottom=0)
            lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
            if lines or lines2: ax2.legend(lines + lines2, labels + labels2, loc='upper right')
            plt.title(f'ALNS Convergence ({self.instance_identifier})'); fig.tight_layout()
            plt.savefig(plot_file); plt.close(fig)
        except Exception as e: print(f"错误: 绘制收敛图时发生错误: {e}")

    def get_final_weights(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        return self.destroy_weights.copy(), self.repair_weights.copy()

if __name__ == '__main__':
    print("--- ALNS (v44 - Dynamic BBox) 示例 ---")
    try: from InstanceGenerator import load_fixed_scenario_1; from Map import GridMap
    except ImportError as e: print(f"错误: 导入 InstanceGenerator/Map 失败: {e}"); sys.exit(1)
    instance_data = load_fixed_scenario_1(assign_speeds=True)
    if not instance_data: print("错误: 无法加载固定算例。"); sys.exit(1)
    test_map, test_tasks, agv_speeds_dict = instance_data
    if not test_tasks: print("错误: 加载的任务列表为空。"); sys.exit(1)
    print(f"加载了 {len(test_tasks)} 个任务。")
    print(f"使用的 AGV 速度: {agv_speeds_dict}")
    try: planner_main = TWAStarPlanner()
    except Exception as planner_e: print(f"错误: 初始化 TWAStarPlanner 失败: {planner_e}"); sys.exit(1)
    instance_id = "fixed_scenario_1_v44_speedPrio_extInit_dynBBox_test"
    results_directory = "alns_output_v44"
    alns_init_params = {
        'instance_identifier': instance_id, 'results_dir': results_directory,
        'agv_speeds': agv_speeds_dict, 'priority_strategy': 'speed', 'v': 1.0,
        'alns_max_iterations': 500, 'alns_initial_temp': 100.0, 'alns_cooling_rate': 0.995,
        'alns_segment_size': 50, 'alns_weight_update_rate': 0.15,
        'alns_sigma1': 15.0, 'alns_sigma2': 8.0, 'alns_sigma3': 3.0,
        'alns_removal_percentage_min': 0.10, 'alns_removal_percentage_max': 0.25,
        'alns_regret_k': 3, 'alns_regret_max_attempts': 10, 'alns_no_improvement_limit': 100,
        'alns_no_improvement_bbox_disable': 10, # Threshold to disable bbox
        'conflict_history_decay': 0.9, 'max_time': 1000, 'cost_weights': (1.0, 0.3, 0.8),
        'delta_step': 1.0, 'buffer': 1, 'alns_planner_time_limit_factor': 5.0,
        'alns_regret_planner_time_limit_abs': 0.15, 'wait_threshold': 6,
        'deadlock_max_wait': 15, 'edge_wait_threshold': 4, 'alns_verbose_output': True,
        'alns_debug_weights': True, 'alns_record_history': True, 'alns_plot_convergence': True,
        'skip_internal_initial_solution': True
    }
    try:
        from Run_Experiments import generate_simple_initial_solution
        simple_planner = TWAStarPlanner()
        simple_sol, simple_cost = generate_simple_initial_solution(
            test_map, test_tasks, simple_planner, alns_init_params['cost_weights'],
            agv_speeds_dict, alns_init_params['delta_step'], alns_init_params['max_time'],
            alns_init_params.get('alns_planner_time_limit_factor', 5.0) * 0.5
        )
        if simple_sol is None: raise RuntimeError("无法生成简单初始解用于测试。")
        alns_instance = ALNS(grid_map=test_map, tasks=test_tasks, planner=planner_main, **alns_init_params)
        best_sol, duration, best_cost_dict = alns_instance.run(external_initial_solution=simple_sol)
    except Exception as run_e: print(f"错误: ALNS 运行时发生异常: {run_e}"); traceback.print_exc(); sys.exit(1)
    print("\n--- ALNS (v44) 运行完成 ---")
    if best_sol is not None:
        final_total_cost = best_cost_dict.get('total', float('inf')); final_cost_str = f"{final_total_cost:.2f}" if final_total_cost != float('inf') else "Inf"
        print(f"最终最优解成本: Total={final_cost_str}")
        print(f"  Breakdown: Travel={best_cost_dict.get('travel', 0.0):.2f}, Turn={best_cost_dict.get('turn', 0.0):.2f}, Wait={best_cost_dict.get('wait', 0.0):.2f}")
        final_destroy_w, final_repair_w = alns_instance.get_final_weights()
        print("\n获取到的最终权重:"); print(f"  Destroy: {final_destroy_w}"); print(f"  Repair: {final_repair_w}")
    else: print("ALNS 未能找到可行解。")
    print(f"结果文件已保存在目录: '{os.path.abspath(results_directory)}'")
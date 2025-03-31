# Operators.py-v8 (修复终点占用 Bug, 完整代码, 调整结构)
"""
包含自适应大邻域搜索 (ALNS) 算法使用的所有破坏和修复算子函数，
以及它们直接依赖的辅助函数。

与论文的关联:
- 破坏算子: 实现论文第 3.4 节描述的算子策略 (Random, Worst, Congestion, Related, Conflict History)。
- 修复算子: 实现论文第 3.5 节描述的算子策略 (Greedy, Regret, Wait Adjustment)。
    - 算子内部调用核心规划器 (TWA*, Planner.py v15+)，该规划器
      现在支持可选的 bounding_box 参数来实现区域分割 (论文 3.5.2)，
      以加速搜索。
    - 目标函数评估和约束满足与 TWA* 自身关联（见 Planner.py 文档）。
- 辅助函数: 支持算子实现，如构建动态障碍、计算相关性、计算包围盒、
             查找插入选项、检查冲突段、尝试等待调整等。

版本变更 (v7 -> v8):
- **结构**: 修改了修复算子末尾的返回逻辑。
- **保持**: v7 的终点占用 Bug 修复逻辑 (`_build_dynamic_obstacles`) 和其他功能保持不变。
"""
import random
import math
import copy
from collections import defaultdict, Counter
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict, Set, Callable, NamedTuple

# --- 类型提示导入 ---
if TYPE_CHECKING:
    from ALNS import ALNS
    from DataTypes import Solution, Task, Path, CostDict, TimeStep, Node, DynamicObstacles, State
    from Map import GridMap
    from Planner import TWAStarPlanner

# --- 从 DataTypes 导入必要类 ---
try:
    from DataTypes import Path as PathType
    from DataTypes import Solution as SolutionType
    from DataTypes import Task as TaskType
    from DataTypes import DynamicObstacles as DynObsType
    from DataTypes import State as StateType
    from DataTypes import Node as NodeType
except ImportError as e:
     print(f"错误: 导入 Operators 依赖项失败 (DataTypes): {e}")
     PathType = type('Path', (object,), {})
     SolutionType = Dict
     TaskType = type('Task', (object,), {})
     DynObsType = Dict
     StateType = Tuple
     NodeType = Tuple

# --- InsertionCost 类  ---
class InsertionCost:
    """存储单个 AGV 的多个插入选项及其成本，用于计算后悔值。"""
    def __init__(self, agv_id: int):
        self.agv_id = agv_id
        self.options: List[Tuple[Optional['PathType'], float]] = []
        self.best_cost = float('inf')
        self.second_best_cost = float('inf')
        self.regret = 0.0

    def add_option(self, path: Optional['PathType'], cost: float):
        if path is not None and not isinstance(path, PathType):
            raise TypeError(f"路径必须是 Path 类型或 None，得到 {type(path)}")
        self.options.append((path, cost))
        if cost < self.best_cost:
            self.second_best_cost = self.best_cost
            self.best_cost = cost
        elif cost < self.second_best_cost:
            self.second_best_cost = cost

    def calculate_regret(self, k: int):
        if not self.options:
            self.regret = 0.0; return
        self.options.sort(key=lambda x: x[1])
        self.best_cost = self.options[0][1] if self.options else float('inf')
        self.second_best_cost = self.options[1][1] if len(self.options) > 1 else float('inf')
        if self.best_cost != float('inf') and self.second_best_cost != float('inf'):
            self.regret = self.second_best_cost - self.best_cost
        else: self.regret = 0.0
        self.regret = max(0.0, self.regret)

    def get_best_path(self) -> Optional['PathType']:
        if not self.options: return None
        best_option = self.options[0]
        return best_option[0] if best_option[1] != float('inf') else None

    def __lt__(self, other: 'InsertionCost'):
        return self.regret > other.regret

# ==================================
# --- 辅助函数 ---
# ==================================

# --- _build_dynamic_obstacles (保持 v7: 修复终点占用 Bug) ---
def _build_dynamic_obstacles(
    alns_instance: 'ALNS', # 需要访问 alns_instance 获取任务和 T_max
    solution: 'SolutionType',
    exclude_agv_id: Optional[int] = None
) -> 'DynObsType':
    """
    根据当前（部分）解构建动态障碍物字典。
    用于传递给 TWA* 规划器，以满足节点冲突约束 (论文公式 12)。

    **v7 改进:** 解决了终点占用问题。如果一个 AGV 到达了它的目标节点，
    那么该目标节点将在其到达时间之后的所有时间步（直到 T_max）被标记为占用。
    """
    dynamic_obstacles: DynObsType = {}
    if not isinstance(solution, dict): return dynamic_obstacles

    tasks_list = getattr(alns_instance, 'tasks', [])
    if not tasks_list: print("警告 (_build_dynamic_obstacles): 无法从 ALNS 实例获取任务列表。")
    tasks_map = {task.agv_id: task for task in tasks_list}

    max_time = getattr(alns_instance, 'max_time', 0)
    if max_time <= 0:
         print("警告 (_build_dynamic_obstacles): 无法从 ALNS 实例获取有效的 max_time。终点占用可能不完整。")
         max_time = 0

    for agv_id, path in solution.items():
        if agv_id == exclude_agv_id: continue
        if not isinstance(path, PathType) or not path.sequence: continue

        for node, t in path.sequence:
            if t not in dynamic_obstacles: dynamic_obstacles[t] = set()
            dynamic_obstacles[t].add(node)

        task = tasks_map.get(agv_id)
        if not task: continue

        last_node, t_end = path.sequence[-1]
        goal_node = task.goal_node

        if last_node == goal_node:
            for t_future in range(t_end + 1, max_time + 1):
                if t_future not in dynamic_obstacles: dynamic_obstacles[t_future] = set()
                dynamic_obstacles[t_future].add(goal_node)

    return dynamic_obstacles

# --- _calculate_relatedness (保持 v6) ---
def _calculate_relatedness(
    alns_instance: 'ALNS',
    path1: Optional['PathType'],
    path2: Optional['PathType']
) -> float:
    """
    计算两条路径之间的相关性得分 (用于 Related Removal 算子)。
    相关性基于路径在时间和空间上的接近程度。
    """
    if not path1 or not path2 or not isinstance(path1, PathType) or not isinstance(path2, PathType) or not path1.sequence or not path2.sequence:
        return 0.0

    grid_map = getattr(alns_instance, 'grid_map', None)
    max_time = getattr(alns_instance, 'max_time', 1)
    if not grid_map: print("警告 (Relatedness): 无法访问 ALNS 实例的 grid_map。"); return 0.0

    map_width = getattr(grid_map, 'width', 1); map_height = getattr(grid_map, 'height', 1)
    max_dist_factor = max(map_width, map_height) * math.sqrt(2); max_time_factor = max(1, max_time)
    if max_dist_factor < 1e-6: max_dist_factor = 1.0
    if max_time_factor < 1e-6: max_time_factor = 1.0

    relatedness_sum = 0.0; comparisons = len(path1.sequence)
    path2_nodes_times = {(node,t) for node, t in path2.sequence}

    for idx1, (node1, t1) in enumerate(path1.sequence):
        min_spatial_dist_sq = float('inf'); min_temporal_dist = float('inf')
        for node2, t2 in path2_nodes_times:
            time_diff = abs(t1 - t2); dx = node1[0] - node2[0]; dy = node1[1] - node2[1]; dist_sq = dx**2 + dy**2
            min_spatial_dist_sq = min(min_spatial_dist_sq, dist_sq); min_temporal_dist = min(min_temporal_dist, time_diff)

        spatial_relatedness = 0.0
        if min_spatial_dist_sq != float('inf'):
            spatial_divisor = max_dist_factor * 0.2 + 1e-6
            spatial_relatedness = max(0.0, 1.0 - math.sqrt(min_spatial_dist_sq) / spatial_divisor)

        temporal_relatedness = 0.0
        if min_temporal_dist != float('inf'):
            temporal_divisor = max_time_factor * 0.1 + 1e-6
            temporal_relatedness = max(0.0, 1.0 - min_temporal_dist / temporal_divisor)

        point_relatedness = (spatial_relatedness * 0.6 + temporal_relatedness * 0.4)
        relatedness_sum += point_relatedness

    return relatedness_sum / comparisons if comparisons > 0 else 0.0

# --- _calculate_bounding_box (保持 v6) ---
def _calculate_bounding_box(
    start_node: 'NodeType',
    goal_node: 'NodeType',
    map_width: int,
    map_height: int,
    buffer: int
) -> Tuple[int, int, int, int]:
    """
    (内部辅助) 计算包含起点和终点的（可能扩展的）包围盒。
    用于区域分割优化 (论文 3.5.2)。
    """
    if not (isinstance(start_node, tuple) and len(start_node) == 2): raise ValueError("start_node 格式错误")
    if not (isinstance(goal_node, tuple) and len(goal_node) == 2): raise ValueError("goal_node 格式错误")
    if buffer < 0: raise ValueError("buffer 不能为负数")

    x_coords = [start_node[0], goal_node[0]]; y_coords = [start_node[1], goal_node[1]]
    min_x_nobuf = min(x_coords); max_x_nobuf = max(x_coords)
    min_y_nobuf = min(y_coords); max_y_nobuf = max(y_coords)
    min_x = max(0, min_x_nobuf - buffer); max_x = min(map_width - 1, max_x_nobuf + buffer)
    min_y = max(0, min_y_nobuf - buffer); max_y = min(map_height - 1, max_y_nobuf + buffer)
    if min_x > max_x: min_x = max_x = start_node[0]
    if min_y > max_y: min_y = max_y = start_node[1]
    return min_x, max_x, min_y, max_y

# --- _find_insertion_options (保持 v7: 传递 alns_instance 给 _build) ---
def _find_insertion_options(
    alns_instance: 'ALNS',
    task: 'TaskType',
    current_solution: 'SolutionType',
    k: int,
    max_regret_attempts: int = 5
) -> 'InsertionCost':
    """
    为单个 AGV 查找 k 个最佳插入选项及其成本 (用于 Regret Insertion)。
    应用区域分割优化 (论文 3.5.2)。
    """
    insertion_info = InsertionCost(task.agv_id)
    dynamic_obstacles_base = _build_dynamic_obstacles(alns_instance, current_solution) # v7

    bbox: Optional[Tuple[int, int, int, int]] = None
    try:
        grid_map = getattr(alns_instance, 'grid_map', None); buffer_val = getattr(alns_instance, 'buffer', 0)
        if grid_map and hasattr(grid_map, 'width') and hasattr(grid_map, 'height'):
            bbox = _calculate_bounding_box(task.start_node, task.goal_node, grid_map.width, grid_map.height, buffer_val)
        else: print(f"警告 (Regret): 无法访问地图或 buffer。不使用区域分割。")
    except ValueError as ve: print(f"警告 (Regret): 检查 bbox 输入失败: {ve}。不使用区域分割。")
    except Exception as bbox_e: print(f"警告 (Regret): 计算 AGV {task.agv_id} 包围盒时异常: {bbox_e}。不使用区域分割。")

    best_path: Optional[PathType] = None
    try:
        if hasattr(alns_instance, '_call_planner'):
             best_path = alns_instance._call_planner(task, dynamic_obstacles_base, start_time=0, bounding_box=bbox)
        else: print("错误 (Regret): ALNS 实例缺少 _call_planner 方法。")
    except TypeError as te:
        if 'bounding_box' in str(te):
            print(f"警告 (Regret): ALNS._call_planner 不支持 bbox 参数。尝试无 bbox 调用。")
            try: best_path = alns_instance._call_planner(task, dynamic_obstacles_base, start_time=0)
            except Exception as call_e_no_bbox: print(f"错误: 调用 ALNS._call_planner (无 bbox) 失败: {call_e_no_bbox}")
        else: print(f"错误: 调用 ALNS._call_planner 时发生 TypeError: {te}")
    except Exception as call_e: print(f"错误: 调用 ALNS._call_planner 时发生未知异常: {call_e}")

    best_cost = float('inf')
    if best_path and isinstance(best_path, PathType) and best_path.sequence:
        try:
            grid_map_eval = getattr(alns_instance, 'grid_map', None)
            if grid_map_eval:
                 cost = best_path.get_cost(grid_map_eval, getattr(alns_instance, 'alpha', 1.0), getattr(alns_instance, 'beta', 0.0), getattr(alns_instance, 'gamma_wait', 0.0), getattr(alns_instance, 'v', 1.0), getattr(alns_instance, 'delta_step', 1.0)).get('total', float('inf'))
            else: cost = float('inf')
        except Exception as cost_e: print(f"警告: 计算最优路径成本失败 (AGV {task.agv_id}): {cost_e}"); cost = float('inf')
        if cost != float('inf'): insertion_info.add_option(best_path, cost); best_cost = cost
        else: best_path = None

    if best_path and k > 1:
        secondary_limit = getattr(alns_instance, 'regret_planner_time_limit_abs', 0.1)
        tried_alternatives = 0; attempt_count = 0; max_attempts = max(1, max_regret_attempts)
        path_signature = tuple(best_path.sequence)
        eligible_indices = [i for i in range(1, len(best_path.sequence) - 1) if best_path.sequence[i] != best_path.sequence[i-1]]
        random.shuffle(eligible_indices); blocked_states_cache: Set[StateType] = set()

        for block_idx in eligible_indices:
            if tried_alternatives >= k - 1 or len(insertion_info.options) >= k: break
            if attempt_count >= max_attempts: break
            attempt_count += 1

            state_to_block = best_path.sequence[block_idx]
            if state_to_block in blocked_states_cache: continue
            blocked_states_cache.add(state_to_block)

            temp_dynamic_obstacles = copy.deepcopy(dynamic_obstacles_base)
            block_t = state_to_block[1]
            if block_t not in temp_dynamic_obstacles: temp_dynamic_obstacles[block_t] = set()
            temp_dynamic_obstacles[block_t].add(state_to_block[0])

            alternative_path: Optional[PathType] = None
            try:
                planner_inst = getattr(alns_instance, 'planner', None); grid_map_inst = getattr(alns_instance, 'grid_map', None)
                if planner_inst and grid_map_inst:
                     alternative_path = planner_inst.plan(grid_map_inst, task, temp_dynamic_obstacles, getattr(alns_instance, 'max_time', 400), getattr(alns_instance, 'cost_weights', (1.0, 0.0, 0.0)), getattr(alns_instance, 'v', 1.0), getattr(alns_instance, 'delta_step', 1.0), start_time=0, time_limit=secondary_limit, bounding_box=bbox)
                else: print("错误 (Regret Alt): 无法访问 planner 或 grid_map。")
            except TypeError as te:
                if 'bounding_box' in str(te):
                     print(f"警告 (Regret Alt): Planner.plan 不支持 bbox。尝试无 bbox 调用。")
                     try: alternative_path = planner_inst.plan(grid_map_inst, task, temp_dynamic_obstacles, getattr(alns_instance, 'max_time', 400), getattr(alns_instance, 'cost_weights', (1.0, 0.0, 0.0)), getattr(alns_instance, 'v', 1.0), getattr(alns_instance, 'delta_step', 1.0), start_time=0, time_limit=secondary_limit)
                     except Exception as plan_e_no_bbox: print(f"错误: 调用 Planner.plan (无 bbox) 失败: {plan_e_no_bbox}")
                else: print(f"错误: 调用 Planner.plan 时发生 TypeError: {te}")
            except Exception as plan_e: print(f"错误: 调用 Planner.plan 失败: {plan_e}")

            if alternative_path and isinstance(alternative_path, PathType) and alternative_path.sequence:
                alt_sig = tuple(alternative_path.sequence)
                if alt_sig != path_signature:
                    try:
                        grid_map_eval = getattr(alns_instance, 'grid_map', None)
                        if grid_map_eval: alt_cost = alternative_path.get_cost(grid_map_eval, getattr(alns_instance, 'alpha', 1.0), getattr(alns_instance, 'beta', 0.0), getattr(alns_instance, 'gamma_wait', 0.0), getattr(alns_instance, 'v', 1.0), getattr(alns_instance, 'delta_step', 1.0)).get('total', float('inf'))
                        else: alt_cost = float('inf')
                    except Exception as alt_cost_e: print(f"警告: 计算次优路径成本失败 (AGV {task.agv_id}): {alt_cost_e}"); alt_cost = float('inf')
                    if alt_cost != float('inf') and alt_cost > best_cost + 1e-6:
                        insertion_info.add_option(alternative_path, alt_cost); tried_alternatives += 1
    return insertion_info

# --- _check_sequence_conflicts_segment (保持 v7: 传递 alns_instance 给 _build) ---
def _check_sequence_conflicts_segment(
    alns_instance: 'ALNS', # 需要传递 alns_instance 以构建正确的障碍
    current_solution: 'SolutionType',
    agv_id_to_check: int,
    segment_to_check: List['StateType']
) -> bool:
    """检查路径段是否与（不包括自身的）动态障碍冲突。"""
    dynamic_obs_check = _build_dynamic_obstacles(alns_instance, current_solution, exclude_agv_id=agv_id_to_check) # v7
    for node, t in segment_to_check:
        if not isinstance(node, tuple): print(f"警告: 冲突检查遇到非元组节点: {node}"); continue
        obstacles_at_t = dynamic_obs_check.get(t)
        if obstacles_at_t and node in obstacles_at_t: return True
    return False

# --- _attempt_insert_wait (保持 v7: 适配 _check_sequence_conflicts_segment 签名) ---
def _attempt_insert_wait(
    alns_instance: 'ALNS',
    current_solution: 'SolutionType',
    agv_id: int,
    original_sequence: List['StateType'],
    idx: int,
    wait_duration: int
) -> Tuple[Optional[List['StateType']], float]:
    """尝试在路径指定索引处插入等待。"""
    if idx < 0 or idx >= len(original_sequence) - 1: return None, float('inf')
    node, current_t = original_sequence[idx]
    new_sequence_part = [(node, current_t + 1 + w) for w in range(wait_duration)]
    shifted_sequence_part = []
    max_time_limit = getattr(alns_instance, 'max_time', float('inf'))
    for n, t in original_sequence[idx+1:]:
        new_time = t + wait_duration
        if new_time > max_time_limit: return None, float('inf')
        shifted_sequence_part.append((n, new_time))
    trial_sequence = original_sequence[:idx+1] + new_sequence_part + shifted_sequence_part

    if _check_sequence_conflicts_segment(alns_instance, current_solution, agv_id, new_sequence_part): return None, float('inf') # v7
    if _check_sequence_conflicts_segment(alns_instance, current_solution, agv_id, shifted_sequence_part): return None, float('inf') # v7

    new_total_cost = float('inf')
    try:
        temp_solution = copy.deepcopy(current_solution); temp_solution[agv_id] = PathType(agv_id=agv_id, sequence=trial_sequence)
        new_total_cost = alns_instance._calculate_total_cost(temp_solution).get('total', float('inf'))
    except Exception as cost_e: print(f"错误 (InsertWait): 计算成本时失败: {cost_e}")
    return (trial_sequence, new_total_cost) if new_total_cost != float('inf') else (None, float('inf'))

# --- _attempt_delete_wait (保持 v7: 适配 _check_sequence_conflicts_segment 签名) ---
def _attempt_delete_wait(
    alns_instance: 'ALNS',
    current_solution: 'SolutionType',
    agv_id: int,
    original_sequence: List['StateType'],
    idx: int,
    wait_duration_deleted: int
) -> Tuple[Optional[List['StateType']], float]:
    """尝试在路径指定索引处删除等待。"""
    if idx < wait_duration_deleted or idx >= len(original_sequence): return None, float('inf')
    shifted_sequence_part = []
    for n, t in original_sequence[idx+1:]:
         new_time = t - wait_duration_deleted
         if new_time < 0: return None, float('inf')
         shifted_sequence_part.append((n, new_time))
    trial_sequence = original_sequence[:idx - wait_duration_deleted + 1] + shifted_sequence_part
    max_time_limit = getattr(alns_instance, 'max_time', float('inf'))
    if trial_sequence and trial_sequence[-1][1] > max_time_limit: return None, float('inf')

    if _check_sequence_conflicts_segment(alns_instance, current_solution, agv_id, shifted_sequence_part): return None, float('inf') # v7

    new_total_cost = float('inf')
    try:
        temp_solution = copy.deepcopy(current_solution); temp_solution[agv_id] = PathType(agv_id=agv_id, sequence=trial_sequence)
        new_total_cost = alns_instance._calculate_total_cost(temp_solution).get('total', float('inf'))
    except Exception as cost_e: print(f"错误 (DeleteWait): 计算成本时失败: {cost_e}")
    return (trial_sequence, new_total_cost) if new_total_cost != float('inf') else (None, float('inf'))


# ==================================
# --- 破坏算子 (Destroy Operators) ---
# (v8: 补充 v6 完整实现)
# ==================================

def random_removal(
    alns_instance: 'ALNS',
    solution: 'SolutionType',
    removal_count: int
) -> Tuple['SolutionType', List[int]]:
    """
    随机移除算子 (对应论文 3.4.1 节)。
    完全随机地选择指定数量的 AGV 移除其路径。
    """
    if not solution: return {}, []
    partial_solution = copy.deepcopy(solution)
    agvs_in_solution = [aid for aid, p in partial_solution.items() if p is not None and isinstance(p, PathType) and p.sequence]
    if not agvs_in_solution: return {}, []
    actual_removal_count = min(removal_count, len(agvs_in_solution))
    if actual_removal_count <= 0: return partial_solution, []
    removed_ids = random.sample(agvs_in_solution, actual_removal_count)
    for agv_id in removed_ids:
        if agv_id in partial_solution: del partial_solution[agv_id]
    return partial_solution, removed_ids

def worst_removal(
    alns_instance: 'ALNS',
    solution: 'SolutionType',
    removal_count: int
) -> Tuple['SolutionType', List[int]]:
    """
    最差移除算子 (对应论文 3.4.2 节)。
    移除对总目标函数贡献成本最高的 AGV 的路径。
    """
    if not solution: return {}, []
    costs: List[Tuple[float, int]] = []
    valid_agvs = []
    grid_map_eval = getattr(alns_instance, 'grid_map', None)
    if not grid_map_eval: print("错误 (WorstRemoval): 无法访问地图以计算成本。"); return copy.deepcopy(solution), []

    for agv_id, path in solution.items():
        if path and isinstance(path, PathType) and path.sequence:
            valid_agvs.append(agv_id); cost = float('inf')
            try:
                cost = path.get_cost(grid_map_eval, getattr(alns_instance, 'alpha', 1.0), getattr(alns_instance, 'beta', 0.0), getattr(alns_instance, 'gamma_wait', 0.0), getattr(alns_instance, 'v', 1.0), getattr(alns_instance, 'delta_step', 1.0)).get('total', float('inf'))
            except Exception as e: print(f"警告 (WorstRemoval): 计算 AGV {agv_id} 成本失败: {e}")
            if cost != float('inf'): costs.append((cost, agv_id))

    if not valid_agvs: return copy.deepcopy(solution), []
    costs.sort(key=lambda item: item[0], reverse=True)
    actual_removal_count = min(removal_count, len(costs))
    if actual_removal_count <= 0: return copy.deepcopy(solution), []
    removed_ids = [agv_id for cost, agv_id in costs[:actual_removal_count]]
    partial_solution = copy.deepcopy(solution)
    for agv_id in removed_ids:
        if agv_id in partial_solution: del partial_solution[agv_id]
    return partial_solution, removed_ids

def related_removal(
    alns_instance: 'ALNS',
    solution: 'SolutionType',
    removal_count: int
) -> Tuple['SolutionType', List[int]]:
    """
    相关移除算子 (对应论文 3.4.4 节)。
    随机选择一个 AGV，然后移除与其路径在时空上最相关的其他 AGV。
    """
    if not solution: return {}, []
    partial_solution = copy.deepcopy(solution)
    agvs_in_solution = [aid for aid, p in partial_solution.items() if p is not None and isinstance(p, PathType) and p.sequence]
    if not agvs_in_solution or removal_count <= 0: return partial_solution, []
    actual_removal_count = min(removal_count, len(agvs_in_solution))
    if actual_removal_count <= 0: return partial_solution, []

    seed_agv_id = random.choice(agvs_in_solution)
    removed_ids = [seed_agv_id]
    if seed_agv_id in partial_solution: del partial_solution[seed_agv_id]
    deterministic_factor = 4

    while len(removed_ids) < actual_removal_count:
        if not partial_solution: break
        candidates = list(partial_solution.keys())
        if not candidates: break

        relatedness_scores: List[Tuple[float, int]] = []
        last_removed_id = removed_ids[-1]
        path_last = solution.get(last_removed_id)

        for candidate_id in candidates:
            path_candidate = partial_solution.get(candidate_id)
            relatedness = _calculate_relatedness(alns_instance, path_last, path_candidate)
            relatedness_scores.append((relatedness, candidate_id))

        if not relatedness_scores: break
        relatedness_scores.sort(key=lambda item: item[0], reverse=True)
        rand_val = random.random(); max_index = len(relatedness_scores) - 1
        index_to_pick = min(int(len(relatedness_scores) * (rand_val ** deterministic_factor)), max_index)
        index_to_pick = max(0, index_to_pick)
        agv_to_remove = relatedness_scores[index_to_pick][1]
        removed_ids.append(agv_to_remove)
        if agv_to_remove in partial_solution: del partial_solution[agv_to_remove]

    return partial_solution, removed_ids

def congestion_removal(
    alns_instance: 'ALNS',
    solution: 'SolutionType',
    removal_count: int
) -> Tuple['SolutionType', List[int]]:
    """
    拥挤度移除算子 (对应论文 3.4.3 节)。
    移除路径经过的时空节点最拥挤的 AGV。
    """
    if not solution: return {}, []
    node_time_counts: Dict[StateType, int] = defaultdict(int); max_t = 0
    for path in solution.values():
        if path and isinstance(path, PathType) and path.sequence:
            max_t = max(max_t, path.get_makespan())
            for state in path.sequence: node_time_counts[state] += 1

    path_congestion_scores: List[Tuple[float, int]] = []
    valid_agvs = []
    for agv_id, path in solution.items():
        if path and isinstance(path, PathType) and path.sequence:
            valid_agvs.append(agv_id); congestion_score = 0.0
            for state in path.sequence:
                 occupancy = node_time_counts.get(state, 0)
                 if occupancy > 1: congestion_score += float(occupancy - 1)
            path_len = len(path.sequence); normalized_score = congestion_score / path_len if path_len > 0 else 0
            path_congestion_scores.append((normalized_score, agv_id))

    if not valid_agvs: return copy.deepcopy(solution), []
    path_congestion_scores.sort(key=lambda item: item[0], reverse=True)
    actual_removal_count = min(removal_count, len(path_congestion_scores))
    if actual_removal_count <= 0: return copy.deepcopy(solution), []
    removed_ids = [agv_id for score, agv_id in path_congestion_scores[:actual_removal_count]]
    partial_solution = copy.deepcopy(solution)
    for agv_id in removed_ids:
        if agv_id in partial_solution: del partial_solution[agv_id]
    return partial_solution, removed_ids

def conflict_history_removal(
    alns_instance: 'ALNS',
    solution: 'SolutionType',
    removal_count: int
) -> Tuple['SolutionType', List[int]]:
    """
    冲突历史移除算子 (对应论文 3.4.5 节)。
    移除近期参与冲突（等待或被重规划）次数最多的 AGV。
    """
    if not solution: return {}, []
    agvs_in_solution = [aid for aid, p in solution.items() if p is not None and isinstance(p, PathType) and p.sequence]
    if not agvs_in_solution: return {}, []

    conflict_counter = getattr(alns_instance, 'agv_conflict_counts', Counter())
    conflict_counts = Counter({agv_id: conflict_counter.get(agv_id, 0) for agv_id in agvs_in_solution})
    sorted_by_conflict = conflict_counts.most_common()
    actual_removal_count = min(removal_count, len(sorted_by_conflict))
    if actual_removal_count <= 0: return copy.deepcopy(solution), []

    removed_ids = [agv_id for agv_id, count in sorted_by_conflict[:actual_removal_count]]
    partial_solution = copy.deepcopy(solution)
    for agv_id in removed_ids:
        if agv_id in partial_solution: del partial_solution[agv_id]

    verbose = getattr(alns_instance, 'verbose', False); debug_weights = getattr(alns_instance, 'debug_weights', False)
    if verbose and debug_weights:
        removed_counts = [conflict_counts[aid] for aid in removed_ids]
        print(f"    ConflictHistoryRemoval: Removed {len(removed_ids)} AGVs: {removed_ids} (Counts: {removed_counts})")
    return partial_solution, removed_ids


# ==================================
# --- 修复算子 (Repair Operators) ---
# ==================================

# --- greedy_insertion ---
def greedy_insertion(
    alns_instance: 'ALNS',
    partial_solution: 'SolutionType',
    removed_ids: List[int]
) -> Optional['SolutionType']:
    """
    贪婪插入算子 (对应论文 3.5.3 节)。
    按随机顺序，依次为每个被移除的 AGV 规划成本最低的无冲突路径并插入。
    应用区域分割优化 (论文 3.5.2)。
    """
    if not isinstance(partial_solution, dict): return None
    solution = copy.deepcopy(partial_solution)
    tasks_to_insert = [t for t in getattr(alns_instance, 'tasks', []) if t.agv_id in removed_ids]
    if not tasks_to_insert: return solution
    random.shuffle(tasks_to_insert)

    for task in tasks_to_insert:
        agv_id = task.agv_id
        dynamic_obstacles = _build_dynamic_obstacles(alns_instance, solution) # v7
        bbox: Optional[Tuple[int, int, int, int]] = None
        try:
            grid_map = getattr(alns_instance, 'grid_map', None); buffer_val = getattr(alns_instance, 'buffer', 0)
            if grid_map and hasattr(grid_map, 'width') and hasattr(grid_map, 'height'):
                bbox = _calculate_bounding_box(task.start_node, task.goal_node, grid_map.width, grid_map.height, buffer_val)
            # else: print(f"警告 (Greedy): 无法访问地图或 buffer。不使用区域分割。") # 可选打印
        except Exception as bbox_e: print(f"警告 (Greedy): 计算 AGV {agv_id} 包围盒失败: {bbox_e}。")

        new_path: Optional[PathType] = None
        try:
            if hasattr(alns_instance, '_call_planner'):
                new_path = alns_instance._call_planner(task, dynamic_obstacles, start_time=0, bounding_box=bbox)
            # else: print("错误 (Greedy): ALNS 实例缺少 _call_planner 方法。") # 可选打印
        except TypeError as te:
            if 'bounding_box' in str(te):
                # print(f"警告 (Greedy): ALNS._call_planner 不支持 bbox。尝试无 bbox 调用。") # 可选打印
                try: new_path = alns_instance._call_planner(task, dynamic_obstacles, start_time=0)
                except Exception as call_e_no_bbox: print(f"错误: 调用 ALNS._call_planner (无 bbox) 失败: {call_e_no_bbox}")
            else: print(f"错误: 调用 ALNS._call_planner 时发生 TypeError: {te}")
        except Exception as call_e: print(f"错误: 调用 ALNS._call_planner 失败: {call_e}")

        if new_path and isinstance(new_path, PathType) and new_path.sequence:
            solution[agv_id] = new_path
        else:
            if getattr(alns_instance, 'verbose', False):
                 print(f"    Greedy Insert: 失败，无法为 AGV {agv_id} 找到路径。")
            return None # 修复失败

    num_expected_agvs = getattr(alns_instance, 'num_agvs', len(getattr(alns_instance, 'tasks', [])))
    if len(solution) == num_expected_agvs:
         final_cost = float('inf')
         try:
             final_cost = alns_instance._calculate_total_cost(solution).get('total', float('inf'))
         except Exception as final_cost_e:
             print(f"错误 (Greedy): 计算最终成本失败: {final_cost_e}")
             return None 
         # 检查成本是否有效
         if final_cost != float('inf'):
             return solution # 修复成功
         else:
             if getattr(alns_instance, 'verbose', False):
                 print("    Greedy Insert: 失败，最终成本为 Inf")
             return None 
    else:
         if getattr(alns_instance, 'verbose', False):
             num_actual_agvs = len(solution)
             print(f"    Greedy Insert: 失败，最终 AGV 数量 ({num_actual_agvs}) != 预期 ({num_expected_agvs})。")
         return None 

# --- regret_insertion  ---
def regret_insertion(
    alns_instance: 'ALNS',
    partial_solution: 'SolutionType',
    removed_ids: List[int]
) -> Optional['SolutionType']:
    """
    后悔插入算子 (对应论文 3.5.4 节)。
    优先插入“后悔值”最高的 AGV。内部调用 _find_insertion_options (v7)，
    该函数应用区域分割优化并使用修复后的动态障碍构建。
    """
    if not isinstance(partial_solution, dict): return None
    solution = copy.deepcopy(partial_solution)
    tasks_map = {t.agv_id: t for t in getattr(alns_instance, 'tasks', []) if t.agv_id in removed_ids}
    if not tasks_map and removed_ids: print("错误 (Regret): 无法获取 AGV 任务信息。"); return None
    if not removed_ids: return solution

    unassigned_agv_ids = removed_ids[:]
    iteration = 0; max_regret_iterations = len(unassigned_agv_ids)

    while unassigned_agv_ids and iteration < max_regret_iterations:
        iteration += 1
        insertion_candidates: List[InsertionCost] = []
        regret_k = getattr(alns_instance, 'regret_k', 2)

        for agv_id in unassigned_agv_ids:
            task = tasks_map.get(agv_id)
            if not task: print(f"警告 (Regret): 找不到 AGV {agv_id} 的任务。"); continue
            candidate_info = _find_insertion_options(alns_instance, task, solution, regret_k) # v7
            candidate_info.calculate_regret(regret_k)
            insertion_candidates.append(candidate_info)

        if not insertion_candidates: break
        insertion_candidates.sort(key=lambda c: c.regret, reverse=True)
        best_candidate_to_insert = insertion_candidates[0]
        agv_id_to_insert = best_candidate_to_insert.agv_id
        best_path_for_agv = best_candidate_to_insert.get_best_path()

        if best_path_for_agv and isinstance(best_path_for_agv, PathType):
            solution[agv_id_to_insert] = best_path_for_agv
            if agv_id_to_insert in unassigned_agv_ids: unassigned_agv_ids.remove(agv_id_to_insert)
        else:
            if agv_id_to_insert in unassigned_agv_ids: unassigned_agv_ids.remove(agv_id_to_insert)

    if not unassigned_agv_ids:
         final_cost = float('inf')
         try:
             final_cost = alns_instance._calculate_total_cost(solution).get('total', float('inf'))
         except Exception as final_cost_e:
             print(f"错误 (Regret): 计算最终成本失败: {final_cost_e}")
             return None 
         # 检查成本是否有效
         if final_cost != float('inf'):
             return solution # 修复成功
         else:
             if getattr(alns_instance, 'verbose', False):
                 print("    Regret Insert: 失败，最终成本为 Inf")
             return None 
    else:
         if getattr(alns_instance, 'verbose', False):
             print(f"    Regret Insert: 失败，有 {len(unassigned_agv_ids)} 个 AGV 未插入: {unassigned_agv_ids}")
         return None 

# --- wait_adjustment_repair  ---
def wait_adjustment_repair(
    alns_instance: 'ALNS',
    partial_solution: 'SolutionType',
    removed_ids: List[int]
) -> Optional['SolutionType']:
    """
    等待调整修复算子 (对应论文 3.5.5 节)。
    先用贪婪法插入移除的 AGV (该过程应用区域分割并使用 v7 动态障碍)，
    然后对随机选择的部分 AGV 路径尝试插入/删除短暂等待。
    """
    if not isinstance(partial_solution, dict): return None

    temp_solution = greedy_insertion(alns_instance, partial_solution, removed_ids) # v7
    if temp_solution is None:
        if getattr(alns_instance, 'verbose', False): print("    WaitAdjust: 基础修复失败。")
        return None

    current_solution = copy.deepcopy(temp_solution)
    num_agvs = getattr(alns_instance, 'num_agvs', len(current_solution))
    num_agvs_to_adjust = max(1, int(num_agvs * 0.2))
    agv_ids_to_consider = list(current_solution.keys()); random.shuffle(agv_ids_to_consider)
    agv_ids_to_adjust = agv_ids_to_consider[:num_agvs_to_adjust]
    made_change_overall = False; max_adjustment_attempts_per_agv = 5; wait_delta_max = 2

    for agv_id in agv_ids_to_adjust:
        path_obj = current_solution.get(agv_id)
        if not path_obj or not isinstance(path_obj, PathType) or not path_obj.sequence or len(path_obj.sequence) < 2: continue
        original_total_cost = float('inf')
        try: original_total_cost = alns_instance._calculate_total_cost(current_solution).get('total', float('inf'))
        except Exception as cost_e: print(f"警告 (WaitAdjust): 计算 AGV {agv_id} 初始成本失败: {cost_e}"); continue
        best_known_sequence_for_agv = path_obj.sequence; best_known_cost_for_agv = original_total_cost
        attempts = 0; path_indices = list(range(len(best_known_sequence_for_agv) - 1)); random.shuffle(path_indices)
        agv_made_change_this_loop = False

        for idx in path_indices:
            if attempts >= max_adjustment_attempts_per_agv: break
            attempts += 1; improvement_found_at_idx = False

            for wait_duration in range(1, wait_delta_max + 1):
                trial_sequence, new_cost = _attempt_insert_wait(alns_instance, current_solution, agv_id, best_known_sequence_for_agv, idx, wait_duration) # v7
                if trial_sequence is not None and new_cost < best_known_cost_for_agv - 1e-6:
                    best_known_sequence_for_agv = trial_sequence; best_known_cost_for_agv = new_cost
                    agv_made_change_this_loop = True; improvement_found_at_idx = True
                    current_solution[agv_id].sequence = best_known_sequence_for_agv
                    break
            if improvement_found_at_idx: continue

            if idx > 0 and best_known_sequence_for_agv[idx][0] == best_known_sequence_for_agv[idx-1][0]:
                prev_node, prev_t = best_known_sequence_for_agv[idx-1]; current_node, current_t = best_known_sequence_for_agv[idx]
                wait_duration_at_prev = current_t - prev_t
                if wait_duration_at_prev > 0 and wait_duration_at_prev <= wait_delta_max:
                    trial_sequence, new_cost = _attempt_delete_wait(alns_instance, current_solution, agv_id, best_known_sequence_for_agv, idx, wait_duration_at_prev) # v7
                    if trial_sequence is not None and new_cost < best_known_cost_for_agv - 1e-6:
                        best_known_sequence_for_agv = trial_sequence; best_known_cost_for_agv = new_cost
                        agv_made_change_this_loop = True
                        current_solution[agv_id].sequence = best_known_sequence_for_agv
        if agv_made_change_this_loop: made_change_overall = True

    final_check_cost = float('inf')
    try:
        final_check_cost = alns_instance._calculate_total_cost(current_solution).get('total', float('inf'))
    except Exception as final_cost_e_check:
        print(f"错误 (WaitAdjust): 计算最终检查成本失败: {final_cost_e_check}")
        return None 

    # 检查最终成本是否有效
    if final_check_cost != float('inf'):
        return current_solution 
    else:
        if getattr(alns_instance, 'verbose', False):
            print("    WaitAdjust: 调整后最终成本为 Inf，算子失败。")
        return None 
# Operators.py-v9 (集成 Shaw Removal, 优化 Regret, 新增 String Removal)
"""
包含自适应大邻域搜索 (ALNS) 算法使用的所有破坏和修复算子函数，
以及它们直接依赖的辅助函数。

与论文的关联:
- 破坏算子: 实现论文第 3.4 节描述的算子策略，并新增/改进:
    - Random Removal (3.4.1)
    - Worst Removal (3.4.2)
    - Congestion Removal (3.4.3)
    - Related Removal (3.4.4 - Shaw Removal 是其更系统化的变体)
    - Conflict History Removal (3.4.5)
    - **新增: Shaw Removal (基于相似性的关联移除)**
    - **新增: String Removal (基于移除重复路径段)**
- 修复算子: 实现论文第 3.5 节描述的算子策略，并改进:
    - Greedy Insertion (3.5.3)
    - **改进: Regret Insertion (3.5.4 - 优化次优查找尝试次数)**
    - Wait Adjustment Repair (3.5.5)
- 辅助函数: 支持算子实现。

版本变更 (v8 -> v9):
- **新增**: 实现 `shaw_removal` 破坏算子。
- **新增**: 实现 `string_removal` 破坏算子。
- **修改**: 优化 `_find_insertion_options`，增加寻找次优解的尝试次数。
- **修改**: 调整部分函数的类型提示和文档。
- **保持**: 其他算子和辅助函数的核心逻辑。
- **依赖**: ALNS, DataTypes (v11+), Map (v9+), Planner (v24+)。
"""
import random
import math
import copy
from collections import defaultdict, Counter
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict, Set, Callable, NamedTuple, Sequence

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
    from DataTypes import CostDict as CostDictType # 明确导入 CostDict
except ImportError as e:
     print(f"错误: 导入 Operators 依赖项失败 (DataTypes): {e}")
     PathType = type('Path', (object,), {})
     SolutionType = Dict; TaskType = type('Task', (object,), {})
     DynObsType = Dict; StateType = Tuple; NodeType = Tuple; CostDictType = Dict

# --- InsertionCost 类 (保持不变) ---
class InsertionCost:
    """存储单个 AGV 的多个插入选项及其成本，用于计算后悔值。"""
    def __init__(self, agv_id: int):
        self.agv_id = agv_id; self.options: List[Tuple[Optional['PathType'], float]] = []
        self.best_cost = float('inf'); self.second_best_cost = float('inf'); self.regret = 0.0
    def add_option(self, path: Optional['PathType'], cost: float):
        if path is not None and not isinstance(path, PathType): raise TypeError(f"路径必须是 Path 类型或 None，得到 {type(path)}")
        self.options.append((path, cost))
        if cost < self.best_cost: self.second_best_cost = self.best_cost; self.best_cost = cost
        elif cost < self.second_best_cost: self.second_best_cost = cost
    def calculate_regret(self, k: int):
        if not self.options: self.regret = 0.0; return
        self.options.sort(key=lambda x: x[1])
        self.best_cost = self.options[0][1] if self.options else float('inf')
        self.second_best_cost = self.options[1][1] if len(self.options) > 1 else float('inf')
        # 计算后悔值：次优与最优之差
        # 如果只有一项或最优为inf，则后悔值为0
        if self.best_cost != float('inf') and self.second_best_cost != float('inf'):
            self.regret = self.second_best_cost - self.best_cost
        else: self.regret = 0.0
        self.regret = max(0.0, self.regret) # 确保后悔值非负
    def get_best_path(self) -> Optional['PathType']:
        if not self.options: return None
        # 确保按成本排序后取第一个
        self.options.sort(key=lambda x: x[1])
        best_option = self.options[0]
        return best_option[0] if best_option[1] != float('inf') else None
    def __lt__(self, other: 'InsertionCost'): return self.regret > other.regret # 按后悔值降序排序

# ==================================
# --- 辅助函数 ---
# ==================================

# --- _build_dynamic_obstacles (保持 v8 逻辑，即 v7 修复) ---
def _build_dynamic_obstacles(alns_instance: 'ALNS', solution: 'SolutionType', exclude_agv_id: Optional[int] = None) -> 'DynObsType':
    dynamic_obstacles: DynObsType = defaultdict(set) # 使用 defaultdict 简化
    if not isinstance(solution, dict): return dynamic_obstacles
    tasks_list = getattr(alns_instance, 'tasks', [])
    if not tasks_list: print("警告 (_build_dynamic_obstacles): 无法从 ALNS 实例获取任务列表。"); return dynamic_obstacles
    tasks_map = {task.agv_id: task for task in tasks_list}
    max_time = getattr(alns_instance, 'max_time', 0)
    if max_time <= 0: print("警告 (_build_dynamic_obstacles): 无法获取有效 max_time。"); max_time = 0

    for agv_id, path in solution.items():
        if agv_id == exclude_agv_id: continue
        if not isinstance(path, PathType) or not path.sequence: continue
        for node, t in path.sequence: dynamic_obstacles[t].add(node)
        task = tasks_map.get(agv_id)
        if not task: continue
        last_node, t_end = path.sequence[-1]; goal_node = task.goal_node
        if last_node == goal_node:
            for t_future in range(t_end + 1, max_time + 1): dynamic_obstacles[t_future].add(goal_node)
    return dict(dynamic_obstacles) # 转回普通字典

# --- _calculate_relatedness (保持 v8 逻辑，即 v6) ---
def _calculate_relatedness(alns_instance: 'ALNS', path1: Optional['PathType'], path2: Optional['PathType']) -> float:
    if not path1 or not path2 or not isinstance(path1, PathType) or not isinstance(path2, PathType) or not path1.sequence or not path2.sequence: return 0.0
    grid_map = getattr(alns_instance, 'grid_map', None); max_time = getattr(alns_instance, 'max_time', 1)
    if not grid_map: return 0.0
    map_width = getattr(grid_map, 'width', 1); map_height = getattr(grid_map, 'height', 1)
    max_dist_factor = max(map_width, map_height) * math.sqrt(2); max_time_factor = max(1, max_time)
    if max_dist_factor < 1e-6: max_dist_factor = 1.0
    if max_time_factor < 1e-6: max_time_factor = 1.0

    relatedness_sum = 0.0; comparisons = len(path1.sequence)
    path2_nodes_times = {(node,t) for node, t in path2.sequence} # 用集合加速查找

    for node1, t1 in path1.sequence:
        min_spatial_dist_sq = float('inf'); min_temporal_dist = float('inf')
        # 优化：可以只比较时间相近的点，但这里保持原逻辑
        for node2, t2 in path2_nodes_times:
            time_diff = abs(t1 - t2); dx = node1[0] - node2[0]; dy = node1[1] - node2[1]; dist_sq = dx**2 + dy**2
            min_spatial_dist_sq = min(min_spatial_dist_sq, dist_sq); min_temporal_dist = min(min_temporal_dist, time_diff)

        spatial_relatedness = 0.0
        if min_spatial_dist_sq != float('inf'):
            spatial_divisor = max_dist_factor * 0.2 + 1e-6 # 空间影响因子
            spatial_relatedness = max(0.0, 1.0 - math.sqrt(min_spatial_dist_sq) / spatial_divisor)
        temporal_relatedness = 0.0
        if min_temporal_dist != float('inf'):
            temporal_divisor = max_time_factor * 0.1 + 1e-6 # 时间影响因子
            temporal_relatedness = max(0.0, 1.0 - min_temporal_dist / temporal_divisor)
        # 加权组合，空间相关性权重略高
        point_relatedness = (spatial_relatedness * 0.6 + temporal_relatedness * 0.4)
        relatedness_sum += point_relatedness

    return relatedness_sum / comparisons if comparisons > 0 else 0.0

# --- _calculate_bounding_box (保持 v8 逻辑，即 v6) ---
def _calculate_bounding_box(start_node: 'NodeType', goal_node: 'NodeType', map_width: int, map_height: int, buffer: int) -> Tuple[int, int, int, int]:
    if not (isinstance(start_node, tuple) and len(start_node) == 2): raise ValueError("start_node 格式错误")
    if not (isinstance(goal_node, tuple) and len(goal_node) == 2): raise ValueError("goal_node 格式错误")
    if buffer < 0: raise ValueError("buffer 不能为负数")
    x_coords = [start_node[0], goal_node[0]]; y_coords = [start_node[1], goal_node[1]]
    min_x_nobuf = min(x_coords); max_x_nobuf = max(x_coords); min_y_nobuf = min(y_coords); max_y_nobuf = max(y_coords)
    min_x = max(0, min_x_nobuf - buffer); max_x = min(map_width - 1, max_x_nobuf + buffer)
    min_y = max(0, min_y_nobuf - buffer); max_y = min(map_height - 1, max_y_nobuf + buffer)
    if min_x > max_x: min_x = max_x = start_node[0] # 处理特殊情况
    if min_y > max_y: min_y = max_y = start_node[1]
    return min_x, max_x, min_y, max_y

# --- _find_insertion_options (v9 - 优化 Regret 尝试次数) ---
def _find_insertion_options(
    alns_instance: 'ALNS',
    task: 'TaskType',
    current_solution: 'SolutionType',
    k: int,
    max_regret_attempts: int = 10 # **增加默认尝试次数**
) -> 'InsertionCost':
    """
    为单个 AGV 查找 k 个最佳插入选项及其成本 (用于 Regret Insertion)。
    v9: 增加了寻找次优解的尝试次数。
    """
    insertion_info = InsertionCost(task.agv_id)
    dynamic_obstacles_base = _build_dynamic_obstacles(alns_instance, current_solution)

    bbox: Optional[Tuple[int, int, int, int]] = None
    try: # 计算包围盒
        grid_map = getattr(alns_instance, 'grid_map', None); buffer_val = getattr(alns_instance, 'buffer', 0)
        if grid_map and hasattr(grid_map, 'width') and hasattr(grid_map, 'height'):
            bbox = _calculate_bounding_box(task.start_node, task.goal_node, grid_map.width, grid_map.height, buffer_val)
    except Exception as bbox_e: print(f"警告 (Regret): 计算 AGV {task.agv_id} 包围盒时异常: {bbox_e}。")

    # 1. 查找最优路径
    best_path: Optional[PathType] = None; best_cost = float('inf')
    try:
        if hasattr(alns_instance, '_call_planner'):
             best_path = alns_instance._call_planner(task, dynamic_obstacles_base, start_time=0, bounding_box=bbox)
        else: print("错误 (Regret): ALNS 实例缺少 _call_planner 方法。"); return insertion_info
    except Exception as call_e: print(f"错误: 调用 ALNS._call_planner 时发生异常: {call_e}"); return insertion_info

    if best_path and isinstance(best_path, PathType) and best_path.sequence:
        try: # 计算最优成本
            grid_map_eval = getattr(alns_instance, 'grid_map', None)
            cost_weights_eval = getattr(alns_instance, 'cost_weights', (1.0,0,0))
            v_eval = getattr(alns_instance, 'v', 1.0); delta_step_eval = getattr(alns_instance, 'delta_step', 1.0)
            if grid_map_eval:
                 cost_dict = best_path.get_cost(grid_map_eval, *cost_weights_eval, v_eval, delta_step_eval)
                 cost = cost_dict.get('total', float('inf'))
            else: cost = float('inf')
        except Exception as cost_e: print(f"警告: 计算最优路径成本失败 (AGV {task.agv_id}): {cost_e}"); cost = float('inf')
        if cost != float('inf'): insertion_info.add_option(best_path, cost); best_cost = cost
        else: best_path = None # 如果成本无效，路径也视为无效

    # 2. 查找次优路径 (如果需要 k > 1 且找到了最优路径)
    if best_path and k > 1:
        secondary_limit = getattr(alns_instance, 'regret_planner_time_limit_abs', 0.1) # 次优规划时间限制
        tried_alternatives = 0; attempt_count = 0; max_attempts_actual = max(1, max_regret_attempts)
        path_signature = tuple(best_path.sequence) # 用于比较路径是否不同
        # 寻找可以阻塞的点 (非起点终点，且是实际移动的点)
        eligible_indices = [i for i in range(1, len(best_path.sequence) - 1) if best_path.sequence[i][0] != best_path.sequence[i-1][0]]
        random.shuffle(eligible_indices); blocked_states_cache: Set[StateType] = set() # 避免重复阻塞同一点

        # 尝试阻塞不同点来寻找次优解
        for block_idx in eligible_indices:
            if tried_alternatives >= k - 1 or len(insertion_info.options) >= k: break # 找到了足够选项
            if attempt_count >= max_attempts_actual: # 达到了最大尝试次数
                if getattr(alns_instance, 'verbose', False) and len(insertion_info.options) < k:
                     print(f"    Regret Info (AGV {task.agv_id}): 达到最大尝试次数 {max_attempts_actual}，只找到 {len(insertion_info.options)}/{k} 个选项。")
                break
            attempt_count += 1

            state_to_block = best_path.sequence[block_idx]
            if state_to_block in blocked_states_cache: continue # 跳过已尝试阻塞的点
            blocked_states_cache.add(state_to_block)

            # 创建临时的动态障碍
            temp_dynamic_obstacles = copy.deepcopy(dynamic_obstacles_base)
            block_t = state_to_block[1]
            if block_t not in temp_dynamic_obstacles: temp_dynamic_obstacles[block_t] = set()
            temp_dynamic_obstacles[block_t].add(state_to_block[0]) # 在该时间点阻塞该节点

            # 尝试规划次优路径
            alternative_path: Optional[PathType] = None
            try:
                planner_inst = getattr(alns_instance, 'planner', None); grid_map_inst = getattr(alns_instance, 'grid_map', None)
                if planner_inst and grid_map_inst:
                     alternative_path = planner_inst.plan(grid_map_inst, task, temp_dynamic_obstacles, getattr(alns_instance, 'max_time', 400), getattr(alns_instance, 'cost_weights', (1.0, 0.0, 0.0)), getattr(alns_instance, 'v', 1.0), getattr(alns_instance, 'delta_step', 1.0), start_time=0, time_limit=secondary_limit, bounding_box=bbox)
                else: print("错误 (Regret Alt): 无法访问 planner 或 grid_map。")
            except Exception as plan_e: print(f"错误: 调用 Planner.plan 失败: {plan_e}")

            if alternative_path and isinstance(alternative_path, PathType) and alternative_path.sequence:
                alt_sig = tuple(alternative_path.sequence)
                if alt_sig != path_signature: # 确保路径与最优路径不同
                    try: # 计算次优成本
                        grid_map_eval = getattr(alns_instance, 'grid_map', None)
                        cost_weights_eval = getattr(alns_instance, 'cost_weights', (1.0,0,0))
                        v_eval = getattr(alns_instance, 'v', 1.0); delta_step_eval = getattr(alns_instance, 'delta_step', 1.0)
                        if grid_map_eval:
                             alt_cost_dict = alternative_path.get_cost(grid_map_eval, *cost_weights_eval, v_eval, delta_step_eval)
                             alt_cost = alt_cost_dict.get('total', float('inf'))
                        else: alt_cost = float('inf')
                    except Exception as alt_cost_e: print(f"警告: 计算次优路径成本失败 (AGV {task.agv_id}): {alt_cost_e}"); alt_cost = float('inf')
                    # 确保存储的是有效的、比最优差的次优解
                    if alt_cost != float('inf') and alt_cost > best_cost + 1e-6:
                        insertion_info.add_option(alternative_path, alt_cost); tried_alternatives += 1

    # 如果在多次尝试后仍未找到足够的次优选项
    if len(insertion_info.options) < k and best_path:
        if getattr(alns_instance, 'verbose', False):
             print(f"    Regret Info (AGV {task.agv_id}): 尝试 {attempt_count} 次后，只找到 {len(insertion_info.options)}/{k} 个有效选项。")

    return insertion_info

# --- _check_sequence_conflicts_segment (保持 v8 逻辑，即 v7) ---
def _check_sequence_conflicts_segment(alns_instance: 'ALNS', current_solution: 'SolutionType', agv_id_to_check: int, segment_to_check: List['StateType']) -> bool:
    if not segment_to_check: return False
    dynamic_obs_check = _build_dynamic_obstacles(alns_instance, current_solution, exclude_agv_id=agv_id_to_check)
    for node, t in segment_to_check:
        if not isinstance(node, tuple): continue
        obstacles_at_t = dynamic_obs_check.get(t)
        if obstacles_at_t and node in obstacles_at_t: return True
    return False

# --- _attempt_insert_wait (保持 v8 逻辑，即 v7) ---
def _attempt_insert_wait(alns_instance: 'ALNS', current_solution: 'SolutionType', agv_id: int, original_sequence: List['StateType'], idx: int, wait_duration: int) -> Tuple[Optional[List['StateType']], float]:
    if idx < 0 or idx >= len(original_sequence) - 1 or wait_duration <= 0: return None, float('inf')
    node, current_t = original_sequence[idx]; new_sequence_part = []; shifted_sequence_part = []
    max_time_limit = getattr(alns_instance, 'max_time', float('inf'))
    wait_end_time = current_t
    for w in range(wait_duration): # 创建等待状态
        wait_time = current_t + 1 + w
        if wait_time > max_time_limit: return None, float('inf') # 超时
        new_sequence_part.append((node, wait_time)); wait_end_time = wait_time
    time_shift = wait_duration
    for n, t in original_sequence[idx+1:]: # 创建后续平移状态
        new_time = t + time_shift
        if new_time > max_time_limit: return None, float('inf') # 超时
        shifted_sequence_part.append((n, new_time))
    trial_sequence = original_sequence[:idx+1] + new_sequence_part + shifted_sequence_part
    # 检查新插入和移动的部分是否有冲突
    if _check_sequence_conflicts_segment(alns_instance, current_solution, agv_id, new_sequence_part): return None, float('inf')
    if _check_sequence_conflicts_segment(alns_instance, current_solution, agv_id, shifted_sequence_part): return None, float('inf')
    # 计算新成本
    new_total_cost = float('inf')
    try:
        temp_solution = copy.deepcopy(current_solution); temp_solution[agv_id] = PathType(agv_id=agv_id, sequence=trial_sequence)
        new_total_cost = alns_instance._calculate_total_cost(temp_solution).get('total', float('inf'))
    except Exception as cost_e: print(f"错误 (InsertWait): 计算成本时失败: {cost_e}")
    return (trial_sequence, new_total_cost) if new_total_cost != float('inf') else (None, float('inf'))

# --- _attempt_delete_wait (保持 v8 逻辑，即 v7) ---
def _attempt_delete_wait(alns_instance: 'ALNS', current_solution: 'SolutionType', agv_id: int, original_sequence: List['StateType'], idx: int, wait_duration_deleted: int) -> Tuple[Optional[List['StateType']], float]:
    if idx < wait_duration_deleted or idx >= len(original_sequence) or wait_duration_deleted <= 0: return None, float('inf')
    # 确保删除的是真正的等待段
    node_before_wait, time_before_wait_start = original_sequence[idx - wait_duration_deleted]
    for i in range(wait_duration_deleted):
        node_wait, time_wait = original_sequence[idx - wait_duration_deleted + 1 + i]
        if node_wait != node_before_wait or time_wait != time_before_wait_start + 1 + i:
            # print(f"调试: 尝试删除的段并非连续等待 @ index {idx}")
            return None, float('inf') # 不是连续等待

    shifted_sequence_part = []; time_shift = -wait_duration_deleted
    max_time_limit = getattr(alns_instance, 'max_time', float('inf'))
    for n, t in original_sequence[idx+1:]: # 创建后续平移状态
         new_time = t + time_shift
         if new_time < 0: return None, float('inf') # 时间不能为负
         shifted_sequence_part.append((n, new_time))
    trial_sequence = original_sequence[:idx - wait_duration_deleted + 1] + shifted_sequence_part
    if trial_sequence and trial_sequence[-1][1] > max_time_limit: return None, float('inf') # 检查最终时间
    # 检查移动的部分是否有冲突
    if _check_sequence_conflicts_segment(alns_instance, current_solution, agv_id, shifted_sequence_part): return None, float('inf')
    # 计算新成本
    new_total_cost = float('inf')
    try:
        temp_solution = copy.deepcopy(current_solution); temp_solution[agv_id] = PathType(agv_id=agv_id, sequence=trial_sequence)
        new_total_cost = alns_instance._calculate_total_cost(temp_solution).get('total', float('inf'))
    except Exception as cost_e: print(f"错误 (DeleteWait): 计算成本时失败: {cost_e}")
    return (trial_sequence, new_total_cost) if new_total_cost != float('inf') else (None, float('inf'))

# ==================================
# --- 破坏算子 (Destroy Operators) ---
# (v9: 新增 Shaw Removal, String Removal)
# ==================================

# --- random_removal (保持 v8) ---
def random_removal(alns_instance: 'ALNS', solution: 'SolutionType', removal_count: int) -> Tuple['SolutionType', List[int]]:
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

# --- worst_removal (保持 v8) ---
def worst_removal(alns_instance: 'ALNS', solution: 'SolutionType', removal_count: int) -> Tuple['SolutionType', List[int]]:
    if not solution: return {}, []
    costs: List[Tuple[float, int]] = []
    valid_agvs = []
    grid_map_eval = getattr(alns_instance, 'grid_map', None)
    if not grid_map_eval: print("错误 (WorstRemoval): 无法访问地图以计算成本。"); return copy.deepcopy(solution), []
    cost_weights_eval = getattr(alns_instance, 'cost_weights', (1.0,0,0))
    v_eval = getattr(alns_instance, 'v', 1.0); delta_step_eval = getattr(alns_instance, 'delta_step', 1.0)

    for agv_id, path in solution.items():
        if path and isinstance(path, PathType) and path.sequence:
            valid_agvs.append(agv_id); cost = float('inf')
            try: cost = path.get_cost(grid_map_eval, *cost_weights_eval, v_eval, delta_step_eval).get('total', float('inf'))
            except Exception as e: print(f"警告 (WorstRemoval): 计算 AGV {agv_id} 成本失败: {e}")
            if cost != float('inf'): costs.append((cost, agv_id)) # 存储 (成本, AGV ID)

    if not valid_agvs: return copy.deepcopy(solution), []
    costs.sort(key=lambda item: item[0], reverse=True) # 按成本降序排序
    actual_removal_count = min(removal_count, len(costs))
    if actual_removal_count <= 0: return copy.deepcopy(solution), []
    removed_ids = [agv_id for cost, agv_id in costs[:actual_removal_count]] # 取成本最高的
    partial_solution = copy.deepcopy(solution)
    for agv_id in removed_ids:
        if agv_id in partial_solution: del partial_solution[agv_id]
    return partial_solution, removed_ids

# --- related_removal (保持 v8) ---
def related_removal(alns_instance: 'ALNS', solution: 'SolutionType', removal_count: int) -> Tuple['SolutionType', List[int]]:
    if not solution: return {}, []
    partial_solution = copy.deepcopy(solution)
    agvs_in_solution = [aid for aid, p in partial_solution.items() if p is not None and isinstance(p, PathType) and p.sequence]
    if not agvs_in_solution or removal_count <= 0: return partial_solution, []
    actual_removal_count = min(removal_count, len(agvs_in_solution))
    if actual_removal_count <= 0: return partial_solution, []

    seed_agv_id = random.choice(agvs_in_solution) # 随机选种子
    removed_ids = [seed_agv_id]
    if seed_agv_id in partial_solution: del partial_solution[seed_agv_id]
    deterministic_factor = 4 # 随机性参数

    while len(removed_ids) < actual_removal_count:
        if not partial_solution: break # 没有更多 AGV 可移除
        candidates = list(partial_solution.keys())
        if not candidates: break

        relatedness_scores: List[Tuple[float, int]] = []
        last_removed_id = removed_ids[-1] # 基于最新移除的 AGV 计算相关性
        path_last = solution.get(last_removed_id) # 从原始解获取完整路径

        for candidate_id in candidates:
            path_candidate = partial_solution.get(candidate_id) # 获取候选者的当前路径（如果存在）
            if path_candidate: # 确保候选者路径存在
                 relatedness = _calculate_relatedness(alns_instance, path_last, path_candidate)
                 relatedness_scores.append((relatedness, candidate_id))

        if not relatedness_scores: break # 没有可计算相关性的候选者
        relatedness_scores.sort(key=lambda item: item[0], reverse=True) # 按相关性降序
        # --- 选择移除对象 (带随机性) ---
        rand_val = random.random(); max_index = len(relatedness_scores) - 1
        # 指数加权随机选择，倾向于选择更相关的，但也有机会选不太相关的
        index_to_pick = min(int(len(relatedness_scores) * (rand_val ** deterministic_factor)), max_index)
        index_to_pick = max(0, index_to_pick) # 确保索引有效
        # --- ----------------------- ---
        agv_to_remove = relatedness_scores[index_to_pick][1]
        removed_ids.append(agv_to_remove)
        if agv_to_remove in partial_solution: del partial_solution[agv_to_remove]

    return partial_solution, removed_ids

# --- congestion_removal (保持 v8) ---
def congestion_removal(alns_instance: 'ALNS', solution: 'SolutionType', removal_count: int) -> Tuple['SolutionType', List[int]]:
    if not solution: return {}, []
    node_time_counts: Dict[StateType, int] = defaultdict(int); max_t = 0
    # 1. 统计时空节点占用次数
    for path in solution.values():
        if path and isinstance(path, PathType) and path.sequence:
            max_t = max(max_t, path.get_makespan())
            for state in path.sequence: node_time_counts[state] += 1
    # 2. 计算每个路径的拥挤度得分
    path_congestion_scores: List[Tuple[float, int]] = []
    valid_agvs = []
    for agv_id, path in solution.items():
        if path and isinstance(path, PathType) and path.sequence:
            valid_agvs.append(agv_id); congestion_score = 0.0
            for state in path.sequence:
                 occupancy = node_time_counts.get(state, 0)
                 if occupancy > 1: congestion_score += float(occupancy - 1) # 累加冲突次数
            # 按路径长度归一化拥挤度
            path_len = len(path.sequence); normalized_score = congestion_score / path_len if path_len > 0 else 0
            path_congestion_scores.append((normalized_score, agv_id))
    # 3. 移除拥挤度最高的
    if not valid_agvs: return copy.deepcopy(solution), []
    path_congestion_scores.sort(key=lambda item: item[0], reverse=True) # 按拥挤度降序
    actual_removal_count = min(removal_count, len(path_congestion_scores))
    if actual_removal_count <= 0: return copy.deepcopy(solution), []
    removed_ids = [agv_id for score, agv_id in path_congestion_scores[:actual_removal_count]]
    partial_solution = copy.deepcopy(solution)
    for agv_id in removed_ids:
        if agv_id in partial_solution: del partial_solution[agv_id]
    return partial_solution, removed_ids

# --- conflict_history_removal (保持 v8) ---
def conflict_history_removal(alns_instance: 'ALNS', solution: 'SolutionType', removal_count: int) -> Tuple['SolutionType', List[int]]:
    if not solution: return {}, []
    agvs_in_solution = [aid for aid, p in solution.items() if p is not None and isinstance(p, PathType) and p.sequence]
    if not agvs_in_solution: return {}, []
    # 从 ALNS 实例获取冲突计数器
    conflict_counter = getattr(alns_instance, 'agv_conflict_counts', Counter())
    # 只考虑当前在解中的 AGV 的冲突次数
    conflict_counts = Counter({agv_id: conflict_counter.get(agv_id, 0) for agv_id in agvs_in_solution})
    sorted_by_conflict = conflict_counts.most_common() # 按冲突次数降序排序
    actual_removal_count = min(removal_count, len(sorted_by_conflict))
    if actual_removal_count <= 0: return copy.deepcopy(solution), []
    removed_ids = [agv_id for agv_id, count in sorted_by_conflict[:actual_removal_count]] # 取冲突最多的
    partial_solution = copy.deepcopy(solution)
    for agv_id in removed_ids:
        if agv_id in partial_solution: del partial_solution[agv_id]
    # 可选的调试打印
    verbose = getattr(alns_instance, 'verbose', False); debug_weights = getattr(alns_instance, 'debug_weights', False)
    if verbose and debug_weights:
        removed_counts = [conflict_counts[aid] for aid in removed_ids]
        print(f"    ConflictHistoryRemoval: Removed {len(removed_ids)} AGVs: {removed_ids} (Counts: {removed_counts})")
    return partial_solution, removed_ids

# --- 新增: shaw_removal (v9) ---
def shaw_removal(
    alns_instance: 'ALNS',
    solution: 'SolutionType',
    removal_count: int
) -> Tuple['SolutionType', List[int]]:
    """
    Shaw 移除算子 (基于相似性)。
    随机选择一个种子 AGV，然后移除与其最相似的其他 AGV。
    相似性基于起终点距离和时间窗口重叠。
    """
    if not solution: return {}, []
    partial_solution = copy.deepcopy(solution)
    agvs_in_solution = {aid: p for aid, p in partial_solution.items() if p is not None and isinstance(p, PathType) and p.sequence}
    agv_ids_list = list(agvs_in_solution.keys())
    if not agv_ids_list or removal_count <= 0: return partial_solution, []
    actual_removal_count = min(removal_count, len(agv_ids_list))
    if actual_removal_count <= 0: return partial_solution, []

    # 1. 获取任务信息和地图参数
    tasks_list = getattr(alns_instance, 'tasks', [])
    tasks_map = {task.agv_id: task for task in tasks_list}
    grid_map = getattr(alns_instance, 'grid_map', None)
    max_time = getattr(alns_instance, 'max_time', 1)
    if not grid_map or not tasks_map: print("错误 (ShawRemoval): 无法获取地图或任务信息。"); return partial_solution, []
    map_width = getattr(grid_map, 'width', 1); map_height = getattr(grid_map, 'height', 1)
    max_dist = max(map_width, map_height) * math.sqrt(2) if map_width > 0 and map_height > 0 else 1.0

    # 2. 随机选择种子 AGV
    seed_agv_id = random.choice(agv_ids_list)
    removed_ids = [seed_agv_id]
    if seed_agv_id in partial_solution: del partial_solution[seed_agv_id]
    deterministic_factor = 4 # 随机性参数 (与 related_removal 类似)

    # 3. 计算相似性并移除
    while len(removed_ids) < actual_removal_count:
        if not partial_solution: break # 没有更多 AGV 可移除
        candidates = list(partial_solution.keys())
        if not candidates: break

        similarity_scores: List[Tuple[float, int]] = []
        task_seed = tasks_map.get(seed_agv_id)
        path_seed = solution.get(seed_agv_id) # 注意：从原始 solution 获取路径用于比较
        if not task_seed or not path_seed: break # 种子信息缺失

        for candidate_id in candidates:
            task_candidate = tasks_map.get(candidate_id)
            path_candidate = partial_solution.get(candidate_id) # 获取候选者的当前路径
            if task_candidate and path_candidate: # 确保信息完整
                # --- 计算相似性 ---
                dist_start = math.dist(task_seed.start_node, task_candidate.start_node) if task_seed and task_candidate else max_dist
                dist_goal = math.dist(task_seed.goal_node, task_candidate.goal_node) if task_seed and task_candidate else max_dist
                dist_similarity = max(0.0, 1.0 - (dist_start + dist_goal) / (2 * max_dist + 1e-9))

                makespan_seed = path_seed.get_makespan() if path_seed else 0
                makespan_cand = path_candidate.get_makespan() if path_candidate else 0
                start_seed = path_seed.sequence[0][1] if path_seed and path_seed.sequence else 0
                start_cand = path_candidate.sequence[0][1] if path_candidate and path_candidate.sequence else 0
                overlap_start = max(start_seed, start_cand)
                overlap_end = min(makespan_seed, makespan_cand)
                overlap_duration = max(0, overlap_end - overlap_start)
                max_duration = max(makespan_seed - start_seed, makespan_cand - start_cand)
                time_similarity = overlap_duration / (max_duration + 1e-9) if max_duration > 0 else 0

                # 加权组合 (权重可调)
                similarity = 0.6 * dist_similarity + 0.4 * time_similarity
                # --------------------
                similarity_scores.append((similarity, candidate_id))

        if not similarity_scores: break
        similarity_scores.sort(key=lambda item: item[0], reverse=True) # 按相似度降序
        # --- 选择移除对象 (带随机性) ---
        rand_val = random.random(); max_index = len(similarity_scores) - 1
        index_to_pick = min(int(len(similarity_scores) * (rand_val ** deterministic_factor)), max_index)
        index_to_pick = max(0, index_to_pick)
        # --- ----------------------- ---
        agv_to_remove = similarity_scores[index_to_pick][1]
        removed_ids.append(agv_to_remove)
        if agv_to_remove in partial_solution: del partial_solution[agv_to_remove]

    return partial_solution, removed_ids

# --- 新增: string_removal (v9) ---
def string_removal(
    alns_instance: 'ALNS',
    solution: 'SolutionType',
    removal_count: int,
    min_string_len: int = 3, # 考虑的最小路径串长度
    max_strings_to_consider: int = 10 # 最多考虑多少种高频串
) -> Tuple['SolutionType', List[int]]:
    """
    路径串移除算子。
    识别路径中频繁出现的连续节点子序列（路径串），
    然后移除包含某个随机选中的高频串的 AGV。
    """
    if not solution: return {}, []
    partial_solution = copy.deepcopy(solution)
    agvs_in_solution = [aid for aid, p in partial_solution.items() if p is not None and isinstance(p, PathType) and p.sequence]
    if not agvs_in_solution or removal_count <= 0: return partial_solution, []

    # 1. 提取所有长度 >= min_string_len 的路径串及其出现次数
    string_counts: Counter = Counter()
    agv_strings: Dict[int, Set[Tuple[NodeType, ...]]] = defaultdict(set) # 记录每个 AGV 包含哪些串

    for agv_id, path in solution.items():
        if path and isinstance(path, PathType) and len(path.sequence) >= min_string_len:
            nodes_only = tuple(state[0] for state in path.sequence) # 只关心节点序列
            for i in range(len(nodes_only) - min_string_len + 1):
                for length in range(min_string_len, len(nodes_only) - i + 1): # 考虑不同长度的串
                    sub_sequence = nodes_only[i : i + length]
                    string_counts[sub_sequence] += 1
                    agv_strings[agv_id].add(sub_sequence)

    if not string_counts: return partial_solution, [] # 没有找到足够长的串

    # 2. 选择高频串
    # 过滤掉只出现一次的串，并按频率排序
    frequent_strings = [(s, count) for s, count in string_counts.items() if count > 1]
    if not frequent_strings: return partial_solution, [] # 没有重复出现的串
    frequent_strings.sort(key=lambda item: item[1], reverse=True)

    # 3. 随机选择一个高频串进行移除
    num_strings_to_choose_from = min(max_strings_to_consider, len(frequent_strings))
    if num_strings_to_choose_from <= 0: return partial_solution, []
    chosen_string, _ = random.choice(frequent_strings[:num_strings_to_choose_from])

    # 4. 移除包含该串的 AGV
    removed_ids = []
    agvs_to_check = list(agvs_in_solution) # 创建副本迭代
    random.shuffle(agvs_to_check) # 增加随机性

    for agv_id in agvs_to_check:
        if len(removed_ids) >= removal_count: break # 达到移除数量上限
        if agv_id in partial_solution and chosen_string in agv_strings.get(agv_id, set()):
            removed_ids.append(agv_id)
            if agv_id in partial_solution: del partial_solution[agv_id]

    if not removed_ids and agvs_in_solution: # 如果没移除任何AGV，随机移除一个
         fallback_id = random.choice(agvs_in_solution)
         removed_ids.append(fallback_id)
         if fallback_id in partial_solution: del partial_solution[fallback_id]

    if getattr(alns_instance, 'verbose', False) and getattr(alns_instance, 'debug_weights', False):
        print(f"    StringRemoval: Based on string {chosen_string}, removed AGVs: {removed_ids}")

    return partial_solution, removed_ids


# ==================================
# --- 修复算子 (Repair Operators) ---
# (v9: Regret 优化)
# ==================================

# --- greedy_insertion (保持 v8) ---
def greedy_insertion(alns_instance: 'ALNS', partial_solution: 'SolutionType', removed_ids: List[int]) -> Optional['SolutionType']:
    if not isinstance(partial_solution, dict): return None
    solution = copy.deepcopy(partial_solution)
    tasks_to_insert = [t for t in getattr(alns_instance, 'tasks', []) if t.agv_id in removed_ids]
    if not tasks_to_insert: return solution
    random.shuffle(tasks_to_insert) # 随机修复顺序

    for task in tasks_to_insert:
        agv_id = task.agv_id
        dynamic_obstacles = _build_dynamic_obstacles(alns_instance, solution)
        bbox: Optional[Tuple[int, int, int, int]] = None
        try: # 计算包围盒
            grid_map = getattr(alns_instance, 'grid_map', None); buffer_val = getattr(alns_instance, 'buffer', 0)
            if grid_map and hasattr(grid_map, 'width') and hasattr(grid_map, 'height'):
                bbox = _calculate_bounding_box(task.start_node, task.goal_node, grid_map.width, grid_map.height, buffer_val)
        except Exception as bbox_e: print(f"警告 (Greedy): 计算 AGV {agv_id} 包围盒失败: {bbox_e}。")

        new_path: Optional[PathType] = None
        try: # 调用规划器
            if hasattr(alns_instance, '_call_planner'):
                new_path = alns_instance._call_planner(task, dynamic_obstacles, start_time=0, bounding_box=bbox)
        except Exception as call_e: print(f"错误: 调用 ALNS._call_planner 失败: {call_e}")

        if new_path and isinstance(new_path, PathType) and new_path.sequence:
            solution[agv_id] = new_path # 插入成功
        else:
            if getattr(alns_instance, 'verbose', False): print(f"    Greedy Insert: 失败，无法为 AGV {agv_id} 找到路径。")
            return None # 修复失败

    # 检查最终解是否完整且有效
    num_expected_agvs = getattr(alns_instance, 'num_agvs', len(getattr(alns_instance, 'tasks', [])))
    if len(solution) == num_expected_agvs:
         final_cost = float('inf')
         try: final_cost = alns_instance._calculate_total_cost(solution).get('total', float('inf'))
         except Exception: return None
         return solution if final_cost != float('inf') else None
    else: return None

# --- regret_insertion (保持 v8，依赖优化的 _find_insertion_options) ---
def regret_insertion(alns_instance: 'ALNS', partial_solution: 'SolutionType', removed_ids: List[int]) -> Optional['SolutionType']:
    if not isinstance(partial_solution, dict): return None
    solution = copy.deepcopy(partial_solution)
    tasks_map = {t.agv_id: t for t in getattr(alns_instance, 'tasks', []) if t.agv_id in removed_ids}
    if not tasks_map and removed_ids: return None
    if not removed_ids: return solution

    unassigned_agv_ids = removed_ids[:]
    iteration = 0; max_regret_iterations = len(unassigned_agv_ids) * 2 # 增加迭代次数上限

    while unassigned_agv_ids and iteration < max_regret_iterations:
        iteration += 1
        insertion_candidates: List[InsertionCost] = []
        regret_k = getattr(alns_instance, 'regret_k', 2)
        max_attempts_regret = getattr(alns_instance, 'regret_max_attempts', 10) # 从配置获取

        # 1. 为所有未分配的 AGV 计算插入选项和后悔值
        for agv_id in unassigned_agv_ids:
            task = tasks_map.get(agv_id)
            if not task: continue
            # 调用优化的 _find_insertion_options (v9)
            candidate_info = _find_insertion_options(alns_instance, task, solution, regret_k, max_attempts_regret)
            candidate_info.calculate_regret(regret_k) # 确保计算后悔值
            insertion_candidates.append(candidate_info)

        if not insertion_candidates: break # 没有候选者了

        # 2. 按后悔值排序，选择最高的插入
        insertion_candidates.sort(key=lambda c: c.regret, reverse=True)
        best_candidate_to_insert = insertion_candidates[0]
        agv_id_to_insert = best_candidate_to_insert.agv_id
        best_path_for_agv = best_candidate_to_insert.get_best_path()

        # 3. 插入最佳路径
        if best_path_for_agv and isinstance(best_path_for_agv, PathType):
            solution[agv_id_to_insert] = best_path_for_agv
            if agv_id_to_insert in unassigned_agv_ids: unassigned_agv_ids.remove(agv_id_to_insert)
            # 可选: 打印插入信息
            # if getattr(alns_instance, 'verbose', False):
            #     print(f"    Regret Insert: 插入 AGV {agv_id_to_insert} (Regret={best_candidate_to_insert.regret:.2f})")
        else:
            # 如果后悔值最高的 AGV 无法找到有效路径，将其移除，避免死循环
            if getattr(alns_instance, 'verbose', False):
                 print(f"    Regret Insert: 警告，后悔值最高的 AGV {agv_id_to_insert} 无法找到有效路径，将其移出待修复列表。")
            if agv_id_to_insert in unassigned_agv_ids: unassigned_agv_ids.remove(agv_id_to_insert)

    # 检查最终解是否完整且有效
    if not unassigned_agv_ids:
         final_cost = float('inf')
         try: final_cost = alns_instance._calculate_total_cost(solution).get('total', float('inf'))
         except Exception: return None
         return solution if final_cost != float('inf') else None
    else:
         if getattr(alns_instance, 'verbose', False):
             print(f"    Regret Insert: 失败，有 {len(unassigned_agv_ids)} 个 AGV 未插入: {unassigned_agv_ids}")
         return None

# --- wait_adjustment_repair (保持 v8) ---
def wait_adjustment_repair(alns_instance: 'ALNS', partial_solution: 'SolutionType', removed_ids: List[int]) -> Optional['SolutionType']:
    if not isinstance(partial_solution, dict): return None
    # 1. 先用贪婪法修复
    temp_solution = greedy_insertion(alns_instance, partial_solution, removed_ids)
    if temp_solution is None:
        if getattr(alns_instance, 'verbose', False): print("    WaitAdjust: 基础修复 (Greedy) 失败。")
        return None
    current_solution = copy.deepcopy(temp_solution)
    num_agvs = getattr(alns_instance, 'num_agvs', len(current_solution))
    if num_agvs == 0: return current_solution # 没有 AGV 无需调整

    # 2. 随机选择部分 AGV 进行等待调整
    num_agvs_to_adjust = max(1, int(num_agvs * 0.2)) # 调整 20% 的 AGV
    agv_ids_to_consider = list(current_solution.keys()); random.shuffle(agv_ids_to_consider)
    agv_ids_to_adjust = agv_ids_to_consider[:num_agvs_to_adjust]
    made_change_overall = False; max_adjustment_attempts_per_agv = 5; wait_delta_max = 3 # 最多插入/删除 3 步等待

    for agv_id in agv_ids_to_adjust:
        path_obj = current_solution.get(agv_id)
        if not path_obj or not isinstance(path_obj, PathType) or not path_obj.sequence or len(path_obj.sequence) < 2: continue

        original_total_cost = float('inf')
        try: original_total_cost = alns_instance._calculate_total_cost(current_solution).get('total', float('inf'))
        except Exception: continue # 无法计算当前成本，跳过此 AGV
        if original_total_cost == float('inf'): continue

        best_known_sequence_for_agv = list(path_obj.sequence) # 使用列表方便修改
        best_known_cost_for_agv = original_total_cost
        attempts = 0; path_indices = list(range(len(best_known_sequence_for_agv) - 1)); random.shuffle(path_indices)
        agv_made_change_this_loop = False

        # 尝试在不同位置插入/删除等待
        for idx in path_indices:
            if attempts >= max_adjustment_attempts_per_agv: break
            attempts += 1; improvement_found_at_idx = False

            # 尝试插入等待
            for wait_duration in range(1, wait_delta_max + 1):
                trial_sequence, new_cost = _attempt_insert_wait(alns_instance, current_solution, agv_id, best_known_sequence_for_agv, idx, wait_duration)
                if trial_sequence is not None and new_cost < best_known_cost_for_agv - 1e-6:
                    best_known_sequence_for_agv = trial_sequence; best_known_cost_for_agv = new_cost
                    agv_made_change_this_loop = True; improvement_found_at_idx = True
                    current_solution[agv_id].sequence = best_known_sequence_for_agv # 更新解
                    break # 找到改进，不再尝试在此处插入更长等待
            if improvement_found_at_idx: continue # 移动到下一个索引

            # 尝试删除等待 (如果当前索引之前是等待)
            if idx > 0:
                 node_before, time_before = best_known_sequence_for_agv[idx-1]
                 node_current, time_current = best_known_sequence_for_agv[idx]
                 if node_current == node_before: # 确认是等待
                     wait_duration_at_prev = time_current - time_before
                     if wait_duration_at_prev > 0 and wait_duration_at_prev <= wait_delta_max:
                         trial_sequence, new_cost = _attempt_delete_wait(alns_instance, current_solution, agv_id, best_known_sequence_for_agv, idx, wait_duration_at_prev)
                         if trial_sequence is not None and new_cost < best_known_cost_for_agv - 1e-6:
                             best_known_sequence_for_agv = trial_sequence; best_known_cost_for_agv = new_cost
                             agv_made_change_this_loop = True
                             current_solution[agv_id].sequence = best_known_sequence_for_agv # 更新解

        if agv_made_change_this_loop: made_change_overall = True

    # 检查最终成本是否有效
    final_check_cost = float('inf')
    try: final_check_cost = alns_instance._calculate_total_cost(current_solution).get('total', float('inf'))
    except Exception: return None
    return current_solution if final_check_cost != float('inf') else None
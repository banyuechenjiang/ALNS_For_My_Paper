# Operators.py-v6 (Refactored for Readability and PEP 8 Compliance)
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

版本变更 (v5 -> v6):
- 新增: 添加辅助函数 `_calculate_bounding_box` 用于计算区域分割的包围盒 (论文 3.5.2)。
- 修改: 修复算子 (`greedy_insertion`, `_find_insertion_options` for `regret_insertion`)
           现在会计算包围盒，并将其传递给 `alns_instance._call_planner`
           (假设 ALNS._call_planner v38+ 会接收此参数)。
- **修正**: 重构代码以符合 PEP 8 标准，消除非标准结构和压缩错误，提高可读性。
- 保持: v5 的所有其他功能、逻辑和详细注释均被保留或适当更新。
- 更新了模块和相关函数的文档字符串，以反映区域分割的实现。
"""
import random
import math
import copy
from collections import defaultdict, Counter
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict, Set, Callable, NamedTuple

# --- 类型提示导入 ---
if TYPE_CHECKING:
    # 依赖 v38+ 的 ALNS
    from ALNS import ALNS
    # 依赖 v9 的 DataTypes
    from DataTypes import Solution, Task, Path, CostDict, TimeStep, Node, DynamicObstacles, State
    # 依赖 v7 的 Map
    from Map import GridMap
    # 依赖 v15+ 的 Planner
    from Planner import TWAStarPlanner

# --- 从 DataTypes 导入必要类 (保持 v5) ---
try:
    from DataTypes import Path as PathType
    from DataTypes import Solution as SolutionType
    from DataTypes import Task as TaskType
    from DataTypes import DynamicObstacles as DynObsType
    from DataTypes import State as StateType
    from DataTypes import Node as NodeType
except ImportError as e:
     print(f"错误: 导入 Operators 依赖项失败 (DataTypes): {e}")
     # Fallback types for static analysis if imports fail
     PathType = type('Path', (object,), {})
     SolutionType = Dict
     TaskType = type('Task', (object,), {})
     DynObsType = Dict
     StateType = Tuple
     NodeType = Tuple

# --- InsertionCost 类 (保持 v5) ---
class InsertionCost:
    """存储单个 AGV 的多个插入选项及其成本，用于计算后悔值。"""
    def __init__(self, agv_id: int):
        self.agv_id = agv_id
        self.options: List[Tuple[Optional['PathType'], float]] = []
        self.best_cost = float('inf')
        self.second_best_cost = float('inf')
        self.regret = 0.0

    def add_option(self, path: Optional['PathType'], cost: float):
        """添加一个插入选项。"""
        if path is not None and not isinstance(path, PathType):
            raise TypeError(f"路径必须是 Path 类型或 None，得到 {type(path)}")
        self.options.append((path, cost))
        # 实时更新最优和次优成本
        if cost < self.best_cost:
            self.second_best_cost = self.best_cost
            self.best_cost = cost
        elif cost < self.second_best_cost:
            self.second_best_cost = cost

    def calculate_regret(self, k: int):
        """计算后悔值 (通常基于最优和次优)。"""
        if not self.options:
            self.regret = 0.0
            return
        # 显式排序以确保 best 和 second_best 正确
        self.options.sort(key=lambda x: x[1])
        self.best_cost = self.options[0][1] if self.options else float('inf')
        self.second_best_cost = self.options[1][1] if len(self.options) > 1 else float('inf')
        # 计算后悔值
        if self.best_cost != float('inf') and self.second_best_cost != float('inf'):
            self.regret = self.second_best_cost - self.best_cost
        else:
            self.regret = 0.0
        # 后悔值非负
        self.regret = max(0.0, self.regret)

    def get_best_path(self) -> Optional['PathType']:
        """获取成本最低的有效路径。"""
        if not self.options:
            return None
        # 假设已排序
        best_option = self.options[0]
        return best_option[0] if best_option[1] != float('inf') else None

    def __lt__(self, other: 'InsertionCost'):
        """定义比较行为，用于按后悔值降序排序。"""
        return self.regret > other.regret


# ==================================
# --- 辅助函数 ---
# ==================================

# --- _build_dynamic_obstacles (保持 v5) ---
def _build_dynamic_obstacles(
    solution: 'SolutionType',
    exclude_agv_id: Optional[int] = None
) -> 'DynObsType':
    """
    根据当前（部分）解构建动态障碍物字典。
    用于传递给 TWA* 规划器，以满足节点冲突约束 (论文公式 12)。
    """
    dynamic_obstacles: DynObsType = {}
    if not isinstance(solution, dict):
        return dynamic_obstacles
    for agv_id, path in solution.items():
        if agv_id == exclude_agv_id:
            continue
        if not isinstance(path, PathType) or not path.sequence:
            continue
        # 遍历路径中的每个时空状态
        for node, t in path.sequence:
            if t not in dynamic_obstacles:
                dynamic_obstacles[t] = set()
            dynamic_obstacles[t].add(node) # 将 (t, node) 标记为占用
    return dynamic_obstacles

# --- _calculate_relatedness (保持 v5) ---
def _calculate_relatedness(
    alns_instance: 'ALNS',
    path1: Optional['PathType'],
    path2: Optional['PathType']
) -> float:
    """
    计算两条路径之间的相关性得分 (用于 Related Removal 算子)。
    相关性基于路径在时间和空间上的接近程度。
    """
    if not path1 or not path2:
        return 0.0
    if not isinstance(path1, PathType) or not isinstance(path2, PathType):
        return 0.0
    if not path1.sequence or not path2.sequence:
        return 0.0

    # 使用健壮的方式访问属性，避免 AttributeError
    grid_map = getattr(alns_instance, 'grid_map', None)
    max_time = getattr(alns_instance, 'max_time', 1) # 默认 max_time 为 1
    if not grid_map:
        print("警告 (Relatedness): 无法访问 ALNS 实例的 grid_map。")
        return 0.0 # 如果无法访问地图，无法计算

    map_width = getattr(grid_map, 'width', 1)
    map_height = getattr(grid_map, 'height', 1)

    # 获取归一化因子
    max_dist_factor = max(map_width, map_height) * math.sqrt(2)
    max_time_factor = max(1, max_time) # 确保 max_time 至少为 1
    # 避免除以零
    if max_dist_factor < 1e-6:
        max_dist_factor = 1.0
    if max_time_factor < 1e-6:
        max_time_factor = 1.0

    relatedness_sum = 0.0
    comparisons = len(path1.sequence)
    # 预计算路径2的时空节点集合以提高查找效率
    path2_nodes_times = {(node,t) for node, t in path2.sequence}

    # 遍历路径1的每个时空点
    for idx1, (node1, t1) in enumerate(path1.sequence):
        min_spatial_dist_sq = float('inf')
        min_temporal_dist = float('inf')
        # 找到路径2中与当前点 (node1, t1) 最接近的点
        for node2, t2 in path2_nodes_times:
            time_diff = abs(t1 - t2)
            dx = node1[0] - node2[0]
            dy = node1[1] - node2[1]
            dist_sq = dx**2 + dy**2
            min_spatial_dist_sq = min(min_spatial_dist_sq, dist_sq)
            min_temporal_dist = min(min_temporal_dist, time_diff)

        # 计算空间和时间相关性 (归一化到 0-1)
        spatial_relatedness = 0.0
        if min_spatial_dist_sq != float('inf'):
            spatial_divisor = max_dist_factor * 0.2 + 1e-6 # 避免除零
            spatial_relatedness = max(0.0, 1.0 - math.sqrt(min_spatial_dist_sq) / spatial_divisor)

        temporal_relatedness = 0.0
        if min_temporal_dist != float('inf'):
            temporal_divisor = max_time_factor * 0.1 + 1e-6 # 避免除零
            temporal_relatedness = max(0.0, 1.0 - min_temporal_dist / temporal_divisor)

        # 加权组合空间和时间相关性
        point_relatedness = (spatial_relatedness * 0.6 + temporal_relatedness * 0.4)
        relatedness_sum += point_relatedness

    # 返回平均相关性
    return relatedness_sum / comparisons if comparisons > 0 else 0.0

# --- 新增: _calculate_bounding_box (v6) ---
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
    if not (isinstance(start_node, tuple) and len(start_node) == 2):
        raise ValueError("start_node 格式错误")
    if not (isinstance(goal_node, tuple) and len(goal_node) == 2):
        raise ValueError("goal_node 格式错误")
    if buffer < 0:
        raise ValueError("buffer 不能为负数")

    # 计算初始边界
    x_coords = [start_node[0], goal_node[0]]
    y_coords = [start_node[1], goal_node[1]]
    min_x_nobuf = min(x_coords)
    max_x_nobuf = max(x_coords)
    min_y_nobuf = min(y_coords)
    max_y_nobuf = max(y_coords)

    # 应用 buffer 并限制在地图范围内
    min_x = max(0, min_x_nobuf - buffer)
    max_x = min(map_width - 1, max_x_nobuf + buffer)
    min_y = max(0, min_y_nobuf - buffer)
    max_y = min(map_height - 1, max_y_nobuf + buffer)

    # 处理特殊情况（例如起点等于终点）
    if min_x > max_x:
        min_x = max_x = start_node[0]
    if min_y > max_y:
        min_y = max_y = start_node[1]

    return min_x, max_x, min_y, max_y

# --- _find_insertion_options (修改 - v6) ---
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
    dynamic_obstacles_base = _build_dynamic_obstacles(current_solution)

    # --- v6: 计算包围盒 ---
    bbox: Optional[Tuple[int, int, int, int]] = None
    try:
        # 健壮地访问属性
        grid_map = getattr(alns_instance, 'grid_map', None)
        buffer_val = getattr(alns_instance, 'buffer', 0)
        if grid_map and hasattr(grid_map, 'width') and hasattr(grid_map, 'height'):
            bbox = _calculate_bounding_box(
                task.start_node,
                task.goal_node,
                grid_map.width,
                grid_map.height,
                buffer_val
            )
        else:
            print(f"警告 (Regret): 无法访问地图或 buffer，不使用区域分割。")
    except ValueError as ve:
        print(f"警告 (Regret): 检查 bbox 输入失败: {ve}。不使用区域分割。")
    except Exception as bbox_e:
        print(f"警告 (Regret): 计算 AGV {task.agv_id} 包围盒时异常: {bbox_e}。不使用区域分割。")

    # 1. 查找最优路径 (传入 bbox)
    best_path: Optional[PathType] = None
    # --- 假设 ALNS (v38+) 有 _call_planner 方法支持 bbox ---
    try:
        if hasattr(alns_instance, '_call_planner'):
             # 尝试调用带 bbox 的版本
             best_path = alns_instance._call_planner(
                 task=task,
                 dynamic_obstacles=dynamic_obstacles_base,
                 start_time=0,
                 bounding_box=bbox # 传递计算出的 bbox
             )
        else:
             print("错误 (Regret): ALNS 实例缺少 _call_planner 方法。")
    except TypeError as te:
        # 如果 _call_planner 不接受 bbox 参数，则回退
        if 'bounding_box' in str(te):
            print(f"警告 (Regret): ALNS._call_planner 不支持 bbox 参数。尝试无 bbox 调用。")
            try:
                best_path = alns_instance._call_planner(
                    task=task,
                    dynamic_obstacles=dynamic_obstacles_base,
                    start_time=0
                )
            except Exception as call_e_no_bbox:
                print(f"错误: 调用 ALNS._call_planner (无 bbox) 失败: {call_e_no_bbox}")
        else:
            # 如果是其他 TypeError，则重新抛出
            print(f"错误: 调用 ALNS._call_planner 时发生 TypeError: {te}")
            # raise te # 或者根据需要处理
    except Exception as call_e:
        print(f"错误: 调用 ALNS._call_planner 时发生未知异常: {call_e}")

    # 评估最优路径成本
    best_cost = float('inf')
    if best_path and isinstance(best_path, PathType) and best_path.sequence:
        try:
            grid_map_eval = getattr(alns_instance, 'grid_map', None)
            if grid_map_eval:
                 cost_dict = best_path.get_cost(
                     grid_map_eval,
                     getattr(alns_instance, 'alpha', 1.0),
                     getattr(alns_instance, 'beta', 0.0),
                     getattr(alns_instance, 'gamma_wait', 0.0),
                     getattr(alns_instance, 'v', 1.0),
                     getattr(alns_instance, 'delta_step', 1.0)
                 )
                 cost = cost_dict.get('total', float('inf'))
            else: cost = float('inf')
        except Exception as cost_e:
            print(f"警告: 计算最优路径成本失败 (AGV {task.agv_id}): {cost_e}")
            cost = float('inf')

        if cost != float('inf'):
            insertion_info.add_option(best_path, cost)
            best_cost = cost
        else:
            best_path = None # 标记最优路径无效

    # 2. 查找次优路径 (如果 k > 1 且找到了有效的最优路径)
    if best_path and k > 1:
        secondary_limit = getattr(alns_instance, 'regret_planner_time_limit_abs', 0.1)
        tried_alternatives = 0
        attempt_count = 0
        max_attempts = max(1, max_regret_attempts)
        path_signature = tuple(best_path.sequence)

        # 筛选可以在最优路径上设置障碍的索引
        eligible_indices = [i for i in range(1, len(best_path.sequence) - 1)
                            if best_path.sequence[i] != best_path.sequence[i-1]] # 避免在等待点设障碍
        random.shuffle(eligible_indices)
        blocked_states_cache: Set[StateType] = set()

        for block_idx in eligible_indices:
            if tried_alternatives >= k - 1 or len(insertion_info.options) >= k:
                break
            if attempt_count >= max_attempts:
                break
            attempt_count += 1

            state_to_block = best_path.sequence[block_idx]
            if state_to_block in blocked_states_cache:
                continue # 跳过已尝试阻塞的状态
            blocked_states_cache.add(state_to_block)

            # 创建包含临时障碍的动态障碍物
            temp_dynamic_obstacles = copy.deepcopy(dynamic_obstacles_base)
            block_t = state_to_block[1]
            if block_t not in temp_dynamic_obstacles:
                temp_dynamic_obstacles[block_t] = set()
            temp_dynamic_obstacles[block_t].add(state_to_block[0])

            # --- 直接调用 planner.plan，确保传递 bbox ---
            alternative_path: Optional[PathType] = None
            try:
                planner_inst = getattr(alns_instance, 'planner', None)
                grid_map_inst = getattr(alns_instance, 'grid_map', None)
                if planner_inst and grid_map_inst:
                     # v6: 传递 bbox
                     alternative_path = planner_inst.plan(
                         grid_map=grid_map_inst,
                         task=task,
                         dynamic_obstacles=temp_dynamic_obstacles,
                         max_time=getattr(alns_instance, 'max_time', 400),
                         cost_weights=getattr(alns_instance, 'cost_weights', (1.0, 0.0, 0.0)),
                         v=getattr(alns_instance, 'v', 1.0),
                         delta_step=getattr(alns_instance, 'delta_step', 1.0),
                         start_time=0,
                         time_limit=secondary_limit,
                         bounding_box=bbox # 传递 bbox
                     )
                else:
                     print("错误 (Regret Alt): 无法访问 planner 或 grid_map。")
            except TypeError as te: # 捕获可能的参数不匹配错误
                if 'bounding_box' in str(te):
                     print(f"警告 (Regret Alt): Planner.plan 不支持 bbox。尝试无 bbox 调用。")
                     try:
                          alternative_path = planner_inst.plan(
                              grid_map=grid_map_inst, task=task, dynamic_obstacles=temp_dynamic_obstacles,
                              max_time=getattr(alns_instance, 'max_time', 400),
                              cost_weights=getattr(alns_instance, 'cost_weights', (1.0, 0.0, 0.0)),
                              v=getattr(alns_instance, 'v', 1.0), delta_step=getattr(alns_instance, 'delta_step', 1.0),
                              start_time=0, time_limit=secondary_limit
                          )
                     except Exception as plan_e_no_bbox:
                          print(f"错误: 调用 Planner.plan (无 bbox) 失败: {plan_e_no_bbox}")
                else:
                     print(f"错误: 调用 Planner.plan 时发生 TypeError: {te}")
                     # raise te
            except Exception as plan_e:
                print(f"错误: 调用 Planner.plan 失败: {plan_e}")

            # 处理找到的次优路径
            if alternative_path and isinstance(alternative_path, PathType) and alternative_path.sequence:
                alt_sig = tuple(alternative_path.sequence)
                if alt_sig != path_signature: # 确保路径不同
                    try:
                        grid_map_eval = getattr(alns_instance, 'grid_map', None)
                        if grid_map_eval:
                            alt_cost_dict = alternative_path.get_cost(
                                grid_map_eval,
                                getattr(alns_instance, 'alpha', 1.0),
                                getattr(alns_instance, 'beta', 0.0),
                                getattr(alns_instance, 'gamma_wait', 0.0),
                                getattr(alns_instance, 'v', 1.0),
                                getattr(alns_instance, 'delta_step', 1.0)
                            )
                            alt_cost = alt_cost_dict.get('total', float('inf'))
                        else: alt_cost = float('inf')
                    except Exception as alt_cost_e:
                        print(f"警告: 计算次优路径成本失败 (AGV {task.agv_id}): {alt_cost_e}")
                        alt_cost = float('inf')

                    # 只有当成本有效且确实比最优成本高时，才添加
                    if alt_cost != float('inf') and alt_cost > best_cost + 1e-6:
                        insertion_info.add_option(alternative_path, alt_cost)
                        tried_alternatives += 1

    return insertion_info

# --- _check_sequence_conflicts_segment (保持 v5) ---
def _check_sequence_conflicts_segment(
    dynamic_obs_check: 'DynObsType',
    segment_to_check: List['StateType']
) -> bool:
    """检查路径段是否与动态障碍冲突。"""
    for node, t in segment_to_check:
        if not isinstance(node, tuple):
            print(f"警告: 冲突检查遇到非元组节点: {node}")
            continue
        obstacles_at_t = dynamic_obs_check.get(t)
        if obstacles_at_t and node in obstacles_at_t:
            return True # 发现冲突
    return False # 未发现冲突

# --- _attempt_insert_wait (保持 v5) ---
def _attempt_insert_wait(
    alns_instance: 'ALNS',
    current_solution: 'SolutionType',
    agv_id: int,
    original_sequence: List['StateType'],
    idx: int, # 插入等待的位置索引 (在该索引的状态之后插入)
    wait_duration: int # 等待的时间步长
) -> Tuple[Optional[List['StateType']], float]:
    """尝试在路径指定索引处插入等待。"""
    if idx < 0 or idx >= len(original_sequence) - 1:
        return None, float('inf') # 不能在最后或之前插入

    node, current_t = original_sequence[idx] # 等待发生在此状态之后

    # 构建插入的等待序列
    new_sequence_part = []
    for w in range(wait_duration):
        new_sequence_part.append((node, current_t + 1 + w))

    # 构建时间平移后的后续序列
    shifted_sequence_part = []
    max_time_limit = getattr(alns_instance, 'max_time', float('inf'))
    for n, t in original_sequence[idx+1:]:
        new_time = t + wait_duration
        if new_time > max_time_limit: # 检查时间是否超限
            return None, float('inf')
        shifted_sequence_part.append((n, new_time))

    # 组合成试验路径
    trial_sequence = original_sequence[:idx+1] + new_sequence_part + shifted_sequence_part

    # 构建用于检查的动态障碍 (排除自己)
    dynamic_obs_check = _build_dynamic_obstacles(current_solution, exclude_agv_id=agv_id)

    # 检查新插入的等待段和时间平移后的段是否与障碍冲突
    if _check_sequence_conflicts_segment(dynamic_obs_check, new_sequence_part):
        return None, float('inf')
    if _check_sequence_conflicts_segment(dynamic_obs_check, shifted_sequence_part):
        return None, float('inf')

    # 评估新路径的成本
    new_total_cost = float('inf')
    try:
        temp_solution = copy.deepcopy(current_solution)
        temp_path_obj = PathType(agv_id=agv_id, sequence=trial_sequence)
        temp_solution[agv_id] = temp_path_obj
        new_cost_dict = alns_instance._calculate_total_cost(temp_solution)
        new_total_cost = new_cost_dict.get('total', float('inf'))
    except Exception as cost_e:
        print(f"错误 (InsertWait): 计算成本时失败: {cost_e}")

    if new_total_cost == float('inf'):
        return None, float('inf')
    else:
        return trial_sequence, new_total_cost

# --- _attempt_delete_wait (保持 v5) ---
def _attempt_delete_wait(
    alns_instance: 'ALNS',
    current_solution: 'SolutionType',
    agv_id: int,
    original_sequence: List['StateType'],
    idx: int, # 等待结束点的索引
    wait_duration_deleted: int # 删除的等待时长
) -> Tuple[Optional[List['StateType']], float]:
    """尝试在路径指定索引处删除等待。"""
    if idx < wait_duration_deleted or idx >= len(original_sequence):
        return None, float('inf') # 索引无效

    # 构建时间提前后的后续序列
    shifted_sequence_part = []
    for n, t in original_sequence[idx+1:]:
         new_time = t - wait_duration_deleted
         # 基本检查，时间不应为负
         if new_time < 0: return None, float('inf')
         shifted_sequence_part.append((n, new_time))

    # 组合成试验路径 (保留到等待开始前的点，然后直接拼接后续部分)
    trial_sequence = original_sequence[:idx - wait_duration_deleted + 1] + shifted_sequence_part

    # 检查时间是否超限 (虽然不太可能，但保险起见)
    max_time_limit = getattr(alns_instance, 'max_time', float('inf'))
    if trial_sequence and trial_sequence[-1][1] > max_time_limit:
        return None, float('inf')

    # 构建用于检查的动态障碍 (排除自己)
    dynamic_obs_check = _build_dynamic_obstacles(current_solution, exclude_agv_id=agv_id)

    # 检查时间提前后的段是否与障碍冲突
    if _check_sequence_conflicts_segment(dynamic_obs_check, shifted_sequence_part):
        return None, float('inf')

    # 评估新路径的成本
    new_total_cost = float('inf')
    try:
        temp_solution = copy.deepcopy(current_solution)
        temp_path_obj = PathType(agv_id=agv_id, sequence=trial_sequence)
        temp_solution[agv_id] = temp_path_obj
        new_cost_dict = alns_instance._calculate_total_cost(temp_solution)
        new_total_cost = new_cost_dict.get('total', float('inf'))
    except Exception as cost_e:
        print(f"错误 (DeleteWait): 计算成本时失败: {cost_e}")

    if new_total_cost == float('inf'):
        return None, float('inf')
    else:
        return trial_sequence, new_total_cost


# ==================================
# --- 破坏算子 (Destroy Operators) ---
# (保持 v5 逻辑和注释, 格式调整)
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
    if not solution:
        return {}, []
    partial_solution = copy.deepcopy(solution)
    # 获取有有效路径的 AGV ID 列表
    agvs_in_solution = [
        aid for aid, p in partial_solution.items()
        if p is not None and isinstance(p, PathType) and p.sequence
    ]
    if not agvs_in_solution:
        return {}, [] # 没有可移除的

    # 确保移除数量不超过现有数量
    actual_removal_count = min(removal_count, len(agvs_in_solution))
    if actual_removal_count <= 0:
        return partial_solution, []

    # 随机抽样
    removed_ids = random.sample(agvs_in_solution, actual_removal_count)
    # 从部分解中删除路径
    for agv_id in removed_ids:
        if agv_id in partial_solution:
            del partial_solution[agv_id]

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
    if not solution:
        return {}, []
    costs: List[Tuple[float, int]] = [] # 存储 (成本, agv_id)
    valid_agvs = []
    grid_map_eval = getattr(alns_instance, 'grid_map', None)

    if not grid_map_eval:
         print("错误 (WorstRemoval): 无法访问地图以计算成本。")
         return copy.deepcopy(solution), []

    for agv_id, path in solution.items():
        if path and isinstance(path, PathType) and path.sequence:
            valid_agvs.append(agv_id)
            cost = float('inf')
            try:
                # 计算单个 AGV 的路径成本 (基于目标函数 1)
                cost_dict = path.get_cost(
                    grid_map_eval,
                    getattr(alns_instance, 'alpha', 1.0),
                    getattr(alns_instance, 'beta', 0.0),
                    getattr(alns_instance, 'gamma_wait', 0.0),
                    getattr(alns_instance, 'v', 1.0),
                    getattr(alns_instance, 'delta_step', 1.0)
                )
                cost = cost_dict.get('total', float('inf'))
            except Exception as e:
                print(f"警告 (WorstRemoval): 计算 AGV {agv_id} 成本失败: {e}")
            # 只有有效的成本才加入列表
            if cost != float('inf'):
                costs.append((cost, agv_id))

    if not valid_agvs:
        return copy.deepcopy(solution), []

    # 按成本降序排序
    costs.sort(key=lambda item: item[0], reverse=True)

    actual_removal_count = min(removal_count, len(costs))
    if actual_removal_count <= 0:
        return copy.deepcopy(solution), []

    # 选择成本最高的 AGV
    removed_ids = [agv_id for cost, agv_id in costs[:actual_removal_count]]
    partial_solution = copy.deepcopy(solution)
    for agv_id in removed_ids:
        if agv_id in partial_solution:
            del partial_solution[agv_id]

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
    if not solution:
        return {}, []
    partial_solution = copy.deepcopy(solution)
    # 获取有有效路径的 AGV ID 列表
    agvs_in_solution = [
        aid for aid, p in partial_solution.items()
        if p is not None and isinstance(p, PathType) and p.sequence
    ]
    if not agvs_in_solution or removal_count <= 0:
        return partial_solution, []

    actual_removal_count = min(removal_count, len(agvs_in_solution))
    if actual_removal_count <= 0:
        return partial_solution, []

    # 随机选择种子 AGV
    seed_agv_id = random.choice(agvs_in_solution)
    removed_ids = [seed_agv_id]
    # 移除种子
    if seed_agv_id in partial_solution:
        del partial_solution[seed_agv_id]

    deterministic_factor = 4 # 影响随机性

    # 循环选择并移除与上一个移除的 AGV 最相关的 AGV
    while len(removed_ids) < actual_removal_count:
        if not partial_solution:
            break # 没有更多 AGV 可移除
        candidates = list(partial_solution.keys())
        if not candidates:
            break

        relatedness_scores: List[Tuple[float, int]] = [] # (相关性得分, agv_id)
        last_removed_id = removed_ids[-1]
        # 获取原始解中最后移除的 AGV 路径 (注意从原始 solution 获取)
        path_last = solution.get(last_removed_id)

        # 计算与候选 AGV 的相关性
        for candidate_id in candidates:
            path_candidate = partial_solution.get(candidate_id)
            relatedness = _calculate_relatedness(alns_instance, path_last, path_candidate)
            relatedness_scores.append((relatedness, candidate_id))

        if not relatedness_scores:
            break # 没有可计算相关性的候选者

        # 按相关性降序排序
        relatedness_scores.sort(key=lambda item: item[0], reverse=True)

        # 引入随机性选择下一个要移除的 AGV
        rand_val = random.random()
        max_index = len(relatedness_scores) - 1
        index_to_pick = min(int(len(relatedness_scores) * (rand_val ** deterministic_factor)), max_index)
        index_to_pick = max(0, index_to_pick) # 确保索引有效

        agv_to_remove = relatedness_scores[index_to_pick][1]
        removed_ids.append(agv_to_remove)
        if agv_to_remove in partial_solution:
            del partial_solution[agv_to_remove]

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
    if not solution:
        return {}, []
    node_time_counts: Dict[StateType, int] = defaultdict(int)
    max_t = 0
    # 统计所有时空节点的占用次数
    for path in solution.values():
        if path and isinstance(path, PathType) and path.sequence:
            max_t = max(max_t, path.get_makespan())
            for state in path.sequence:
                node_time_counts[state] += 1

    path_congestion_scores: List[Tuple[float, int]] = [] # (拥挤度得分, agv_id)
    valid_agvs = []
    # 计算每个 AGV 路径的平均拥挤度
    for agv_id, path in solution.items():
        if path and isinstance(path, PathType) and path.sequence:
            valid_agvs.append(agv_id)
            # 拥挤度 = 路径上每个点被占用的总次数之和 (减去自身占用的一次)
            congestion_score = 0.0
            for state in path.sequence:
                 occupancy = node_time_counts.get(state, 0)
                 if occupancy > 1:
                      congestion_score += float(occupancy - 1)

            # 标准化拥挤度得分（除以路径长度）
            path_len = len(path.sequence)
            normalized_score = congestion_score / path_len if path_len > 0 else 0
            path_congestion_scores.append((normalized_score, agv_id))

    if not valid_agvs:
        return copy.deepcopy(solution), []

    # 按拥挤度降序排序
    path_congestion_scores.sort(key=lambda item: item[0], reverse=True)

    actual_removal_count = min(removal_count, len(path_congestion_scores))
    if actual_removal_count <= 0:
        return copy.deepcopy(solution), []

    # 选择最拥挤的 AGV
    removed_ids = [agv_id for score, agv_id in path_congestion_scores[:actual_removal_count]]
    partial_solution = copy.deepcopy(solution)
    for agv_id in removed_ids:
        if agv_id in partial_solution:
            del partial_solution[agv_id]

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
    if not solution:
        return {}, []
    # 获取有有效路径的 AGV ID 列表
    agvs_in_solution = [
        aid for aid, p in solution.items()
        if p is not None and isinstance(p, PathType) and p.sequence
    ]
    if not agvs_in_solution:
        return {}, []

    # 安全访问冲突计数
    conflict_counter = getattr(alns_instance, 'agv_conflict_counts', Counter())
    conflict_counts = Counter({
        agv_id: conflict_counter.get(agv_id, 0)
        for agv_id in agvs_in_solution
    })

    # 按冲突次数降序排序
    sorted_by_conflict = conflict_counts.most_common()

    actual_removal_count = min(removal_count, len(sorted_by_conflict))
    if actual_removal_count <= 0:
        return copy.deepcopy(solution), []

    # 选择冲突最多的 AGV
    removed_ids = [agv_id for agv_id, count in sorted_by_conflict[:actual_removal_count]]
    partial_solution = copy.deepcopy(solution)
    for agv_id in removed_ids:
        if agv_id in partial_solution:
            del partial_solution[agv_id]

    # 打印调试信息 (如果启用)
    verbose = getattr(alns_instance, 'verbose', False)
    debug_weights = getattr(alns_instance, 'debug_weights', False)
    if verbose and debug_weights:
        removed_counts = [conflict_counts[aid] for aid in removed_ids]
        print(f"    ConflictHistoryRemoval: Removed {len(removed_ids)} AGVs: {removed_ids} (Counts: {removed_counts})")

    return partial_solution, removed_ids


# ==================================
# --- 修复算子 (Repair Operators) ---
# (修改 - v6: 调用 planner 时传递 bbox, 格式调整)
# ==================================

# --- greedy_insertion (修改 - v6) ---
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
    if not isinstance(partial_solution, dict):
        return None
    solution = copy.deepcopy(partial_solution)
    tasks_to_insert = [
        task for task in getattr(alns_instance, 'tasks', [])
        if task.agv_id in removed_ids
    ]
    if not tasks_to_insert: # 如果没有需要插入的任务
         return solution # 直接返回当前解 (可能已经是完整的)

    random.shuffle(tasks_to_insert) # 随机插入顺序

    for task in tasks_to_insert:
        agv_id = task.agv_id
        dynamic_obstacles = _build_dynamic_obstacles(solution)

        # --- v6: 计算包围盒 ---
        bbox: Optional[Tuple[int, int, int, int]] = None
        try:
            grid_map = getattr(alns_instance, 'grid_map', None)
            buffer_val = getattr(alns_instance, 'buffer', 0)
            if grid_map and hasattr(grid_map, 'width') and hasattr(grid_map, 'height'):
                bbox = _calculate_bounding_box(
                    task.start_node, task.goal_node,
                    grid_map.width, grid_map.height, buffer_val
                )
            else:
                print(f"警告 (Greedy): 无法访问地图或 buffer。不使用区域分割。")
        except Exception as bbox_e:
            print(f"警告 (Greedy): 计算 AGV {agv_id} 包围盒失败: {bbox_e}。")

        # --- 调用核心规划器 (假设 ALNS._call_planner v38+ 支持 bbox) ---
        new_path: Optional[PathType] = None
        try:
            if hasattr(alns_instance, '_call_planner'):
                new_path = alns_instance._call_planner(
                    task=task,
                    dynamic_obstacles=dynamic_obstacles,
                    start_time=0,
                    bounding_box=bbox # 传递 bbox
                )
            else:
                print("错误 (Greedy): ALNS 实例缺少 _call_planner 方法。")
        except TypeError as te:
            if 'bounding_box' in str(te):
                print(f"警告 (Greedy): ALNS._call_planner 不支持 bbox。尝试无 bbox 调用。")
                try:
                    new_path = alns_instance._call_planner(task, dynamic_obstacles, start_time=0)
                except Exception as call_e_no_bbox:
                    print(f"错误: 调用 ALNS._call_planner (无 bbox) 失败: {call_e_no_bbox}")
            else:
                print(f"错误: 调用 ALNS._call_planner 时发生 TypeError: {te}")
                # raise te
        except Exception as call_e:
            print(f"错误: 调用 ALNS._call_planner 失败: {call_e}")

        # --- 处理规划结果 ---
        if new_path and isinstance(new_path, PathType) and new_path.sequence:
            solution[agv_id] = new_path
        else:
            # 如果任何一个 AGV 无法插入，则修复失败
            if getattr(alns_instance, 'verbose', False):
                print(f"    Greedy Insert: 失败，无法为 AGV {agv_id} 找到路径。")
            return None # 修复失败

    # --- 检查最终解的完整性和成本有效性 ---
    num_expected_agvs = getattr(alns_instance, 'num_agvs', len(getattr(alns_instance, 'tasks', [])))
    if len(solution) == num_expected_agvs:
         final_cost = float('inf')
         try:
             final_cost_dict = alns_instance._calculate_total_cost(solution)
             final_cost = final_cost_dict.get('total', float('inf'))
         except Exception as final_cost_e:
             print(f"错误 (Greedy): 计算最终成本失败: {final_cost_e}")
             return None

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

# --- regret_insertion (保持 v5 结构，依赖修改后的 _find_insertion_options) ---
def regret_insertion(
    alns_instance: 'ALNS',
    partial_solution: 'SolutionType',
    removed_ids: List[int]
) -> Optional['SolutionType']:
    """
    后悔插入算子 (对应论文 3.5.4 节)。
    优先插入“后悔值”最高的 AGV。内部调用 _find_insertion_options，
    该函数应用区域分割优化 (论文 3.5.2)。
    """
    if not isinstance(partial_solution, dict):
        return None
    solution = copy.deepcopy(partial_solution)
    tasks_map = {
        task.agv_id: task for task in getattr(alns_instance, 'tasks', [])
        if task.agv_id in removed_ids
    }
    if not tasks_map and removed_ids: # 确保任务列表可用
        print("错误 (Regret): 无法获取 AGV 任务信息。")
        return None
    if not removed_ids: # 没有需要插入的
        return solution

    unassigned_agv_ids = removed_ids[:]
    iteration = 0
    max_regret_iterations = len(unassigned_agv_ids) # 最多迭代次数

    # 循环直到所有 AGV 都被插入
    while unassigned_agv_ids and iteration < max_regret_iterations:
        iteration += 1
        insertion_candidates: List[InsertionCost] = []
        regret_k = getattr(alns_instance, 'regret_k', 2) # 获取后悔k值

        # 对每个未分配的 AGV，计算其插入选项和后悔值
        for agv_id in unassigned_agv_ids:
            task = tasks_map.get(agv_id)
            if not task:
                print(f"警告 (Regret): 找不到 AGV {agv_id} 的任务。")
                continue
            # v6: _find_insertion_options 内部会计算并使用 bbox
            candidate_info = _find_insertion_options(alns_instance, task, solution, regret_k)
            candidate_info.calculate_regret(regret_k) # 计算后悔值
            insertion_candidates.append(candidate_info)

        if not insertion_candidates:
            break # 没有候选者，退出

        # 选择后悔值最大的 AGV 进行插入
        insertion_candidates.sort(key=lambda c: c.regret, reverse=True) # 按后悔值降序排序
        best_candidate_to_insert = insertion_candidates[0]
        agv_id_to_insert = best_candidate_to_insert.agv_id
        best_path_for_agv = best_candidate_to_insert.get_best_path()

        # 插入找到的最优路径
        if best_path_for_agv and isinstance(best_path_for_agv, PathType):
            solution[agv_id_to_insert] = best_path_for_agv
            # 从待插入列表移除
            if agv_id_to_insert in unassigned_agv_ids:
                unassigned_agv_ids.remove(agv_id_to_insert)
            if getattr(alns_instance, 'verbose', False):
                regret_val = best_candidate_to_insert.regret
                print(f"    Regret Insert (Iter {iteration}): 成功插入 AGV {agv_id_to_insert} (Regret={regret_val:.2f})")
        else:
            # 如果找不到路径，也将其从未分配列表移除，避免无限循环
            if agv_id_to_insert in unassigned_agv_ids:
                unassigned_agv_ids.remove(agv_id_to_insert)
            if getattr(alns_instance, 'verbose', False):
                print(f"    Regret Insert (Iter {iteration}): 失败，无法为 AGV {agv_id_to_insert} 找到路径，已移除。")

    # --- 检查最终解的完整性和成本有效性 ---
    if not unassigned_agv_ids: # 如果所有 AGV 都处理完毕
         final_cost = float('inf')
         try:
             final_cost_dict = alns_instance._calculate_total_cost(solution)
             final_cost = final_cost_dict.get('total', float('inf'))
         except Exception as final_cost_e:
             print(f"错误 (Regret): 计算最终成本失败: {final_cost_e}")
             return None

         if final_cost != float('inf'):
             return solution # 修复成功
         else:
             if getattr(alns_instance, 'verbose', False):
                 print("    Regret Insert: 失败，最终成本为 Inf")
             return None
    else: # 如果有 AGV 未能插入
         if getattr(alns_instance, 'verbose', False):
             print(f"    Regret Insert: 失败，有 {len(unassigned_agv_ids)} 个 AGV 未插入: {unassigned_agv_ids}")
         return None

# --- wait_adjustment_repair (保持 v5 结构，bbox 在其调用的 greedy_insertion 中处理) ---
def wait_adjustment_repair(
    alns_instance: 'ALNS',
    partial_solution: 'SolutionType',
    removed_ids: List[int]
) -> Optional['SolutionType']:
    """
    等待调整修复算子 (对应论文 3.5.5 节)。
    先用贪婪法插入移除的 AGV (该过程会应用区域分割优化)，
    然后对随机选择的部分 AGV 路径尝试插入/删除短暂等待。
    """
    if not isinstance(partial_solution, dict):
        return None

    # 1. 基础修复 (v6 的 greedy_insertion 会使用 bbox)
    temp_solution = greedy_insertion(alns_instance, partial_solution, removed_ids)
    if temp_solution is None:
        if getattr(alns_instance, 'verbose', False):
            print("    WaitAdjust: 基础修复失败，无法进行调整。")
        return None
    if getattr(alns_instance, 'verbose', False) and getattr(alns_instance, 'debug_weights', False):
        print("    WaitAdjust: 基础修复完成，开始等待调整...")

    current_solution = copy.deepcopy(temp_solution) # 在副本上操作
    num_agvs = getattr(alns_instance, 'num_agvs', len(current_solution))
    num_agvs_to_adjust = max(1, int(num_agvs * 0.2)) # 调整约 20% 的 AGV
    agv_ids_to_consider = list(current_solution.keys())
    random.shuffle(agv_ids_to_consider)
    agv_ids_to_adjust = agv_ids_to_consider[:num_agvs_to_adjust] # 随机选择

    made_change_overall = False # 标记是否进行了任何有效调整
    max_adjustment_attempts_per_agv = 5 # 每个 AGV 最多尝试调整次数
    wait_delta_max = 2 # 尝试插入/删除的最大等待时长

    # 2. 对选定的 AGV 进行等待调整
    for agv_id in agv_ids_to_adjust:
        path_obj = current_solution.get(agv_id)
        # 跳过无效路径
        if not path_obj or not isinstance(path_obj, PathType) or not path_obj.sequence or len(path_obj.sequence) < 2:
            continue

        # 获取当前总成本
        original_total_cost = float('inf')
        try:
            original_total_cost = alns_instance._calculate_total_cost(current_solution).get('total', float('inf'))
        except Exception as cost_e:
            print(f"警告 (WaitAdjust): 计算 AGV {agv_id} 初始成本失败: {cost_e}")
            continue # 无法计算成本，跳过此 AGV

        best_known_sequence_for_agv = path_obj.sequence # 当前 AGV 的最佳已知序列
        best_known_cost_for_agv = original_total_cost # 对应的总成本

        attempts = 0
        # 可以尝试调整的位置 (在两个状态之间)
        path_indices = list(range(len(best_known_sequence_for_agv) - 1))
        random.shuffle(path_indices)
        agv_made_change_this_loop = False

        # 随机尝试在不同位置插入或删除等待
        for idx in path_indices:
            if attempts >= max_adjustment_attempts_per_agv:
                break
            attempts += 1
            improvement_found_at_idx = False

            # --- 尝试插入等待 ---
            for wait_duration in range(1, wait_delta_max + 1):
                # 调用辅助函数尝试插入
                trial_sequence, new_cost = _attempt_insert_wait(
                    alns_instance, current_solution, agv_id,
                    best_known_sequence_for_agv, idx, wait_duration
                )
                # 如果成功且成本降低，则接受更改
                if trial_sequence is not None and new_cost < best_known_cost_for_agv - 1e-6:
                    if getattr(alns_instance, 'verbose', False) and getattr(alns_instance, 'debug_weights', False):
                        print(f"      WaitAdjust AGV {agv_id}: Inserted {wait_duration} wait at index {idx}. Cost {best_known_cost_for_agv:.2f} -> {new_cost:.2f}")
                    best_known_sequence_for_agv = trial_sequence
                    best_known_cost_for_agv = new_cost
                    agv_made_change_this_loop = True
                    improvement_found_at_idx = True
                    # 更新当前解以反映更改，影响后续尝试
                    current_solution[agv_id].sequence = best_known_sequence_for_agv
                    break # 在此位置找到改进，尝试下一个位置

            if improvement_found_at_idx:
                continue # 如果插入成功，跳过在此位置的删除尝试

            # --- 尝试删除等待 ---
            # 检查当前点是否是等待结束点 (当前节点与前一节点相同)
            if idx > 0 and best_known_sequence_for_agv[idx][0] == best_known_sequence_for_agv[idx-1][0]:
                prev_node, prev_t = best_known_sequence_for_agv[idx-1]
                current_node, current_t = best_known_sequence_for_agv[idx]
                # 注意：这里计算的是 idx 处的等待时间，而不是后面一段的
                wait_duration_at_prev = current_t - prev_t # 计算实际等待时间

                # 只尝试删除短暂的等待
                if wait_duration_at_prev > 0 and wait_duration_at_prev <= wait_delta_max:
                    # 调用辅助函数尝试删除 (注意：idx 对应的是等待段的结束点)
                    trial_sequence, new_cost = _attempt_delete_wait(
                        alns_instance, current_solution, agv_id,
                        best_known_sequence_for_agv, idx, wait_duration_at_prev
                    )
                    # 如果成功且成本降低，则接受更改
                    if trial_sequence is not None and new_cost < best_known_cost_for_agv - 1e-6:
                        if getattr(alns_instance, 'verbose', False) and getattr(alns_instance, 'debug_weights', False):
                            print(f"      WaitAdjust AGV {agv_id}: Deleted {wait_duration_at_prev} wait ending at index {idx}. Cost {best_known_cost_for_agv:.2f} -> {new_cost:.2f}")
                        best_known_sequence_for_agv = trial_sequence
                        best_known_cost_for_agv = new_cost
                        agv_made_change_this_loop = True
                        # 更新当前解
                        current_solution[agv_id].sequence = best_known_sequence_for_agv
                        # 删除成功后，可能需要重新计算可调整的位置索引？
                        # 为简单起见，暂时不重新计算，继续尝试下一个随机位置

        if agv_made_change_this_loop:
            made_change_overall = True # 标记整体有变化

    # 如果进行了任何有效调整，打印最终成本
    verbose = getattr(alns_instance, 'verbose', False)
    debug_weights = getattr(alns_instance, 'debug_weights', False)
    if made_change_overall and verbose and debug_weights:
        final_cost = float('inf')
        try:
            final_cost_dict = alns_instance._calculate_total_cost(current_solution)
            final_cost = final_cost_dict.get('total', float('inf'))
        except Exception as final_cost_e:
            print(f"警告 (WaitAdjust): 计算最终成本失败: {final_cost_e}")
        print(f"    WaitAdjust: Finished adjustments. Final cost: {final_cost:.2f}")

    # --- 检查最终解的成本有效性 ---
    final_check_cost = float('inf')
    try:
        final_cost_dict_check = alns_instance._calculate_total_cost(current_solution)
        final_check_cost = final_cost_dict_check.get('total', float('inf'))
    except Exception as final_cost_e_check:
        print(f"错误 (WaitAdjust): 计算最终检查成本失败: {final_cost_e_check}")
        return None

    if final_check_cost == float('inf'):
         if verbose:
             print("    WaitAdjust: 调整后最终成本为 Inf，算子失败。")
         return None

    return current_solution # 返回调整后的解
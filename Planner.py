# Planner.py-v24 (支持 AGV 特定速度)
"""
实现时空 A* (Time-Window A*) 路径规划算法，用于为单个 AGV 规划无冲突路径。
这是 ALNS 和基准算法的核心路径查找组件。

与论文的关联:
- 核心规划器: 作为 ALNS (论文 3.5.1) 和基准算法 (论文 4.1) 中用于生成
             单条 AGV 路径的核心算法。
- 成本评估:
    - _heuristic: 计算启发式成本，通常基于到目标的估计行驶时间。
                  **v24 修改**: 现在基于特定 AGV 速度计算。
    - _calculate_turn_cost: 计算转弯成本，对应论文目标函数 (公式 1) 的 beta 项。
    - 内部成本累加: g_cost 的累加考虑了行驶 (alpha)、转弯 (beta) 和等待 (gamma_wait)
                    成本，与论文目标函数一致。
- 约束处理:
    - 节点冲突 (论文 约束 12): 通过检查 `dynamic_obstacles` 来避免。
    - 障碍物 (论文 约束 13): 通过调用 `grid_map.is_valid` 或类似方法避免。
    - 边冲突 (非模型约束): 通过可选的启发式惩罚 (`edge_conflict_penalty_factor`)
                      或在 ALNS 的冲突解决阶段处理。
- **v24 修改**: `plan` 方法接收并使用 `agv_speed` 参数，以支持异构 AGV 场景。

版本变更 (v23 -> v24):
- **修改**: `plan` 方法签名，增加 `agv_speed` 参数。
- **修改**: `_heuristic` 方法签名，接收 `agv_speed` 参数，并使用它计算启发值。
- **修改**: `plan` 方法内部调用 `calculate_tij` 和 `_heuristic` 时传递 `agv_speed`。
- 保持: v23 的其他所有类、方法和逻辑（包括 PlannerState, _calculate_turn_cost, _reconstruct_path, 节点类型检查等）保持不变。
- 依赖: Map.py(v9+), DataTypes.py(v11+)。
"""
import heapq
import time
import math
from typing import List, Tuple, Dict, Set, Optional, NamedTuple, TYPE_CHECKING
import traceback

# --- 类型提示导入 ---
if TYPE_CHECKING:
    from Map import GridMap, Node # 假设 Map.py 提供了 GridMap 和 Node
# --- 从项目模块导入 ---
try:
    # 确保从包含最新定义的 DataTypes (v11+) 导入
    from DataTypes import Task, Path, TimeStep, State, DynamicObstacles, calculate_tij, Node as NodeType, CostDict
except ImportError as e:
    print(f"错误: 导入 Planner 依赖项失败 (DataTypes v11+): {e}")
    # 定义临时的占位符类型，以便代码能解析
    Task = type('Task', (object,), {})
    Path = type('Path', (object,), {})
    TimeStep = int
    State = Tuple
    DynamicObstacles = Dict
    NodeType = Tuple
    CostDict = Dict
    # 定义占位符 calculate_tij
    def calculate_tij(*args, **kwargs) -> TimeStep: return 1

# --- PlannerState---
class PlannerState(NamedTuple):
    """存储规划过程中的状态信息，用于优先队列。"""
    f_cost: float # 估计总成本 (g + h)
    g_cost: float # 从起点到当前状态的实际成本
    h_cost: float # 从当前状态到目标的启发式成本
    state: State # 当前状态 (Node, TimeStep)
    parent: Optional['PlannerState'] # 父状态，用于路径回溯
    turn_count: int = 0 # 从起点到当前状态的转弯次数 (可选，用于调试或复杂成本)

    def __lt__(self, other: 'PlannerState') -> bool:
        """比较函数，用于优先队列排序。优先 f_cost，然后 h_cost。"""
        # 优先比较 f_cost，如果接近则比较 h_cost (启发式优先)
        if abs(self.f_cost - other.f_cost) < 1e-9:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost

    def __eq__(self, other: object) -> bool:
        """比较状态是否相等 (仅比较核心 state)。"""
        if not isinstance(other, PlannerState):
            return NotImplemented
        return self.state == other.state

    def __hash__(self) -> int:
        """计算状态的哈希值 (仅基于核心 state)。"""
        return hash(self.state)

# --- TWA* Planner 类 (v24 - 支持 AGV 特定速度) ---
class TWAStarPlanner:
    """
    实现时空 A* (Time-Window A*) 路径规划算法。
    """
    def __init__(self):
        """初始化规划器。"""
        # 边冲突启发式惩罚因子 (可以调整, 0 表示禁用)
        self.edge_conflict_penalty_factor: float = 0.5 # 示例值

    # --- 修改: 接收 agv_speed ---
    def _heuristic(self, node: 'NodeType', goal: 'NodeType', alpha: float, agv_speed: float, delta_step: float) -> float:
        """
        计算从 node 到 goal 的启发式成本 (Octile 距离的估计时间步 * alpha)。
        **v24 修改**: 使用特定的 `agv_speed`。

        Args:
            node: 当前节点。
            goal: 目标节点。
            alpha: 行驶成本权重。
            agv_speed: 此 AGV 的速度 (格/秒)。
            delta_step: 每个时间步的时长 (秒)。

        Returns:
            启发式成本估计值。
        """
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        # Octile distance steps calculation (比曼哈顿距离更准确)
        orthogonal_steps = 1.0 # 直线移动成本（步数）

        # --- 使用传入的 agv_speed ---
        # 估计对角线移动所需的时间
        diagonal_time = math.sqrt(2.0) / agv_speed if agv_speed > 0 else float('inf')
        # 将时间转换为时间步数（向上取整）
        diagonal_steps = math.ceil(diagonal_time / delta_step) if delta_step > 0 and agv_speed > 0 else float('inf')
        # --- ---------------------- ---

        # 确保对角线移动至少花费 1 步，且成本不低于直线移动
        diagonal_steps = max(1.0, diagonal_steps)
        # Octile 距离步数 = 直线步数 * (长边差) + (对角线步数 - 直线步数) * (短边差)
        h_steps = orthogonal_steps * max(dx, dy) + (diagonal_steps - orthogonal_steps) * min(dx, dy)
        # 最终启发成本 = 步数 * 行驶成本权重
        return alpha * h_steps

    # --- _calculate_turn_cost---
    def _calculate_turn_cost(self, parent_node: Optional['NodeType'], current_node: 'NodeType', next_node: 'NodeType', beta: float) -> float:
        """
        计算从 parent->current 到 current->next 的转弯成本。

        Args:
            parent_node: 上一个节点 (如果存在)。
            current_node: 当前节点。
            next_node: 下一个节点。
            beta: 转弯成本权重。

        Returns:
            转弯成本 (如果发生转弯则为 beta，否则为 0)。
        """
        # 如果没有父节点、原地等待、移动到相同节点或直接掉头，则无转弯成本
        if parent_node is None or parent_node == current_node or current_node == next_node or parent_node == next_node:
            return 0.0

        # 计算移动向量
        dx1 = current_node[0] - parent_node[0]
        dy1 = current_node[1] - parent_node[1]
        dx2 = next_node[0] - current_node[0]
        dy2 = next_node[1] - current_node[1]

        # 检查零向量 (例如，在起点或等待后移动)
        if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
            return 0.0

        # 使用叉积检查向量是否共线 (叉积非零表示不共线，即发生转弯)
        # 比计算角度更快
        if abs(dx1 * dy2 - dx2 * dy1) > 1e-9:
             return beta # 应用转弯成本

        # 如果共线（直线前进或 180 度掉头），无转弯成本
        return 0.0

    # --- _reconstruct_path---
    def _reconstruct_path(self, goal_state_info: PlannerState, agv_id: int) -> 'Path':
        """从目标状态回溯以构建最终路径。"""
        sequence: List[State] = []
        current: Optional[PlannerState] = goal_state_info
        while current is not None:
            sequence.append(current.state)
            current = current.parent
        sequence.reverse() # 路径应从起点到终点
        if not sequence:
            # 理论上不应发生，如果 goal_state_info 有效
            print(f"严重错误 (Planner): 找到目标状态 {goal_state_info} 但无法回溯路径！AGV={agv_id}")
            # 返回一个包含单个状态的路径，避免后续处理空路径出错
            return Path(agv_id, [goal_state_info.state] if goal_state_info else [])
        return Path(agv_id, sequence)

    # --- 修改: plan 方法接收 agv_speed ---
    def plan(self, grid_map: 'GridMap', task: Task, dynamic_obstacles: DynamicObstacles, max_time: TimeStep,
             cost_weights: Tuple[float, float, float],
             agv_speed: float, # <--- 新增: AGV 特定速度
             delta_step: float, start_time: TimeStep = 0,
             time_limit: Optional[float] = None, bounding_box: Optional[Tuple[int, int, int, int]] = None
             ) -> Optional['Path']:
        """
        执行 TWA* 路径规划。
        **v24 修改**: 使用特定的 `agv_speed` 进行规划。

        Args:
            grid_map: GridMap 地图对象。
            task: 当前 AGV 的任务 (起点, 终点)。
            dynamic_obstacles: 动态障碍物字典 {time: set_of_nodes}。
            max_time: 规划的最大时间步。
            cost_weights: (alpha, beta, gamma_wait) 成本权重。
            agv_speed: 此 AGV 的速度 (格/秒)。
            delta_step: 每个时间步的时长 (秒)。
            start_time: AGV 开始规划的时间步。
            time_limit: 规划的 CPU 时间限制 (秒, 可选)。
            bounding_box: 可选的搜索区域限制 (min_x, max_x, min_y, max_y)。

        Returns:
            如果找到路径，则返回 Path 对象，否则返回 None。
        """
        agv_id = task.agv_id
        start_node: NodeType = task.start_node
        goal_node: NodeType = task.goal_node
        alpha, beta, gamma_wait = cost_weights
        # 计算边冲突启发式惩罚值
        edge_conflict_cost_penalty = gamma_wait * self.edge_conflict_penalty_factor

        # --- 基本有效性检查---
        try:
            if not grid_map.is_within_bounds(*start_node):
                # print(f"调试 (Planner): 起点 {start_node} 超出边界。")
                return None
            if not grid_map.is_within_bounds(*goal_node):
                # print(f"调试 (Planner): 终点 {goal_node} 超出边界。")
                return None
            if grid_map.is_obstacle(*start_node):
                # print(f"调试 (Planner): 起点 {start_node} 是障碍物。")
                return None # 起点不能是障碍物
            if grid_map.is_obstacle(*goal_node):
                # print(f"调试 (Planner): 终点 {goal_node} 是障碍物。")
                return None # 终点不能是障碍物
            if bounding_box:
                # 检查起终点是否在包围盒内
                if not (bounding_box[0] <= start_node[0] <= bounding_box[1] and bounding_box[2] <= start_node[1] <= bounding_box[3]):
                    # print(f"调试 (Planner): 起点 {start_node} 不在包围盒 {bounding_box} 内。")
                    return None
                if not (bounding_box[0] <= goal_node[0] <= bounding_box[1] and bounding_box[2] <= goal_node[1] <= bounding_box[3]):
                    # print(f"调试 (Planner): 终点 {goal_node} 不在包围盒 {bounding_box} 内。")
                    return None
        except AttributeError as ae:
            print(f"错误 (Planner): GridMap 缺少必要方法 (is_within_bounds 或 is_obstacle): {ae}")
            return None
        except Exception as e:
            print(f"错误 (Planner): 检查起点/终点时出错: {e}")
            return None
        # ----------------------

        # 初始化 Open Set (优先队列) 和 Closed Set
        open_set: List[PlannerState] = [] # 存储待探索状态
        closed_set: Dict[State, float] = {} # 存储已访问状态及其最小 g_cost
        start_plan_time = time.perf_counter() # 记录规划开始时间

        # --- 初始化起始状态 (使用 agv_speed 计算启发值) ---
        # --- 调用修改后的 _heuristic ---
        initial_h_cost = self._heuristic(start_node, goal_node, alpha, agv_speed, delta_step)
        # --- ------------------------- ---
        if initial_h_cost == float('inf'):
            # print(f"调试 (Planner): 无法计算从 {start_node} 到 {goal_node} 的启发值。")
            return None # 无法到达目标
        initial_state: State = (start_node, start_time)
        initial_planner_state = PlannerState(initial_h_cost, 0.0, initial_h_cost, initial_state, None, 0)
        heapq.heappush(open_set, initial_planner_state)
        closed_set[initial_state] = 0.0 # 记录起点成本为 0
        # --------------

        # --- A* 主循环 ---
        while open_set:
            # 检查 CPU 时间限制
            if time_limit and (time.perf_counter() - start_plan_time > time_limit):
                # print(f"调试 (Planner): AGV {agv_id} 规划超时 ({time_limit}s)。")
                return None

            # 从优先队列获取成本最低的状态
            try:
                current_planner_state: PlannerState = heapq.heappop(open_set)
            except IndexError:
                # print(f"调试 (Planner): AGV {agv_id} Open set 为空，未找到路径。")
                break # Open set 为空，路径未找到

            current_g_cost = current_planner_state.g_cost
            current_state = current_planner_state.state
            current_node, current_time = current_state

            # 如果已找到更优路径到达此状态，则跳过 (Pruning)
            # 增加一个小的容差 (1e-9) 来处理浮点数比较问题
            if current_state in closed_set and current_g_cost > closed_set[current_state] + 1e-9:
                continue

            # 检查是否到达目标节点
            if current_node == goal_node:
                # print(f"调试 (Planner): AGV {agv_id} 找到路径！")
                return self._reconstruct_path(current_planner_state, agv_id) # 找到路径

            # 获取父节点用于计算转弯成本
            parent_node: Optional[NodeType] = current_planner_state.parent.state[0] if current_planner_state.parent else None

            # --- 探索邻居节点（包括原地等待） ---
            try:
                # 获取当前节点的所有有效邻居（通常是四向或八向）
                neighbors = grid_map.get_neighbors(current_node)
                # 将原地等待也作为一个可能的“移动”选项
                possible_next_nodes = neighbors + [current_node]
            except AttributeError as ae:
                print(f"错误 (Planner): GridMap 缺少 get_neighbors 方法: {ae}")
                return None # 无法继续规划
            except Exception as e:
                print(f"错误 (Planner): 获取邻居 {current_node} 时出错: {e}")
                continue # 跳过当前节点，尝试下一个

            for next_node in possible_next_nodes:
                # 1. 包围盒检查 (如果启用)
                if bounding_box:
                    nx, ny = next_node
                    if not (bounding_box[0] <= nx <= bounding_box[1] and bounding_box[2] <= ny <= bounding_box[3]):
                        continue # 节点超出搜索范围

                # 2. 节点类型检查
                is_valid_node_type = False
                try:
                    # 调用 Map 中添加的方法获取节点类型
                    neighbor_type = grid_map.get_node_type(next_node[0], next_node[1])

                    if next_node == goal_node:
                        # 目标节点: 允许，只要不是硬障碍物 (类型 1)
                        if neighbor_type != 1: # 假设类型 1 是唯一的硬障碍物类型
                            is_valid_node_type = True
                    else:
                        # 中间节点: 必须是通道 (类型 0)
                        if neighbor_type == 0: # 假设类型 0 是通道
                            is_valid_node_type = True

                except IndexError: # 处理 get_node_type 可能抛出的界外错误
                    is_valid_node_type = False # 界外节点类型无效
                except AttributeError as ae: # 处理 GridMap 没有 get_node_type 方法的错误
                    print(f"错误 (Planner): GridMap 对象缺少 get_node_type 方法: {ae}。无法执行节点类型检查。")
                    is_valid_node_type = False # 如果无法检查，视为无效
                except Exception as e: # 捕获其他可能的异常
                    print(f"错误 (Planner): 获取节点 {next_node} 类型时发生未知错误: {e}")
                    is_valid_node_type = False # 未知错误视为无效
                if not is_valid_node_type:
                    continue # 跳过无效类型的节点

                # 3. 计算移动时间和下一时间步 (使用 agv_speed)
                is_move = (next_node != current_node)
                try:
                    # --- 调用修改后的 calculate_tij，传入 agv_speed ---
                    ideal_time_steps = calculate_tij(current_node, next_node, agv_speed, delta_step, grid_map) if is_move else 1
                    # --- ------------------------------------------ ---
                except Exception as e:
                    # print(f"调试 (Planner): 计算 tij({current_node}, {next_node}) 出错: {e}")
                    continue # 无法计算时间步，跳过

                if ideal_time_steps == float('inf') or ideal_time_steps <= 0:
                    continue # 无效移动或时间步

                # 在规划时，假设 AGV 按理论时间移动
                actual_time_steps = ideal_time_steps
                next_time = current_time + actual_time_steps

                # 检查是否超过最大时间范围
                if next_time > max_time:
                    continue # 超出时间范围

                # 4. 检查节点冲突 (时空障碍物)
                collision = False
                # 需要检查从 (current_time + 1) 到 next_time 的所有时间步
                # 因为 AGV 在移动过程中会占用 [current_time+1, next_time] 的时间
                for t_check in range(current_time + 1, next_time + 1):
                    obstacles_at_t = dynamic_obstacles.get(t_check)
                    # 如果在 t_check 时刻存在动态障碍，并且目标节点 next_node 被占用
                    if obstacles_at_t and next_node in obstacles_at_t:
                        collision = True
                        break # 发现碰撞，无需继续检查
                if collision:
                    continue # 存在节点冲突，跳过此移动

                # 5. (可选) 检查边冲突启发式
                # 这是一个简化的迎面冲突检查启发式
                # 检查在移动开始的下一个时间步 (t = current_time + 1)，
                # 目标节点 next_node 是否被其他 AGV 占用。
                # 注意：这不能完全避免边冲突，更鲁棒的处理在 ALNS 的冲突解决阶段。
                edge_penalty = 0.0
                if is_move and self.edge_conflict_penalty_factor > 0:
                    check_time_for_edge = current_time + 1
                    if check_time_for_edge <= max_time: # 确保检查时间有效
                        obstacles_at_next_step = dynamic_obstacles.get(check_time_for_edge)
                        # 如果下一个时间步，目标节点被占用
                        if obstacles_at_next_step and next_node in obstacles_at_next_step:
                            edge_penalty = edge_conflict_cost_penalty # 应用惩罚

                # 6. 计算新的 g_cost (累加成本)
                cost_increment = 0.0
                new_turn_count = current_planner_state.turn_count
                if is_move:
                    cost_increment += alpha * float(actual_time_steps) # 行驶成本
                    turn_cost = self._calculate_turn_cost(parent_node, current_node, next_node, beta)
                    cost_increment += turn_cost # 转弯成本
                    if turn_cost > 1e-9: new_turn_count += 1
                    cost_increment += edge_penalty # 加上边冲突启发式惩罚
                else: # 等待
                    cost_increment += gamma_wait * float(actual_time_steps) # 等待成本

                new_g_cost = current_g_cost + cost_increment
                next_state: State = (next_node, next_time)

                # 7. 检查是否已在 closed_set 且成本更差
                if next_state in closed_set and new_g_cost >= closed_set[next_state] - 1e-9:
                    continue # 已有更优或等优路径到达此状态

                # 8. 计算 h_cost 和 f_cost (使用 agv_speed)
                # --- 调用修改后的 _heuristic ---
                new_h_cost = self._heuristic(next_node, goal_node, alpha, agv_speed, delta_step)
                # --- ------------------------- ---
                if new_h_cost == float('inf'):
                    continue # 无法从这里到达目标

                new_f_cost = new_g_cost + new_h_cost

                # 9. 更新 open_set 和 closed_set
                # 创建新的规划器状态
                new_planner_state = PlannerState(new_f_cost, new_g_cost, new_h_cost, next_state, current_planner_state, new_turn_count)
                # 加入 Open Set
                heapq.heappush(open_set, new_planner_state)
                # 更新 Closed Set 中的成本
                closed_set[next_state] = new_g_cost

        # 如果循环结束仍未返回，说明 Open set 为空，未找到路径
        # print(f"调试 (Planner): AGV {agv_id} 最终未找到路径。")
        return None

# --- 示例用法 (更新以测试新的 plan 方法) ---
if __name__ == '__main__':
    print("--- TWA* Planner 测试 (v24 - 支持 AGV 特定速度) ---")
    try:
        # 假设 InstanceGenerator v15 可用
        from InstanceGenerator import load_fixed_scenario_1
        from Map import GridMap
        from DataTypes import Solution, DynamicObstacles, Task, Path as AgentPath, CostDict
    except ImportError as import_error:
        print(f"错误: 导入必需模块失败: {import_error}")
        exit(1)

    print("加载固定算例场景 1 (需要 Map.json)...")
    # 调用 v15 函数，获取地图、任务和速度
    instance_data_s1 = load_fixed_scenario_1(json_map_file="Map.json", expansion_radius=0, assign_speeds=True)
    if not instance_data_s1:
        print("错误: 无法加载场景 1 数据。")
        exit(1)
    test_map_s1, test_tasks_s1, test_speeds_s1 = instance_data_s1 # 解包
    if not test_tasks_s1:
        print("错误: 场景 1 任务列表为空。")
        exit(1)

    # 选择一个 AGV 进行测试
    test_agv_id = 0 # 选择 AGV 0
    test_task_agv0 = next((t for t in test_tasks_s1 if t.agv_id == test_agv_id), None)
    if test_task_agv0 is None:
        print(f"错误: 找不到 AGV {test_agv_id} 的任务。")
        exit(1)
    test_speed_agv0 = test_speeds_s1.get(test_agv_id, 1.0) # 获取其速度

    empty_dyn_obs: DynamicObstacles = {}
    print(f"\n测试 AGV {test_task_agv0.agv_id} (速度: {test_speed_agv0:.2f}) 规划:")
    print(f"  地图: {test_map_s1}")
    print(f"  任务: {test_task_agv0}")

    planner_fixed = TWAStarPlanner()
    cost_w = (1.0, 0.3, 0.8)
    # v_avg = 1.0 # 平均速度，现在由 agv_speed 参数替代
    step_t = 1.0
    max_t_horizon = 600 # 增加最大时间
    time_lim = 30.0

    print(f"\n测试 1: 不使用包围盒")
    start_t1 = time.perf_counter()
    # --- 调用修改后的 plan，传入特定速度 ---
    result_path1: Optional[AgentPath] = planner_fixed.plan(
        test_map_s1, test_task_agv0, empty_dyn_obs, max_t_horizon, cost_w,
        test_speed_agv0, # <--- 传入 AGV 0 的速度
        step_t, time_limit=time_lim, bounding_box=None
    )
    # --- -------------------------------- ---
    duration1 = time.perf_counter() - start_t1
    if result_path1:
        print(f"  规划成功！耗时: {duration1:.4f}s, Makespan: {result_path1.get_makespan()}")
        # --- 调用修改后的 get_cost ---
        cost_dict1 = result_path1.get_cost(test_map_s1, *cost_w, test_speed_agv0, step_t)
        # --- ---------------------- ---
        print(f"  成本: Total={cost_dict1.get('total',-1):.2f}")
    else:
        print(f"  规划失败。耗时: {duration1:.4f}s")

    # --- 测试包围盒 (逻辑不变，但 plan 调用需传入速度) ---
    try:
        s_node = test_task_agv0.start_node; g_node = test_task_agv0.goal_node; buffer = 1
        min_x_test = max(0, min(s_node[0], g_node[0]) - buffer); max_x_test = min(test_map_s1.width - 1, max(s_node[0], g_node[0]) + buffer)
        min_y_test = max(0, min(s_node[1], g_node[1]) - buffer); max_y_test = min(test_map_s1.height - 1, max(s_node[1], g_node[1]) + buffer)
        test_bbox = (min_x_test, max_x_test, min_y_test, max_y_test)
        print(f"\n测试 2: 使用包围盒 {test_bbox}")
        start_t2 = time.perf_counter()
        # --- 调用修改后的 plan，传入特定速度和包围盒 ---
        result_path2: Optional[AgentPath] = planner_fixed.plan(
            test_map_s1, test_task_agv0, empty_dyn_obs, max_t_horizon, cost_w,
            test_speed_agv0, # <--- 传入 AGV 0 的速度
            step_t, time_limit=time_lim, bounding_box=test_bbox
        )
        # --- ----------------------------------------- ---
        duration2 = time.perf_counter() - start_t2
        if result_path2:
            print(f"  规划成功！耗时: {duration2:.4f}s, Makespan: {result_path2.get_makespan()}")
            cost_dict2 = result_path2.get_cost(test_map_s1, *cost_w, test_speed_agv0, step_t)
            print(f"  成本: Total={cost_dict2.get('total',-1):.2f}")
        else:
            print(f"  规划失败。耗时: {duration2:.4f}s")
    except AttributeError as map_err:
        print(f"错误: 测试包围盒时无法访问地图属性: {map_err}")
    except Exception as bbox_err:
        print(f"错误: 测试包围盒时发生异常: {bbox_err}")

    # --- 测试边冲突启发式 (逻辑不变，但 plan 调用需传入速度) ---
    # 假设 AGV 1 的速度也已知
    test_agv1_id = 1
    test_speed_agv1 = test_speeds_s1.get(test_agv1_id, 1.0)
    test_task_agv1_edge = Task(agv_id=test_agv1_id, start_node=(0,1), goal_node=(2,1)) # 假设 AGV 1 有此任务
    edge_dyn_obs: DynamicObstacles = {1: {(1, 1)}} # 模拟 AGV 0 在 t=1 时占用 (1,1)
    print(f"\n测试 3: 边冲突启发式 (测试 AGV {test_agv1_id}, 速度 {test_speed_agv1:.2f})")
    print(f"  任务: {test_task_agv1_edge}")
    print(f"  动态障碍: {edge_dyn_obs}")
    start_t_edge = time.perf_counter()
    # --- 调用修改后的 plan，传入 AGV 1 的速度 ---
    result_path_edge: Optional[AgentPath] = planner_fixed.plan(
        test_map_s1, test_task_agv1_edge, edge_dyn_obs, max_t_horizon, cost_w,
        test_speed_agv1, # <--- 传入 AGV 1 的速度
        step_t, time_limit=time_lim, bounding_box=None
    )
    # --- --------------------------------------- ---
    duration_edge = time.perf_counter() - start_t_edge
    if result_path_edge:
        print(f"  规划成功！耗时: {duration_edge:.4f}s, Makespan: {result_path_edge.get_makespan()}")
        cost_dict_edge = result_path_edge.get_cost(test_map_s1, *cost_w, test_speed_agv1, step_t)
        print(f"  成本: Total={cost_dict_edge.get('total',-1):.2f}")
        # 观察路径是否避开或等待
        if len(result_path_edge.sequence) > 1:
            first_step = result_path_edge.sequence[1]
            if first_step == ((0,1), 1): print("  观察: AGV 1 可能在起点等待。")
            elif first_step[0] != (1,1): print("  观察: AGV 1 可能绕路。")
            else: print("  观察: AGV 1 直接移动到(1,1) (未受启发式影响或惩罚不足)。")
    else:
        print(f"  规划失败。耗时: {duration_edge:.4f}s")
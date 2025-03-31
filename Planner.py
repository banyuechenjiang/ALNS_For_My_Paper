# Planner.py-v17 (重构，添加边冲突启发式，遵循严格规范)
"""
实现带时间窗的 A* (TWA*) 路径规划器。
该规划器是 ALNS 算法内部的核心引擎，负责为单个 AGV 在考虑时空约束
（动态障碍物）和可选的搜索区域限制（包围盒）的情况下规划从起点到终点的
成本最优路径。同时，包含了一个简单的启发式规则来尝试避免一些边冲突。

与论文的关联:
- 核心规划器 (论文 3.5.1): 本模块实现了论文第 3.5.1 节定义的 TWA* 核心规划器。
- 目标函数评估 (论文 公式 1): TWA* 搜索过程中计算的 g_cost 累积了行驶成本 (α * t_ij)
  和转弯成本 (β * θ_ijl)，直接关联目标函数的前两项。等待成本 (γ_wait * w_ikt)
  通过允许原地等待动作隐式考虑。边冲突启发式会增加额外成本。
- 约束满足:
    - 状态转移 (论文 约束 6, 7, 8): 通过 A* 的节点扩展逻辑实现。
    - 节点冲突 (论文 约束 12): 通过检查 dynamic_obstacles 实现 (依赖正确构建的数据)。
    - 障碍物 (论文 约束 13): 通过 grid_map.is_valid 和 get_neighbors 方法处理。
    - 边冲突 (启发式): 新增一个简单启发式尝试减少边冲突，非严格约束。
- 区域分割 (论文 3.5.2): 通过可选的 `bounding_box` 参数实现。

版本变更 (v15 -> v17 - 内部版本迭代):
- **新增**: 添加了一个简单的边冲突启发式惩罚，当下一步节点在紧邻的未来时间步
           (current_time + 1) 被其他 AGV 占用时，增加移动成本。
- **重构**: 完全遵循用户要求的代码结构、注释、错误处理和返回逻辑规范。
- **修复**: 修正了 `if __name__ == '__main__':` 中的导入错误。
- **保持**: 保留了 v15 的包围盒功能和核心 TWA* 逻辑。
"""
import heapq
import time
import math
from typing import List, Tuple, Dict, Set, Optional, NamedTuple, TYPE_CHECKING

# --- 类型提示导入 ---
if TYPE_CHECKING:
    from Map import GridMap, Node # 仅用于静态类型检查
# --- 从项目模块导入 ---
try:
    # 依赖 v9+ 的 DataTypes
    from DataTypes import Task, Path, TimeStep, State, DynamicObstacles, calculate_tij, Node as NodeType, CostDict
except ImportError as e:
    print(f"错误: 导入 Planner 依赖项失败 (DataTypes): {e}")
    # 定义临时的占位符类型以便静态分析
    Task = type('Task', (object,), {})
    Path = type('Path', (object,), {})
    TimeStep = int
    State = Tuple
    DynamicObstacles = Dict
    NodeType = Tuple
    CostDict = Dict

# --- 辅助数据结构 PlannerState ---
class PlannerState(NamedTuple):
    """
    用于 TWA* 优先队列的状态表示。
    存储了 A* 搜索所需的成本信息和路径回溯指针。
    """
    f_cost: float        # 预计总成本 (g_cost + h_cost)
    g_cost: float        # 从起点到当前状态的实际累积成本 (对应部分目标函数)
    h_cost: float        # 从当前节点到目标的启发式成本 (估计剩余成本)
    state: State         # 当前时空状态 (节点, 时间步) - (Node, TimeStep)
    parent: Optional['PlannerState'] # 父状态，用于回溯路径
    turn_count: int = 0 # 跟踪转弯次数

    def __lt__(self, other: 'PlannerState') -> bool:
        """比较函数，用于优先队列排序 (f_cost 优先，h_cost 次之)。"""
        if abs(self.f_cost - other.f_cost) < 1e-9:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost

    def __eq__(self, other: object) -> bool:
        """判断两个 PlannerState 是否相等 (基于时空状态)。"""
        if not isinstance(other, PlannerState):
            return NotImplemented
        return self.state == other.state

    def __hash__(self) -> int:
        """计算哈希值，使其可以在集合(closed_set)中使用。"""
        return hash(self.state)

# --- TWA* Planner 类 ---
class TWAStarPlanner:
    """
    带时间窗的 A* 规划器 (Time-Window A*)。
    作为 ALNS 的核心路径查找引擎 (论文 3.5.1)，在栅格地图上寻找考虑
    动态障碍物、可选区域限制以及简单边冲突避免启发式的最低成本时空路径。
    """

    def __init__(self):
        """初始化规划器。"""
        # 定义简单的边冲突启发式惩罚值 (可以根据需要调整或设为0禁用)
        # 这个值应该与等待成本有一定关系，鼓励等待而不是潜在冲突
        self.edge_conflict_penalty_factor = 0.5 # 例如，相当于半步等待成本

    # --- 启发式函数 _heuristic ---
    def _heuristic(self, node: 'NodeType', goal: 'NodeType', alpha: float, v: float, delta_step: float) -> float:
        """
        计算从当前节点到目标节点的启发式成本 (h_cost)。
        使用欧几里得距离估计最小行驶时间成本。
        """
        dx = abs(node[0] - goal[0]); dy = abs(node[1] - goal[1])
        distance = math.sqrt(dx**2 + dy**2)
        if distance < 1e-9: return 0.0
        if v <= 0 or delta_step <= 0: return float('inf')
        time_exact = distance / v
        min_time_steps = math.ceil(time_exact / delta_step)
        min_time_steps = max(1.0, min_time_steps) # 至少需要一步
        h_cost = alpha * min_time_steps
        return h_cost

    # --- 转弯成本计算 _calculate_turn_cost ---
    def _calculate_turn_cost(self, parent_node: Optional['NodeType'], current_node: 'NodeType', next_node: 'NodeType', beta: float) -> float:
        """
        计算从 parent->current 到 current->next 的移动是否发生转弯。
        """
        # 无效输入或原地移动不产生转弯成本
        if parent_node is None or parent_node == current_node or current_node == next_node:
            return 0.0
        # 180 度掉头也不算（虽然通常不应该发生）
        if parent_node == next_node:
            return 0.0

        dx1 = current_node[0] - parent_node[0]; dy1 = current_node[1] - parent_node[1]
        dx2 = next_node[0] - current_node[0]; dy2 = next_node[1] - current_node[1]
        move1_len_sq = dx1**2 + dy1**2; move2_len_sq = dx2**2 + dy2**2
        # 检查移动向量是否有效
        if move1_len_sq < 1e-9 or move2_len_sq < 1e-9:
            return 0.0

        # 使用叉积判断是否共线（叉积接近0表示共线）
        cross_product = dx1 * dy2 - dx2 * dy1
        epsilon = 1e-9 # 容忍浮点数误差
        if abs(cross_product) > epsilon:
            # 发生转弯
            return beta
        else:
            # 共线，未转弯
            return 0.0

    # --- 路径回溯 _reconstruct_path ---
    def _reconstruct_path(self, goal_state_info: PlannerState, agv_id: int) -> 'Path':
        """
        从目标状态通过 parent 指针回溯生成最终路径。
        """
        sequence: List[State] = []
        current: Optional[PlannerState] = goal_state_info
        while current is not None:
            sequence.append(current.state)
            current = current.parent
        sequence.reverse() # 从起点到终点
        # 确保路径非空 (理论上不会发生，因为 goal_state_info 存在)
        if not sequence:
            # 此处应记录错误或抛出异常，因为找到目标状态却无法回溯路径是内部逻辑错误
            print(f"严重错误 (Planner): 找到目标状态 {goal_state_info} 但无法回溯路径！")
            # 返回一个空路径或根据策略处理
            return Path(agv_id, []) # 或者抛出异常
        return Path(agv_id, sequence)

    # --- 核心规划方法 ---
    def plan(self,
             grid_map: 'GridMap',
             task: Task,
             dynamic_obstacles: DynamicObstacles,
             max_time: TimeStep,
             cost_weights: Tuple[float, float, float],
             v: float,
             delta_step: float,
             start_time: TimeStep = 0,
             time_limit: Optional[float] = None,
             bounding_box: Optional[Tuple[int, int, int, int]] = None
             ) -> Optional['Path']:
        """
        执行 TWA* 规划 (论文 3.5.1)，支持区域分割 (论文 3.5.2) 和简单边冲突启发式。
        寻找从 task.start_node 到 task.goal_node 的时空路径，
        避开 static_obstacles 和 dynamic_obstacles，并最小化综合成本。

        Args:
            grid_map: 地图对象。
            task: 当前 AGV 的任务。
            dynamic_obstacles: 其他 AGV 占用的时空点 {time: set(nodes)}。
                                **注意:** 此数据必须正确包含终点占用信息。
            max_time: 最大规划时间步 T_max。
            cost_weights: 成本权重 (alpha, beta, gamma_wait)。
            v: AGV 速度。
            delta_step: 时间步长。
            start_time: AGV 开始规划的起始时间步。
            time_limit: 单次规划的最大允许时间 (秒)。
            bounding_box (Optional): 区域分割包围盒 (min_x, max_x, min_y, max_y)。

        Returns:
            Path 对象如果找到路径，否则 None。
        """
        # --- 1. 参数提取与初始化 ---
        agv_id = task.agv_id
        start_node: NodeType = task.start_node
        goal_node: NodeType = task.goal_node
        alpha, beta, gamma_wait = cost_weights
        edge_conflict_penalty = gamma_wait * self.edge_conflict_penalty_factor # 边冲突罚金

        # --- 2. 起点终点有效性检查 ---
        try:
            if not grid_map.is_valid(*start_node):
                print(f"错误 (Planner): AGV {agv_id} 起点 {start_node} 无效。")
                return None
            if bounding_box is not None:
                 min_x, max_x, min_y, max_y = bounding_box
                 if not (min_x <= start_node[0] <= max_x and min_y <= start_node[1] <= max_y):
                      print(f"错误 (Planner): AGV {agv_id} 起点 {start_node} 不在指定的包围盒 {bounding_box} 内。")
                      return None
            if not grid_map.is_valid(*goal_node):
                print(f"错误 (Planner): AGV {agv_id} 终点 {goal_node} 无效。")
                return None
            if bounding_box is not None:
                 min_x, max_x, min_y, max_y = bounding_box
                 if not (min_x <= goal_node[0] <= max_x and min_y <= goal_node[1] <= max_y):
                      print(f"错误 (Planner): AGV {agv_id} 终点 {goal_node} 不在指定的包围盒 {bounding_box} 内。")
                      return None
        except AttributeError as ae:
            # is_valid 方法可能不存在
            print(f"错误 (Planner): GridMap 对象缺少必要方法 (如 is_valid): {ae}")
            return None
        except Exception as e:
            print(f"错误 (Planner): 检查起点/终点时出错: {e}")
            return None

        # --- 3. A* 初始化 ---
        open_set: List[PlannerState] = [] # 优先队列（最小堆）
        closed_set: Dict[State, float] = {} # 存储已访问状态及其最小 g_cost {state: g_cost}
        start_plan_time = time.perf_counter() # 记录规划开始时间

        # 计算初始状态的启发式成本
        initial_h_cost = self._heuristic(start_node, goal_node, alpha, v, delta_step)
        if initial_h_cost == float('inf'):
             print(f"错误 (Planner): AGV {agv_id} 无法计算从 {start_node} 到 {goal_node} 的启发式成本。")
             return None

        initial_state: State = (start_node, start_time)
        initial_planner_state = PlannerState(
            f_cost=initial_h_cost, g_cost=0.0, h_cost=initial_h_cost,
            state=initial_state, parent=None, turn_count=0
        )

        # 将初始状态加入 Open Set 和 Closed Set
        heapq.heappush(open_set, initial_planner_state)
        closed_set[initial_state] = 0.0

        # --- 4. A* 主循环 ---
        while open_set:
            # 4.1 检查超时
            if time_limit is not None:
                elapsed_time = time.perf_counter() - start_plan_time
                if elapsed_time > time_limit:
                    # print(f"信息 (Planner): AGV {agv_id} 规划超时 ({elapsed_time:.2f}s > {time_limit}s)。")
                    return None # 超时，返回失败

            # 4.2 获取当前最优状态 (f_cost 最小)
            try:
                current_planner_state: PlannerState = heapq.heappop(open_set)
            except IndexError:
                # Open Set 为空，理论上在循环条件处已处理，但作为保险
                break

            current_g_cost = current_planner_state.g_cost
            current_state = current_planner_state.state
            current_node, current_time = current_state

            # 优化：如果 Closed Set 中已有更优路径到达此状态，则跳过
            # （使用浮点数比较容差）
            if current_state in closed_set and current_g_cost > closed_set[current_state] + 1e-9:
                continue

            # 4.3 检查是否到达目标
            if current_node == goal_node:
                # 找到路径，回溯并返回
                return self._reconstruct_path(current_planner_state, agv_id)

            # 4.4 扩展邻居节点
            parent_node: Optional[NodeType] = current_planner_state.parent.state[0] if current_planner_state.parent else None
            try:
                # 获取当前节点的有效邻居（包括自身以允许等待）
                neighbors = grid_map.get_neighbors(current_node)
                possible_next_nodes = neighbors + [current_node] # 添加原地等待选项
            except AttributeError as ae:
                 print(f"错误 (Planner): GridMap 对象缺少 get_neighbors 方法: {ae}")
                 return None # 无法继续规划
            except Exception as e:
                 print(f"错误 (Planner): 获取邻居 {current_node} 时出错: {e}")
                 continue # 跳过当前状态的扩展

            # --- 遍历可能的下一节点 ---
            for next_node in possible_next_nodes:

                # 4.4.1 **应用包围盒限制 (区域分割)**
                if bounding_box is not None:
                    min_x_bbox, max_x_bbox, min_y_bbox, max_y_bbox = bounding_box
                    nx, ny = next_node # 提取下一节点的坐标
                    if not (min_x_bbox <= nx <= max_x_bbox and min_y_bbox <= ny <= max_y_bbox):
                        continue # 跳过不在包围盒内的节点

                # 4.4.2 计算移动时间 t_ij
                is_move = (next_node != current_node)
                try:
                    ideal_time_steps = calculate_tij(current_node, next_node, v, delta_step, grid_map) if is_move else 1
                except Exception as e:
                    # print(f"警告 (Planner): 计算 tij({current_node}, {next_node}) 时出错: {e}")
                    continue # 跳过无法计算时间的移动
                # 确保时间步数有效
                if ideal_time_steps == float('inf') or ideal_time_steps <= 0:
                    continue

                # 4.4.3 计算下一状态时间戳并检查 T_max
                next_time = current_time + ideal_time_steps
                if next_time > max_time:
                    continue # 超过最大时间范围，跳过

                # 4.4.4 **检查时空冲突 (节点占用)**
                collision = False
                # 检查从 current_time+1 到 next_time 的每个时间步
                for t_check in range(current_time + 1, next_time + 1):
                    # 检查 dynamic_obstacles 中该时间步是否有占用
                    # 并且占用的节点是否是 next_node
                    obstacles_at_t = dynamic_obstacles.get(t_check)
                    if obstacles_at_t and next_node in obstacles_at_t:
                        collision = True
                        break # 发现冲突，无需继续检查
                if collision:
                    continue # 存在节点冲突，跳过此移动

                # 4.4.5 **简单边冲突启发式检查**
                edge_conflict_cost_penalty = 0.0
                if is_move: # 只对移动应用边冲突惩罚
                    # 检查下一节点在紧邻的下一个时间步 (current_time + 1) 是否被占用
                    # 这是一个非常简化的代理，用于捕捉潜在的交叉或迎面风险
                    # 注意：next_time 可能是 current_time + 1 或更大
                    check_time_for_edge = current_time + 1
                    if check_time_for_edge <= max_time: # 确保检查时间有效
                        obstacles_at_next_step = dynamic_obstacles.get(check_time_for_edge)
                        if obstacles_at_next_step and next_node in obstacles_at_next_step:
                            # 如果被占用，施加惩罚
                            edge_conflict_cost_penalty = edge_conflict_penalty

                # 4.4.6 计算成本增量
                cost_increment = 0.0
                new_turn_count = current_planner_state.turn_count
                if is_move:
                    # 行驶成本
                    cost_increment += alpha * float(ideal_time_steps)
                    # 转弯成本
                    turn_cost = self._calculate_turn_cost(parent_node, current_node, next_node, beta)
                    cost_increment += turn_cost
                    if turn_cost > 1e-9: new_turn_count += 1
                    # 加上边冲突启发式惩罚
                    cost_increment += edge_conflict_cost_penalty
                else: # 等待
                    cost_increment += gamma_wait * float(ideal_time_steps) # 等待成本

                # 4.4.7 计算新 g_cost
                new_g_cost = current_g_cost + cost_increment

                # 4.4.8 检查 Closed Set
                next_state: State = (next_node, next_time)
                # 如果 Closed Set 中存在且成本不更优，则跳过
                # （使用浮点数比较容差）
                if next_state in closed_set and new_g_cost >= closed_set[next_state] - 1e-9:
                    continue

                # 4.4.9 计算新 h_cost
                new_h_cost = self._heuristic(next_node, goal_node, alpha, v, delta_step)
                if new_h_cost == float('inf'):
                    # 如果无法到达目标，跳过此路径
                    continue

                # 4.4.10 计算新 f_cost
                new_f_cost = new_g_cost + new_h_cost

                # 4.4.11 创建新状态并加入 Open/Closed Set
                new_planner_state = PlannerState(
                    f_cost=new_f_cost, g_cost=new_g_cost, h_cost=new_h_cost,
                    state=next_state, parent=current_planner_state, turn_count=new_turn_count
                )
                heapq.heappush(open_set, new_planner_state)
                closed_set[next_state] = new_g_cost # 更新或添加状态到 Closed Set
            # --- 遍历下一节点结束 ---
        # --- A* 主循环结束 ---

        # --- 5. Open Set 为空，未找到路径 ---
        # print(f"信息 (Planner): AGV {agv_id} 未找到路径 (Open Set 为空)。")
        return None

# --- 示例用法 ---
if __name__ == '__main__':
    print("--- TWA* Planner 测试 (v17 - Edge Conflict Heuristic + Strict Structure) ---")
    # --- 标准导入 (修复 AgentPath 错误) ---
    try:
        from InstanceGenerator import load_fixed_scenario_1
        from Map import GridMap
        # 正确导入 Path，如果要在示例中用 AgentPath 可以用 'as'
        from DataTypes import Solution, DynamicObstacles, Task, Path as AgentPath
    except ImportError as import_error:
        print(f"错误: 导入必需模块失败: {import_error}")
        exit(1)

    # --- 准备测试环境 ---
    print("加载固定算例场景 1...")
    instance_data_s1 = load_fixed_scenario_1()
    if not instance_data_s1:
        print("错误: 无法加载场景 1 数据。")
        exit(1)
    test_map_s1, test_tasks_s1 = instance_data_s1
    # 确保场景 1 至少有一个任务
    if not test_tasks_s1:
        print("错误: 场景 1 任务列表为空。")
        exit(1)
    test_task_agv0 = next((t for t in test_tasks_s1 if t.agv_id == 0), None)
    if not test_task_agv0:
        print("警告: 场景 1 未找到 AGV 0 的任务，使用第一个可用任务。")
        test_task_agv0 = test_tasks_s1[0]

    empty_dyn_obs: DynamicObstacles = {}
    print(f"\n测试 AGV {test_task_agv0.agv_id} 规划 (使用 Planner v17):")
    print(f"  地图: {test_map_s1}")
    print(f"  任务: {test_task_agv0}")

    # --- 执行规划 ---
    planner_v17 = TWAStarPlanner() # 使用待测试的 v17
    cost_w = (1.0, 0.3, 0.8) # alpha, beta, gamma_wait
    agv_v = 1.0
    step_t = 1.0
    max_t_horizon = 400
    time_lim = 30.0

    print(f"\n测试 1: 不使用包围盒 (bounding_box=None)")
    start_t1 = time.perf_counter()
    result_path1: Optional[AgentPath] = planner_v17.plan(
        grid_map=test_map_s1, task=test_task_agv0, dynamic_obstacles=empty_dyn_obs,
        max_time=max_t_horizon, cost_weights=cost_w, v=agv_v, delta_step=step_t,
        start_time=0, time_limit=time_lim,
        bounding_box=None # 明确传入 None
    )
    end_t1 = time.perf_counter()
    duration1 = end_t1 - start_t1

    # 使用严格的返回结构检查
    if result_path1 is not None:
        print(f"  规划成功！耗时: {duration1:.4f}s, Makespan: {result_path1.get_makespan()}")
        # 计算并打印成本
        cost_dict1 = result_path1.get_cost(test_map_s1, *cost_w, agv_v, step_t)
        print(f"  成本: Total={cost_dict1.get('total',-1):.2f}, "
              f"Travel={cost_dict1.get('travel',-1):.2f}, "
              f"Turn={cost_dict1.get('turn',-1):.2f}, "
              f"Wait={cost_dict1.get('wait',-1):.2f}")
    else:
        print(f"  规划失败。耗时: {duration1:.4f}s")

    # --- 测试包围盒 ---
    s_node = test_task_agv0.start_node
    g_node = test_task_agv0.goal_node
    buffer = 1 # 包围盒扩展量
    try:
        min_x_test = max(0, min(s_node[0], g_node[0]) - buffer)
        max_x_test = min(test_map_s1.width - 1, max(s_node[0], g_node[0]) + buffer)
        min_y_test = max(0, min(s_node[1], g_node[1]) - buffer)
        max_y_test = min(test_map_s1.height - 1, max(s_node[1], g_node[1]) + buffer)
        test_bbox = (min_x_test, max_x_test, min_y_test, max_y_test)

        print(f"\n测试 2: 使用包围盒 {test_bbox}")
        start_t2 = time.perf_counter()
        result_path2: Optional[AgentPath] = planner_v17.plan(
            grid_map=test_map_s1, task=test_task_agv0, dynamic_obstacles=empty_dyn_obs,
            max_time=max_t_horizon, cost_weights=cost_w, v=agv_v, delta_step=step_t,
            start_time=0, time_limit=time_lim,
            bounding_box=test_bbox # 传入计算好的包围盒
        )
        end_t2 = time.perf_counter()
        duration2 = end_t2 - start_t2

        if result_path2 is not None:
            print(f"  规划成功！耗时: {duration2:.4f}s, Makespan: {result_path2.get_makespan()}")
            cost_dict2 = result_path2.get_cost(test_map_s1, *cost_w, agv_v, step_t)
            print(f"  成本: Total={cost_dict2.get('total',-1):.2f}")
        else:
            print(f"  规划失败。耗时: {duration2:.4f}s")

    except AttributeError as map_err:
        print(f"错误: 测试包围盒时无法访问地图属性: {map_err}")
    except Exception as bbox_err:
        print(f"错误: 测试包围盒时发生异常: {bbox_err}")

    # --- 测试边冲突启发式 (构造一个简单场景) ---
    print("\n测试 3: 简单边冲突启发式")
    # 假设 AGV 1 从 (1,0) -> (1,1)，时间是 [0, 1)
    # AGV 0 从 (0,1) -> (1,1)，也在时间 [0, 1) (需要 t_ij=1)
    # 预期 AGV 0 在检查移动到 (1,1) 时，发现 (1,1) 在 t=1 被占用，会增加惩罚
    test_task_agv0_edge = Task(agv_id=0, start_node=(0,1), goal_node=(2,1))
    # 模拟 AGV 1 占用 (1,1) 在 t=1
    edge_dyn_obs: DynamicObstacles = {1: {(1, 1)}}
    print(f"  任务: {test_task_agv0_edge}")
    print(f"  动态障碍: {edge_dyn_obs}")

    start_t_edge = time.perf_counter()
    result_path_edge: Optional[AgentPath] = planner_v17.plan(
        grid_map=test_map_s1, task=test_task_agv0_edge, dynamic_obstacles=edge_dyn_obs,
        max_time=max_t_horizon, cost_weights=cost_w, v=agv_v, delta_step=step_t,
        start_time=0, time_limit=time_lim,
        bounding_box=None
    )
    end_t_edge = time.perf_counter()
    duration_edge = end_t_edge - start_t_edge

    if result_path_edge is not None:
        print(f"  规划成功！耗时: {duration_edge:.4f}s, Makespan: {result_path_edge.get_makespan()}")
        cost_dict_edge = result_path_edge.get_cost(test_map_s1, *cost_w, agv_v, step_t)
        print(f"  成本: Total={cost_dict_edge.get('total',-1):.2f}, "
              f"Travel={cost_dict_edge.get('travel',-1):.2f}, "
              f"Turn={cost_dict_edge.get('turn',-1):.2f}, "
              f"Wait={cost_dict_edge.get('wait',-1):.2f}")
        # 分析路径是否因惩罚而绕路或等待
        if len(result_path_edge.sequence) > 2 and result_path_edge.sequence[1] == ((0,1), 1): # 检查是否原地等待了
             print("  观察: AGV 0 可能在起点等待以避免潜在边冲突。")
        elif len(result_path_edge.sequence) > 1 and result_path_edge.sequence[1] == ((0,0), 1): # 检查是否绕路了
             print("  观察: AGV 0 可能绕路以避免潜在边冲突。")

    else:
        print(f"  规划失败。耗时: {duration_edge:.4f}s")
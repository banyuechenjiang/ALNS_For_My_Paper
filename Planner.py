# Planner.py-v15
"""
实现带时间窗的 A* (TWA*) 路径规划器。
该规划器是 ALNS 算法内部的核心引擎，负责为单个 AGV 在考虑时空约束
（动态障碍物）和可选的搜索区域限制（包围盒）的情况下规划从起点到终点的
成本最优路径。

与论文的关联:
- 核心规划器 (论文 3.5.1): 本模块实现了论文第 3.5.1 节定义的 TWA* 核心规划器。
- 目标函数评估 (论文 公式 1): TWA* 搜索过程中计算的 g_cost 累积了行驶成本 (α * t_ij)
  和转弯成本 (β * θ_ijl)，直接关联目标函数的前两项。等待成本 (γ_wait * w_ikt)
  通过允许原地等待动作隐式考虑。
- 约束满足:
    - 状态转移 (论文 约束 6, 7, 8): 通过 A* 的节点扩展逻辑实现。
    - 节点冲突 (论文 约束 12): 通过检查 dynamic_obstacles 实现。
    - 障碍物 (论文 约束 13): 通过 grid_map.is_valid 和 get_neighbors 方法处理。
- **区域分割 (论文 3.5.2):** 本版本 **实现** 了区域分割功能。
  调用者可以通过传递可选的 `bounding_box` 参数来限制 TWA* 的搜索范围，
  从而加速规划，对应论文 3.5.2 节描述的优化策略。

版本变更 (v14 -> v15):
- **新增:** 在 `plan` 方法中添加了可选的 `bounding_box` 参数，用于实现区域分割优化 (论文 3.5.2)。
- **新增:** 在节点扩展逻辑中加入了检查节点是否在 `bounding_box` 内的判断。
- 更新了模块和 `plan` 方法的文档字符串以反映区域分割的实现。
- 保持了 v14 的其他文档和功能。
"""
import heapq
import time
import math
from typing import List, Tuple, Dict, Set, Optional, NamedTuple, TYPE_CHECKING

# --- 使用 TYPE_CHECKING 避免运行时导入 ---
if TYPE_CHECKING:
    from Map import GridMap, Node # 仅用于静态类型检查
# --- 直接从项目模块导入 ---
# 依赖 v9 的 DataTypes
try:
    from DataTypes import Task, Path as AgentPath, TimeStep, State, DynamicObstacles, calculate_tij, Node as NodeType
except ImportError as e:
    print(f"错误: 导入 Planner 依赖项失败 (DataTypes): {e}")
    # 定义临时的占位符类型以便静态分析
    Task = type('Task', (object,), {})
    AgentPath = type('AgentPath', (object,), {})
    TimeStep = int
    State = Tuple
    DynamicObstacles = Dict
    NodeType = Tuple

# --- 辅助数据结构 PlannerState (保持 v14) ---
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
    turn_count: int = 0 # 跟踪转弯次数 (用于Tie-breaking或未来扩展)

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

# --- TWA* Planner 类 (v15 - 实现区域分割) ---
class TWAStarPlanner:
    """
    带时间窗的 A* 规划器 (Time-Window A*)。
    作为 ALNS 的核心路径查找引擎 (论文 3.5.1)，在栅格地图上寻找考虑
    动态障碍物和可选区域限制的最低成本时空路径。
    """

    def __init__(self):
        """初始化规划器。"""
        pass # 当前版本无需特定初始化

    # --- 启发式函数 _heuristic (保持 v14) ---
    def _heuristic(self, node: 'NodeType', goal: 'NodeType', alpha: float, v: float, delta_step: float) -> float:
        """
        计算从当前节点到目标节点的启发式成本 (h_cost)。
        使用欧几里得距离估计最小行驶时间成本 (忽略障碍物和转弯)，
        以指导 A* 搜索。乘以 alpha 保持与 g_cost 的成本尺度一致，
        确保启发式函数的可接受性 (admissible)。
        """
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        distance = math.sqrt(dx**2 + dy**2)
        if distance < 1e-9: return 0.0
        if v <= 0 or delta_step <= 0: return float('inf')
        time_exact = distance / v
        min_time_steps = math.ceil(time_exact / delta_step)
        min_time_steps = max(1.0, min_time_steps)
        h_cost = alpha * min_time_steps
        return h_cost

    # --- 转弯成本计算 _calculate_turn_cost (保持 v14) ---
    def _calculate_turn_cost(self, parent_node: Optional['NodeType'], current_node: 'NodeType', next_node: 'NodeType', beta: float) -> float:
        """
        计算从 parent->current 到 current->next 的移动是否发生转弯。
        用于累加 g_cost 中的转弯成本部分 (对应目标函数 β * θ_ijl 项)。
        """
        if parent_node is None or parent_node == current_node or current_node == next_node: return 0.0
        if parent_node == next_node: return 0.0

        dx1 = current_node[0] - parent_node[0]; dy1 = current_node[1] - parent_node[1]
        dx2 = next_node[0] - current_node[0]; dy2 = next_node[1] - current_node[1]
        move1_len_sq = dx1**2 + dy1**2
        move2_len_sq = dx2**2 + dy2**2
        if move1_len_sq < 1e-9 or move2_len_sq < 1e-9: return 0.0

        cross_product = dx1 * dy2 - dx2 * dy1
        epsilon = 1e-9
        if abs(cross_product) > epsilon:
            return beta
        else:
            return 0.0

    # --- 路径回溯 _reconstruct_path (保持 v14) ---
    def _reconstruct_path(self, goal_state_info: PlannerState, agv_id: int) -> AgentPath:
        """
        从目标状态通过 parent 指针回溯生成最终路径。
        构建符合论文 3.2 节定义的 Path 对象。
        """
        sequence: List[State] = []
        current: Optional[PlannerState] = goal_state_info
        while current is not None:
            sequence.append(current.state)
            current = current.parent
        sequence.reverse()
        return AgentPath(agv_id, sequence)

    # --- 核心规划方法 (v15 - 添加 bounding_box 参数和逻辑) ---
    def plan(self,
             grid_map: 'GridMap', task: Task, dynamic_obstacles: DynamicObstacles,
             max_time: TimeStep, cost_weights: Tuple[float, float, float],
             v: float, delta_step: float,
             # buffer: int = 0, # buffer 参数在外部计算 bbox 时使用，不再直接传入 TWA*
             start_time: TimeStep = 0,
             time_limit: Optional[float] = None,
             bounding_box: Optional[Tuple[int, int, int, int]] = None # 新增: 包围盒 (min_x, max_x, min_y, max_y)
             ) -> Optional[AgentPath]:
        """
        执行 TWA* 规划 (论文 3.5.1)，支持区域分割 (论文 3.5.2)。
        寻找从 task.start_node 到 task.goal_node 的时空路径，
        避开 static_obstacles (通过 grid_map) 和 dynamic_obstacles，
        并最小化由 cost_weights 定义的综合成本 (对应论文目标函数 1)。
        如果提供了 bounding_box，则搜索范围将被限制在该区域内。

        Args:
            grid_map: 地图对象 (含 is_valid, get_neighbors)。
            task: 当前 AGV 的任务 (含 agv_id, start_node, goal_node)。
            dynamic_obstacles: 其他 AGV 占用的时空点 {time: set(nodes)}。
            max_time: 最大规划时间步 T_max。
            cost_weights: 成本权重 (alpha, beta, gamma_wait)。
            v: AGV 速度。
            delta_step: 时间步长。
            start_time: AGV 开始规划的起始时间步。
            time_limit: 单次规划的最大允许时间 (秒)。
            bounding_box (Optional): 用于区域分割的包围盒 (min_x, max_x, min_y, max_y)。
                                     如果为 None，则不限制搜索区域。

        Returns:
            AgentPath 对象如果找到路径，否则 None。
        """
        # --- 参数提取与初始化 (保持 v14) ---
        agv_id = task.agv_id
        start_node: NodeType = task.start_node
        goal_node: NodeType = task.goal_node
        alpha, beta, gamma_wait = cost_weights

        # --- 起点终点有效性检查 (保持 v14) ---
        try:
            if not grid_map.is_valid(*start_node): print(f"错误 (Planner): AGV {agv_id} 起点 {start_node} 无效。"); return None
            # **新增**: 检查起点是否在包围盒内 (如果提供了 bbox)
            if bounding_box is not None:
                 min_x, max_x, min_y, max_y = bounding_box
                 if not (min_x <= start_node[0] <= max_x and min_y <= start_node[1] <= max_y):
                      print(f"错误 (Planner): AGV {agv_id} 起点 {start_node} 不在指定的包围盒 {bounding_box} 内。")
                      return None
            if not grid_map.is_valid(*goal_node): print(f"错误 (Planner): AGV {agv_id} 终点 {goal_node} 无效。"); return None
            # **新增**: 检查终点是否在包围盒内 (如果提供了 bbox)
            if bounding_box is not None:
                 min_x, max_x, min_y, max_y = bounding_box
                 if not (min_x <= goal_node[0] <= max_x and min_y <= goal_node[1] <= max_y):
                      print(f"错误 (Planner): AGV {agv_id} 终点 {goal_node} 不在指定的包围盒 {bounding_box} 内。")
                      return None
        except Exception as e: print(f"错误 (Planner): 检查起点/终点时出错: {e}"); return None

        # --- A* 初始化 (保持 v14) ---
        open_set: List[PlannerState] = []
        closed_set: Dict[State, float] = {}
        initial_h_cost = self._heuristic(start_node, goal_node, alpha, v, delta_step)
        initial_state: State = (start_node, start_time)
        initial_planner_state = PlannerState(f_cost=initial_h_cost, g_cost=0.0, h_cost=initial_h_cost, state=initial_state, parent=None, turn_count=0)
        heapq.heappush(open_set, initial_planner_state)
        closed_set[initial_state] = 0.0
        start_plan_time = time.perf_counter()

        # --- A* 主循环 ---
        while open_set:
            # 1. 检查超时 (保持 v14)
            if time_limit is not None and (time.perf_counter() - start_plan_time) > time_limit: return None

            # 2. 获取当前状态 (保持 v14)
            current_planner_state: PlannerState = heapq.heappop(open_set)
            current_g_cost = current_planner_state.g_cost
            current_state = current_planner_state.state
            current_node, current_time = current_state
            if current_state in closed_set and current_g_cost > closed_set[current_state] + 1e-9: continue

            # 3. 检查是否到达目标 (保持 v14)
            if current_node == goal_node: return self._reconstruct_path(current_planner_state, agv_id)

            # 4. 扩展邻居节点 (保持 v14)
            parent_node: Optional[NodeType] = current_planner_state.parent.state[0] if current_planner_state.parent else None
            try: neighbors = grid_map.get_neighbors(current_node)
            except Exception as e: print(f"错误 (Planner): 获取邻居 {current_node} 出错: {e}"); return None
            possible_next_nodes = neighbors + [current_node] # 包含原地等待

            # --- 遍历可能的下一节点 (核心循环) ---
            for next_node in possible_next_nodes:

                # =====================================================
                # --- v15 新增: 应用包围盒限制 (区域分割) ---
                # =====================================================
                if bounding_box is not None:
                    min_x_bbox, max_x_bbox, min_y_bbox, max_y_bbox = bounding_box
                    nx, ny = next_node # 提取下一节点的坐标
                    # 如果下一节点不在包围盒内，则跳过，不进行后续处理
                    if not (min_x_bbox <= nx <= max_x_bbox and min_y_bbox <= ny <= max_y_bbox):
                        continue # 跳过不在包围盒内的节点
                # =====================================================

                # 4.1 计算理论时间步 t_ij (保持 v14)
                is_move = (next_node != current_node)
                try: ideal_time_steps = calculate_tij(current_node, next_node, v, delta_step, grid_map) if is_move else 1
                except Exception as e: print(f"错误 (Planner): 计算 tij({current_node}, {next_node}) 时出错: {e}"); continue
                if ideal_time_steps == float('inf'): continue # 跳过无效移动

                # 4.2 计算下一状态时间戳 (保持 v14)
                next_time = current_time + ideal_time_steps
                if next_time > max_time: continue # 检查是否超限

                # 4.3 检查时空冲突 (保持 v14)
                collision = False
                for t_check in range(current_time + 1, next_time + 1):
                    if t_check in dynamic_obstacles and next_node in dynamic_obstacles[t_check]: collision = True; break
                if collision: continue

                # 4.4 计算成本增量 (保持 v14)
                cost_increment = 0.0
                new_turn_count = current_planner_state.turn_count
                if is_move:
                    cost_increment += alpha * float(ideal_time_steps)
                    turn_cost = self._calculate_turn_cost(parent_node, current_node, next_node, beta)
                    cost_increment += turn_cost
                    if turn_cost > 1e-9: new_turn_count += 1
                else: cost_increment += gamma_wait * float(ideal_time_steps)

                # 4.5 计算新 g_cost (保持 v14)
                new_g_cost = current_g_cost + cost_increment
                next_state: State = (next_node, next_time)

                # 4.6 检查 Closed Set (保持 v14)
                if next_state in closed_set and new_g_cost >= closed_set[next_state] - 1e-9: continue

                # 4.7 计算新 h_cost (保持 v14)
                new_h_cost = self._heuristic(next_node, goal_node, alpha, v, delta_step)
                if new_h_cost == float('inf'): continue

                # 4.8 计算新 f_cost (保持 v14)
                new_f_cost = new_g_cost + new_h_cost

                # 4.9 创建新状态并加入 Open/Closed Set (保持 v14)
                new_planner_state = PlannerState(f_cost=new_f_cost, g_cost=new_g_cost, h_cost=new_h_cost, state=next_state, parent=current_planner_state, turn_count=new_turn_count)
                heapq.heappush(open_set, new_planner_state)
                closed_set[next_state] = new_g_cost
            # --- 遍历下一节点结束 ---

        # Open Set 为空，未找到路径 (保持 v14)
        return None

# --- 示例用法 (保持 v14 逻辑，适应新签名) ---
if __name__ == '__main__':
    print("--- TWA* Planner 测试 (v15 - Implemented Bounding Box) ---")
    # --- 标准导入 ---
    try: from InstanceGenerator import load_fixed_scenario_1; from Map import GridMap; from DataTypes import Solution
    except ImportError as import_error: print(f"错误: 导入必需模块失败: {import_error}"); exit(1)

    # --- 准备测试环境 ---
    print("加载固定算例场景 1...")
    instance_data_s1 = load_fixed_scenario_1()
    if not instance_data_s1: print("错误: 无法加载场景 1 数据。"); exit(1)
    test_map_s1, test_tasks_s1 = instance_data_s1
    test_task_agv0 = next((t for t in test_tasks_s1 if t.agv_id == 0), None)
    if not test_task_agv0: print("错误: 找不到 AGV 0 的任务。"); exit(1)
    empty_dyn_obs: DynamicObstacles = {}
    print(f"\n测试 AGV 0 规划 (使用 Planner v15):")
    print(f"  地图: {test_map_s1}")
    print(f"  任务: {test_task_agv0}")

    # --- 执行规划 (不使用包围盒) ---
    planner_v15 = TWAStarPlanner() # 使用待测试的 v15
    cost_w = (1.0, 0.3, 0.8); agv_v = 1.0; step_t = 1.0; max_t_horizon = 400; time_lim = 30.0
    print("\n测试 1: 不使用包围盒 (bounding_box=None)")
    start_t1 = time.perf_counter()
    result_path1 = planner_v15.plan(
        grid_map=test_map_s1, task=test_task_agv0, dynamic_obstacles=empty_dyn_obs,
        max_time=max_t_horizon, cost_weights=cost_w, v=agv_v, delta_step=step_t,
        start_time=0, time_limit=time_lim,
        bounding_box=None # 明确传入 None
    )
    end_t1 = time.perf_counter(); duration1 = end_t1 - start_t1
    if result_path1: print(f"  规划成功！耗时: {duration1:.4f}s, Makespan: {result_path1.get_makespan()}");
    else: print(f"  规划失败。耗时: {duration1:.4f}s")

    # --- 执行规划 (使用一个合理的包围盒) ---
    # 手动计算一个合理的包围盒 (带 buffer=1)
    s_node = test_task_agv0.start_node; g_node = test_task_agv0.goal_node; buffer = 1
    min_x_test = max(0, min(s_node[0], g_node[0]) - buffer)
    max_x_test = min(test_map_s1.width - 1, max(s_node[0], g_node[0]) + buffer)
    min_y_test = max(0, min(s_node[1], g_node[1]) - buffer)
    max_y_test = min(test_map_s1.height - 1, max(s_node[1], g_node[1]) + buffer)
    test_bbox = (min_x_test, max_x_test, min_y_test, max_y_test)
    print(f"\n测试 2: 使用包围盒 {test_bbox}")
    start_t2 = time.perf_counter()
    result_path2 = planner_v15.plan(
        grid_map=test_map_s1, task=test_task_agv0, dynamic_obstacles=empty_dyn_obs,
        max_time=max_t_horizon, cost_weights=cost_w, v=agv_v, delta_step=step_t,
        start_time=0, time_limit=time_lim,
        bounding_box=test_bbox # 传入计算好的包围盒
    )
    end_t2 = time.perf_counter(); duration2 = end_t2 - start_t2
    if result_path2: print(f"  规划成功！耗时: {duration2:.4f}s, Makespan: {result_path2.get_makespan()}");
    else: print(f"  规划失败。耗时: {duration2:.4f}s")

    # --- 执行规划 (使用一个过小的包围盒，预期失败) ---
    invalid_bbox = (0, 1, 0, 1) # 假设这个盒子无法包含路径
    print(f"\n测试 3: 使用过小的包围盒 {invalid_bbox} (预期失败)")
    start_t3 = time.perf_counter()
    result_path3 = planner_v15.plan(
        grid_map=test_map_s1, task=test_task_agv0, dynamic_obstacles=empty_dyn_obs,
        max_time=max_t_horizon, cost_weights=cost_w, v=agv_v, delta_step=step_t,
        start_time=0, time_limit=time_lim,
        bounding_box=invalid_bbox
    )
    end_t3 = time.perf_counter(); duration3 = end_t3 - start_t3
    if result_path3: print(f"  规划成功！(意外) 耗时: {duration3:.4f}s, Makespan: {result_path3.get_makespan()}");
    else: print(f"  规划失败。(符合预期) 耗时: {duration3:.4f}s")
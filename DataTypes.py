# DataTypes.py-v11 (支持 AGV 特定速度)
"""
定义用于仓储 AGV 路径规划问题的核心数据结构。

与论文的关联:
- 核心概念: 定义了任务 (Task)、路径 (Path)、解决方案 (Solution) 和成本字典 (CostDict)，
           这些是构建数学模型 (Chapter 2) 和设计 ALNS 算法 (Chapter 3) 的基础。
- 解决方案表示: Path 类及其 sequence 属性直接对应论文 3.2 节描述的解决方案表示方法，
               并隐式关联数学模型中的位置状态变量 τ_ikt。
- 目标函数评估: Path.get_cost 方法用于计算单个路径的成本，其计算逻辑直接对应
                 论文的目标函数 (公式 1) 的三个组成部分（行驶、转弯、等待）。
                 **v11 修改**: 现在支持基于特定 AGV 速度计算成本。
- 节点类型 V_s, V_p: Task 类中的 start_node (s_k) 和 goal_node (e_k)
                     分别定义了论文模型中的 V_s (起始节点) 和 V_p (目标/拣选站节点)
                     集合中的元素。

版本变更 (v10 -> v11):
- **修改**: `calculate_tij` 函数现在接收 `v` (AGV 速度) 作为参数。
- **修改**: `Path.get_cost` 方法现在接收 `v` (AGV 速度) 作为参数，并将其传递给 `calculate_tij`。
- 保持: v10 的其他所有类、类型别名和功能（包括 `check_time_overlap`）保持不变。
"""
import math
from typing import List, Tuple, NamedTuple, Set, Dict, Optional, TYPE_CHECKING

# --- 使用 TYPE_CHECKING 避免运行时导入 GridMap ---
if TYPE_CHECKING:
    from Map import GridMap # 仅用于静态类型检查

# --- 核心类型别名 ---
Node = Tuple[int, int]
TimeStep = int
State = Tuple[Node, TimeStep]
DynamicObstacles = Dict[TimeStep, Set[Node]]
CostDict = Dict[str, float]

# --- 任务类 ---
class Task(NamedTuple):
    """
    表示一个 AGV 的搬运任务。
    定义了 AGV 的起点和终点，这些点分别属于论文模型中定义的
    V_s (起始节点集) 和 V_p (目标/拣选站节点集)。
    """
    agv_id: int
    start_node: Node
    goal_node: Node

    def __repr__(self) -> str:
        """返回任务的可读字符串表示。"""
        return f"Task(AGV={self.agv_id}, Start(Vs)={self.start_node}, Goal(Vp)={self.goal_node})"

# --- 路径类 ---
class Path:
    """
    表示单个 AGV 的时空路径。
    这是算法中解的核心组成部分，对应论文 3.2 节描述的解决方案表示。
    其 sequence 属性隐式地定义了数学模型中的位置状态变量 τ_ikt。
    """
    def __init__(self, agv_id: int, sequence: List[State]):
        """
        初始化路径对象。

        Args:
            agv_id: AGV 的唯一标识符。
            sequence: 路径状态序列 List[Tuple[Node, TimeStep]]。

        Raises:
            ValueError: 如果路径序列为空、格式错误或时间步不单调。
        """
        if not sequence: raise ValueError("路径序列不能为空。")
        if not all(isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], tuple) and len(s[0]) == 2 and isinstance(s[0][0], int) and isinstance(s[0][1], int) and isinstance(s[1], int) for s in sequence): raise ValueError("路径序列格式错误，应为 List[Tuple[Tuple[int, int], int]]。")
        for i in range(len(sequence) - 1):
             if sequence[i+1][1] < sequence[i][1]: raise ValueError(f"路径序列时间步必须单调不减: {sequence[i]} -> {sequence[i+1]}")
        self.agv_id = agv_id
        self.sequence: List[State] = sequence

    def get_start_node(self) -> Node:
        """获取路径的起始节点。"""
        # 假设 sequence 在 __init__ 中已验证非空
        return self.sequence[0][0]

    def get_goal_node(self) -> Node:
        """获取路径的最终节点。"""
        # 假设 sequence 在 __init__ 中已验证非空
        return self.sequence[-1][0]

    def get_makespan(self) -> TimeStep:
        """获取路径的完成时间 (最后一个状态的时间步)。"""
        # 假设 sequence 在 __init__ 中已验证非空
        return self.sequence[-1][1]

    def nodes_occupied_at_time(self, time: TimeStep) -> Set[Node]:
        """获取在指定时间步被此路径占用的节点集合。"""
        occupied_nodes = set()
        # 优化：如果路径很长，可以考虑二分查找或缓存
        for node, t in self.sequence:
            if t == time:
                occupied_nodes.add(node)
                # 假设一个时间步只在一个节点，找到即可退出内层循环
                break
            elif t > time:
                # 由于序列时间单调，后续时间更大，无需继续查找
                break
        return occupied_nodes

    # --- 修改: 接收 v (AGV 速度) ---
    def get_cost(self, grid_map: 'GridMap', alpha: float, beta: float, gamma_wait: float, v: float, delta_step: float) -> CostDict:
        """
        计算此路径的总成本及其构成 (行驶、转弯、等待)。
        **v11 修改**: 使用传入的特定 AGV 速度 v。

        Args:
            grid_map: GridMap 对象。
            alpha: 行驶时间成本权重。
            beta: 转弯成本权重。
            gamma_wait: 等待时间成本权重。
            v: 此 AGV 的速度 (格/秒)。
            delta_step: 每个时间步的时长 (秒)。

        Returns:
            一个包含 'total', 'travel', 'turn', 'wait' 成本的字典。
            如果路径无效或计算出错，返回包含 inf 的字典。

        Raises:
            TypeError: 如果 grid_map 缺少必要方法。
            ValueError: 如果 v 或 delta_step 无效。
        """
        if not hasattr(grid_map, 'get_move_cost'): raise TypeError("grid_map 参数必须提供 get_move_cost 方法。")
        # --- 使用传入的 v 和 delta_step ---
        if v <= 0 or delta_step <= 0: raise ValueError("速度 v 和时间步长 delta_step 必须为正。")
        # --- ------------------------- ---
        if not self.sequence or len(self.sequence) < 1: return {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}

        total_travel_steps = 0.0
        total_turn_count = 0.0
        total_wait_steps = 0.0
        last_move_direction: Optional[Tuple[int, int]] = None

        for i in range(len(self.sequence) - 1):
            node_i, time_i = self.sequence[i]
            node_j, time_j = self.sequence[i+1]
            time_diff = time_j - time_i

            # 基本检查
            if time_diff < 0:
                print(f"错误 (PathCost): AGV {self.agv_id} 路径时间步减少: {time_i} -> {time_j}")
                return {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}

            if node_i == node_j: # 等待
                # 只有当时间确实增加了才算等待成本
                if time_diff > 0:
                    total_wait_steps += float(time_diff)
            else: # 移动
                # --- 调用修改后的 calculate_tij，传入 v ---
                ideal_time_steps = calculate_tij(node_i, node_j, v, delta_step, grid_map)
                # --- ----------------------------------- ---
                if ideal_time_steps == float('inf'):
                    print(f"错误 (PathCost): AGV {self.agv_id} 移动 {node_i}->{node_j} 的 ideal_time_steps 为 inf。")
                    return {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}

                total_travel_steps += float(ideal_time_steps) # 累加理论行驶时间步

                # 计算实际移动花费的时间步
                actual_move_steps = time_diff
                # 如果实际花费时间 > 理论时间，差值计入等待成本
                # 使用小的容差避免浮点误差
                if actual_move_steps > ideal_time_steps + 1e-6:
                    total_wait_steps += float(actual_move_steps - ideal_time_steps)

                # 计算转弯成本
                current_move_direction = (node_j[0] - node_i[0], node_j[1] - node_i[1])
                if last_move_direction is not None:
                    dx1, dy1 = last_move_direction
                    dx2, dy2 = current_move_direction
                    # 确保两个方向向量都不是零向量 (例如，在等待之后移动)
                    if (dx1**2 + dy1**2 > 1e-9) and (dx2**2 + dy2**2 > 1e-9):
                        # 使用叉积判断是否共线（叉积为零表示共线）
                        cross_product = dx1 * dy2 - dx2 * dy1
                        if abs(cross_product) > 1e-9: # 不共线，发生转弯
                            total_turn_count += 1.0
                last_move_direction = current_move_direction # 更新上一步的移动方向

        # 计算最终加权成本
        travel_cost = alpha * total_travel_steps
        turn_cost = beta * total_turn_count
        wait_cost = gamma_wait * total_wait_steps
        total_cost = travel_cost + turn_cost + wait_cost

        return {'total': total_cost, 'travel': travel_cost, 'turn': turn_cost, 'wait': wait_cost}

    def __len__(self) -> int:
        """返回路径序列的长度 (状态数)。"""
        return len(self.sequence)

    def __repr__(self) -> str:
        """返回路径的可读字符串表示。"""
        seq_repr = "Empty"
        if self.sequence:
            # 简化长路径的表示
            if len(self.sequence) > 4:
                seq_repr = f"[{self.sequence[0]}, {self.sequence[1]}, ..., {self.sequence[-2]}, {self.sequence[-1]}]"
            else:
                seq_repr = str(self.sequence)
        makespan = self.get_makespan() if self.sequence else -1
        return f"Path(AGV={self.agv_id}, Length={len(self)}, Makespan={makespan})"

# --- 解决方案类型别名 ---
Solution = Dict[int, Path]

# --- 辅助函数 calculate_tij (修改: 接收 v) ---
def calculate_tij(node1: Node, node2: Node, v: float, delta_step: float, grid_map: 'GridMap') -> TimeStep:
    """
    计算从 node1 移动到相邻 node2 所需的理论最小时间步数。
    **v11 修改**: 使用传入的特定 AGV 速度 v。

    Args:
        node1: 起始节点。
        node2: 目标节点。
        v: 此 AGV 的速度 (格/秒)。
        delta_step: 每个时间步的时长 (秒)。
        grid_map: GridMap 对象。

    Returns:
        理论最小时间步数 (向上取整)，如果移动无效则返回 float('inf')。

    Raises:
        TypeError: 如果 grid_map 缺少必要方法。
        ValueError: 如果 v 或 delta_step 无效。
    """
    try:
        # 获取节点间的移动成本（通常是距离）
        distance = grid_map.get_move_cost(node1, node2)
    except AttributeError:
        raise TypeError("grid_map 参数必须提供 get_move_cost 方法。")
    except Exception as e:
        print(f"错误: 调用 grid_map.get_move_cost({node1}, {node2}) 出错: {e}")
        return float('inf') # 视为无法移动

    # 如果移动无效（例如，移动到障碍物或距离过远）
    if distance == float('inf'):
        return float('inf')

    # --- 使用传入的 v 和 delta_step ---
    if v <= 0 or delta_step <= 0:
        raise ValueError("速度 v 和时间步长 delta_step 必须为正。")
    # --- ------------------------- ---

    # 处理原地移动或极小距离移动
    if distance < 1e-9:
        # 即使原地不动，也至少需要一个时间步来表示状态变化或等待决策
        return 1

    # 计算实际时间
    time_real = distance / v
    # 转换为时间步数
    time_steps_float = time_real / delta_step
    # 向上取整得到所需的时间步数
    time_steps_int = math.ceil(time_steps_float)

    # 确保移动至少需要 1 个时间步
    return max(1, int(time_steps_int))

# --- 辅助函数 check_time_overlap (保持 v10) ---
def check_time_overlap(start1: TimeStep, end1: TimeStep, start2: TimeStep, end2: TimeStep) -> bool:
    """
    检查两个左闭右开时间段 [start1, end1) 和 [start2, end2) 是否存在重叠。
    用于后续的边冲突检测。
    """
    # 基本验证: 确保 end >= start
    if end1 < start1 or end2 < start2:
        # print(f"警告 (TimeOverlap): 时间段无效: [{start1}, {end1}), [{start2}, {end2})") # 可选警告
        return False # 无效区间不视为重叠，或者可以抛出错误
    # 判断重叠: 如果一个区间的开始时间严格小于另一个区间的结束时间，
    # 并且反过来也成立，那么它们就重叠。
    overlap = (start1 < end2) and (start2 < end1)
    return overlap

# --- 示例用法 (更新以测试新的 get_cost) ---
if __name__ == '__main__':
    print("--- DataTypes 测试 (v11 - 支持 AGV 特定速度) ---")

    # --- 任务示例 ---
    task1 = Task(agv_id=0, start_node=(0, 0), goal_node=(2, 2))
    print("任务示例:", task1)

    # --- 路径示例 ---
    path_seq1 = [((0, 0), 0), ((1, 0), 1), ((2, 0), 2), ((2, 0), 3), ((2, 1), 5), ((2, 2), 7)]
    path1 = Path(agv_id=0, sequence=path_seq1)
    print("路径示例 1:", path1)

    # --- 模拟 GridMap ---
    class TempGridMap:
        def get_move_cost(self, n1, n2):
            dx=abs(n1[0]-n2[0]); dy=abs(n1[1]-n2[1])
            # 允许相邻移动（包括对角线）
            if dx<=1 and dy<=1 and (dx!=0 or dy!=0):
                return math.sqrt(dx**2 + dy**2) # 欧氏距离
            elif dx==0 and dy==0:
                return 0.0 # 原地成本为 0
            return float('inf') # 其他移动无效
        width=5
        height=5
    temp_map = TempGridMap()

    # --- 成本计算测试 ---
    alpha_test = 1.0; beta_test = 0.3; gamma_wait_test = 0.8
    v_test_normal = 1.0; v_test_fast = 1.5; v_test_slow = 0.7
    delta_step_test = 1.0

    print(f"\n成本计算测试 (alpha={alpha_test}, beta={beta_test}, gamma={gamma_wait_test}, delta_step={delta_step_test}):")

    # 测试正常速度
    costs_normal = path1.get_cost(temp_map, alpha_test, beta_test, gamma_wait_test, v_test_normal, delta_step_test)
    print(f"  速度 v={v_test_normal}:")
    print(f"    Total: {costs_normal.get('total', 'N/A'):.2f}")
    print(f"    Travel: {costs_normal.get('travel', 'N/A'):.2f}")
    print(f"    Turn: {costs_normal.get('turn', 'N/A'):.2f}")
    print(f"    Wait: {costs_normal.get('wait', 'N/A'):.2f}")

    # 测试较快速度 (预期行驶成本不变，因为基于理论步数，但如果路径不同，结果会变)
    # 注意: get_cost 本身不改变路径，所以 travel_cost 不会因为 v 变化而变化
    # v 的影响体现在规划器生成路径时，以及 calculate_tij 返回的理论步数
    costs_fast = path1.get_cost(temp_map, alpha_test, beta_test, gamma_wait_test, v_test_fast, delta_step_test)
    print(f"  速度 v={v_test_fast}:")
    print(f"    Total: {costs_fast.get('total', 'N/A'):.2f}")
    print(f"    Travel: {costs_fast.get('travel', 'N/A'):.2f}")
    print(f"    Turn: {costs_fast.get('turn', 'N/A'):.2f}")
    print(f"    Wait: {costs_fast.get('wait', 'N/A'):.2f}")

    # 测试较慢速度
    costs_slow = path1.get_cost(temp_map, alpha_test, beta_test, gamma_wait_test, v_test_slow, delta_step_test)
    print(f"  速度 v={v_test_slow}:")
    print(f"    Total: {costs_slow.get('total', 'N/A'):.2f}")
    print(f"    Travel: {costs_slow.get('travel', 'N/A'):.2f}")
    print(f"    Turn: {costs_slow.get('turn', 'N/A'):.2f}")
    print(f"    Wait: {costs_slow.get('wait', 'N/A'):.2f}")

    # --- calculate_tij 测试 ---
    print("\n测试 calculate_tij:")
    node_a = (0,0); node_b = (1,0); node_c = (1,1)
    print(f"  从 {node_a} 到 {node_b} (距离 1):")
    print(f"    v={v_test_normal}: {calculate_tij(node_a, node_b, v_test_normal, delta_step_test, temp_map)} 步")
    print(f"    v={v_test_fast}: {calculate_tij(node_a, node_b, v_test_fast, delta_step_test, temp_map)} 步") # 速度快，步数可能不变或减少
    print(f"    v={v_test_slow}: {calculate_tij(node_a, node_b, v_test_slow, delta_step_test, temp_map)} 步") # 速度慢，步数可能增加
    print(f"  从 {node_a} 到 {node_c} (距离 sqrt(2)):")
    print(f"    v={v_test_normal}: {calculate_tij(node_a, node_c, v_test_normal, delta_step_test, temp_map)} 步")
    print(f"    v={v_test_fast}: {calculate_tij(node_a, node_c, v_test_fast, delta_step_test, temp_map)} 步")
    print(f"    v={v_test_slow}: {calculate_tij(node_a, node_c, v_test_slow, delta_step_test, temp_map)} 步")
    print(f"  原地 {node_a} 到 {node_a} (距离 0):")
    print(f"    v={v_test_normal}: {calculate_tij(node_a, node_a, v_test_normal, delta_step_test, temp_map)} 步")

    # --- 时间重叠测试 (保持 v10) ---
    print("\n测试 check_time_overlap:")
    print(f"  [0, 2) vs [1, 3)? {check_time_overlap(0, 2, 1, 3)}") # True
    print(f"  [0, 2) vs [2, 4)? {check_time_overlap(0, 2, 2, 4)}") # False
    print(f"  [0, 2) vs [3, 5)? {check_time_overlap(0, 2, 3, 5)}") # False
    print(f"  [1, 3) vs [0, 4)? {check_time_overlap(1, 3, 0, 4)}") # True
    print(f"  [2, 3) vs [0, 2)? {check_time_overlap(2, 3, 0, 2)}") # False (临界不重叠)
    print(f"  [0, 0) vs [0, 1)? {check_time_overlap(0, 0, 0, 1)}") # False (空区间)
    print(f"  [0, 1) vs [1, 0)? {check_time_overlap(0, 1, 1, 0)}") # False (无效区间)
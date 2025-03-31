# DataTypes.py-v10 (新增 check_time_overlap 辅助函数)
"""
定义用于仓储 AGV 路径规划问题的核心数据结构。

与论文的关联:
- 核心概念: 定义了任务 (Task)、路径 (Path)、解决方案 (Solution) 和成本字典 (CostDict)，
           这些是构建数学模型 (Chapter 2) 和设计 ALNS 算法 (Chapter 3) 的基础。
- 解决方案表示: Path 类及其 sequence 属性直接对应论文 3.2 节描述的解决方案表示方法，
               并隐式关联数学模型中的位置状态变量 τ_ikt。
- 目标函数评估: Path.get_cost 方法用于计算单个路径的成本，其计算逻辑直接对应
                 论文的目标函数 (公式 1) 的三个组成部分（行驶、转弯、等待）。
- 节点类型 V_s, V_p: Task 类中的 start_node (s_k) 和 goal_node (e_k)
                     分别定义了论文模型中的 V_s (起始节点) 和 V_p (目标/拣选站节点)
                     集合中的元素。

版本变更 (v9 -> v10):
- 新增: 添加了 'check_time_overlap' 辅助函数，用于检查时间段重叠，
         以支持后续的边冲突处理逻辑。
- 保持: v9 的其他所有类、类型别名和功能保持不变。
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
        return f"Task(AGV={self.agv_id}, Start(Vs)={self.start_node}, Goal(Vp)={self.goal_node})"

# --- 路径类 ---
class Path:
    """
    表示单个 AGV 的时空路径。
    这是算法中解的核心组成部分，对应论文 3.2 节描述的解决方案表示。
    其 sequence 属性隐式地定义了数学模型中的位置状态变量 τ_ikt。
    """
    def __init__(self, agv_id: int, sequence: List[State]):
        # ()
        if not sequence: raise ValueError("路径序列不能为空。")
        if not all(isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], tuple) and len(s[0]) == 2 and isinstance(s[0][0], int) and isinstance(s[0][1], int) and isinstance(s[1], int) for s in sequence): raise ValueError("路径序列格式错误，应为 List[Tuple[Tuple[int, int], int]]。")
        for i in range(len(sequence) - 1):
             if sequence[i+1][1] < sequence[i][1]: raise ValueError(f"路径序列时间步必须单调不减: {sequence[i]} -> {sequence[i+1]}")
        self.agv_id = agv_id
        self.sequence: List[State] = sequence

    def get_start_node(self) -> Node:
        # ()
        return self.sequence[0][0]

    def get_goal_node(self) -> Node:
        # ()
        return self.sequence[-1][0]

    def get_makespan(self) -> TimeStep:
        # ()
        return self.sequence[-1][1]

    def nodes_occupied_at_time(self, time: TimeStep) -> Set[Node]:
        # ()
        occupied_nodes = set()
        for node, t in self.sequence:
            if t == time: occupied_nodes.add(node); break
            if t > time: break
        return occupied_nodes

    def get_cost(self, grid_map: 'GridMap', alpha: float, beta: float, gamma_wait: float, v: float, delta_step: float) -> CostDict:
        # ()
        if not hasattr(grid_map, 'get_move_cost'): raise TypeError("grid_map 参数必须提供 get_move_cost 方法。")
        if v <= 0 or delta_step <= 0: raise ValueError("速度 v 和时间步长 delta_step 必须为正。")
        if not self.sequence or len(self.sequence) < 1: return {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}

        total_travel_steps = 0.0
        total_turn_count = 0.0
        total_wait_steps = 0.0
        last_move_direction: Optional[Tuple[int, int]] = None

        for i in range(len(self.sequence) - 1):
            node_i, time_i = self.sequence[i]
            node_j, time_j = self.sequence[i+1]
            time_diff = time_j - time_i
            if time_diff < 0: return {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}

            if node_i == node_j: # 等待
                if time_diff > 0: total_wait_steps += float(time_diff)
            else: # 移动
                ideal_time_steps = calculate_tij(node_i, node_j, v, delta_step, grid_map)
                if ideal_time_steps == float('inf'): return {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
                total_travel_steps += float(ideal_time_steps)
                actual_move_steps = time_diff
                if actual_move_steps > ideal_time_steps + 1e-6: total_wait_steps += float(actual_move_steps - ideal_time_steps)
                current_move_direction = (node_j[0] - node_i[0], node_j[1] - node_i[1])
                if last_move_direction is not None:
                    dx1, dy1 = last_move_direction; dx2, dy2 = current_move_direction
                    if (dx1**2 + dy1**2 > 1e-9) and (dx2**2 + dy2**2 > 1e-9):
                        cross_product = dx1 * dy2 - dx2 * dy1
                        if abs(cross_product) > 1e-9: total_turn_count += 1.0
                last_move_direction = current_move_direction

        travel_cost = alpha * total_travel_steps
        turn_cost = beta * total_turn_count
        wait_cost = gamma_wait * total_wait_steps
        total_cost = travel_cost + turn_cost + wait_cost
        return {'total': total_cost, 'travel': travel_cost, 'turn': turn_cost, 'wait': wait_cost}

    def __len__(self) -> int:
        # ()
        return len(self.sequence)

    def __repr__(self) -> str:
        # ()
        seq_repr = "Empty"
        if self.sequence:
            if len(self.sequence) > 4: seq_repr = f"[{self.sequence[0]}, {self.sequence[1]}, ..., {self.sequence[-2]}, {self.sequence[-1]}]"
            else: seq_repr = str(self.sequence)
        makespan = self.get_makespan() if self.sequence else -1
        return f"Path(AGV={self.agv_id}, Length={len(self)}, Makespan={makespan})"

# --- 解决方案类型别名 ---
Solution = Dict[int, Path]

# --- 辅助函数 calculate_tij ---
def calculate_tij(node1: Node, node2: Node, v: float, delta_step: float, grid_map: 'GridMap') -> TimeStep:
    # ()
    try: distance = grid_map.get_move_cost(node1, node2)
    except AttributeError: raise TypeError("grid_map 参数必须提供 get_move_cost 方法。")
    except Exception as e: print(f"错误: 调用 grid_map.get_move_cost({node1}, {node2}) 出错: {e}"); return float('inf')
    if distance == float('inf'): return float('inf')
    if v <= 0 or delta_step <= 0: raise ValueError("速度 v 和时间步长 delta_step 必须为正。")
    if distance < 1e-9: return 1 # 原地移动或极小移动视为 1 步
    time_real = distance / v; time_steps_float = time_real / delta_step
    time_steps_int = math.ceil(time_steps_float)
    return max(1, int(time_steps_int)) # 确保至少为 1 步

# --- 新增: 辅助函数 check_time_overlap ---
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

# --- 示例用法 (保持 v9，并添加 overlap 测试) ---
if __name__ == '__main__':
    print("--- DataTypes 测试 (v10 - Added check_time_overlap) ---")
    task1 = Task(agv_id=0, start_node=(0, 0), goal_node=(2, 2))
    print("任务示例:", task1)
    path_seq1 = [((0, 0), 0), ((1, 0), 1), ((2, 0), 2), ((2, 0), 3), ((2, 1), 5), ((2, 2), 7)]
    path1 = Path(agv_id=0, sequence=path_seq1)
    print("路径示例 1:", path1)
    class TempGridMap:
        def get_move_cost(self, n1, n2):
            dx=abs(n1[0]-n2[0]); dy=abs(n1[1]-n2[1]);
            if dx<=1 and dy<=1 and (dx!=0 or dy!=0): return math.sqrt(dx**2 + dy**2);
            elif dx==0 and dy==0: return 0.0
            return float('inf')
        width=5; height=5
    temp_map = TempGridMap()
    alpha_test = 1.0; beta_test = 0.3; gamma_wait_test = 0.8; v_test = 1.0; delta_step_test = 1.0
    costs = path1.get_cost(temp_map, alpha_test, beta_test, gamma_wait_test, v_test, delta_step_test)
    print(f"成本计算测试 (alpha={alpha_test}, beta={beta_test}, gamma={gamma_wait_test}):")
    print(f"  Total: {costs.get('total', 'N/A'):.2f}")
    print(f"  Travel: {costs.get('travel', 'N/A'):.2f}")
    print(f"  Turn: {costs.get('turn', 'N/A'):.2f}")
    print(f"  Wait: {costs.get('wait', 'N/A'):.2f}")

    print("\n测试 check_time_overlap:")
    print(f"[0, 2) vs [1, 3)? {check_time_overlap(0, 2, 1, 3)}") # True
    print(f"[0, 2) vs [2, 4)? {check_time_overlap(0, 2, 2, 4)}") # False
    print(f"[0, 2) vs [3, 5)? {check_time_overlap(0, 2, 3, 5)}") # False
    print(f"[1, 3) vs [0, 4)? {check_time_overlap(1, 3, 0, 4)}") # True
    print(f"[2, 3) vs [0, 2)? {check_time_overlap(2, 3, 0, 2)}") # False (临界不重叠)
    print(f"[0, 0) vs [0, 1)? {check_time_overlap(0, 0, 0, 1)}") # False (空区间)
    print(f"[0, 1) vs [1, 0)? {check_time_overlap(0, 1, 1, 0)}") # False (无效区间)
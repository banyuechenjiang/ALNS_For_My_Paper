# DataTypes.py-v9
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

版本变更 (v8 -> v9):
- 更新了模块和 Task 类文档字符串，明确 Task 如何定义 V_s 和 V_p 中的节点。
- 保持了 v8 的其他文档和功能。
"""
import math
from typing import List, Tuple, NamedTuple, Set, Dict, Optional, TYPE_CHECKING

# --- 使用 TYPE_CHECKING 避免运行时导入 GridMap ---
if TYPE_CHECKING:
    from Map import GridMap # 仅用于静态类型检查

# --- 核心类型别名 (保持 v8) ---
Node = Tuple[int, int] # (x, y) 坐标，对应论文模型中的节点索引 i, j, l
TimeStep = int # 离散时间步，对应论文模型中的时间索引 t
State = Tuple[Node, TimeStep]  # (节点, 时间步)，表示 AGV k 在时间步 t 结束时的位置状态 τ_ikt=1
DynamicObstacles = Dict[TimeStep, Set[Node]] # 用于存储动态障碍，{时间步: {被占用的节点集合}}
CostDict = Dict[str, float] # 存储成本分项，例如: {'total': 10.5, 'travel': 8.0, 'turn': 1.5, 'wait': 1.0}

# --- 任务类 (v9 - 更新文档) ---
class Task(NamedTuple):
    """
    表示一个 AGV 的搬运任务。
    定义了 AGV 的起点和终点，这些点分别属于论文模型中定义的
    V_s (起始节点集) 和 V_p (目标/拣选站节点集)。
    """
    agv_id: int     # AGV 索引 k
    start_node: Node # 起始节点 s_k (属于 V_s)
    goal_node: Node  # 目标节点 e_k (属于 V_p)

    def __repr__(self) -> str:
        """返回任务的可读字符串表示。"""
        return f"Task(AGV={self.agv_id}, Start(Vs)={self.start_node}, Goal(Vp)={self.goal_node})" # 在 repr 中也体现

# --- 路径类 (保持 v8) ---
class Path:
    """
    表示单个 AGV 的时空路径。
    这是算法中解的核心组成部分，对应论文 3.2 节描述的解决方案表示。
    其 sequence 属性隐式地定义了数学模型中的位置状态变量 τ_ikt。
    """
    def __init__(self, agv_id: int, sequence: List[State]):
        """
        初始化路径。
        (验证逻辑保持不变)
        """
        if not sequence: raise ValueError("路径序列不能为空。")
        if not all(isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], tuple) and len(s[0]) == 2 and isinstance(s[0][0], int) and isinstance(s[0][1], int) and isinstance(s[1], int) for s in sequence): raise ValueError("路径序列格式错误，应为 List[Tuple[Tuple[int, int], int]]。")
        for i in range(len(sequence) - 1):
             if sequence[i+1][1] < sequence[i][1]: raise ValueError(f"路径序列时间步必须单调不减: {sequence[i]} -> {sequence[i+1]}")
        self.agv_id = agv_id
        self.sequence: List[State] = sequence

    def get_start_node(self) -> Node:
        """获取路径的起始节点 (s_k)。"""
        return self.sequence[0][0]

    def get_goal_node(self) -> Node:
        """获取路径的最终节点 (e_k)。"""
        return self.sequence[-1][0]

    def get_makespan(self) -> TimeStep:
        """获取路径的完成时间 (最后一个状态的时间戳)。"""
        return self.sequence[-1][1]

    def nodes_occupied_at_time(self, time: TimeStep) -> Set[Node]:
        """获取在给定时间步 t 结束时，AGV 占用的节点 (用于冲突检测)。"""
        occupied_nodes = set()
        for node, t in self.sequence:
            if t == time: occupied_nodes.add(node); break
            if t > time: break
        return occupied_nodes

    def get_cost(self, grid_map: 'GridMap', alpha: float, beta: float, gamma_wait: float, v: float, delta_step: float) -> CostDict:
        """
        计算此路径的总成本及分项成本。
        此方法的计算逻辑直接对应论文第二章的目标函数 (公式 1)。
        TotalCost = α * TravelCostTerm + β * TurnCostTerm + γ_wait * WaitCostTerm

        (实现逻辑保持 v8，注释已充分)
        """
        if not hasattr(grid_map, 'get_move_cost'): raise TypeError("grid_map 参数必须提供 get_move_cost 方法。")
        if v <= 0 or delta_step <= 0: raise ValueError("速度 v 和时间步长 delta_step 必须为正。")
        if not self.sequence or len(self.sequence) < 1: return {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}

        total_travel_steps = 0.0 # 累加理论行驶步数 (对应目标函数中 sum(m_ijkt * t_ij))
        total_turn_count = 0.0   # 累加转弯次数 (对应目标函数中 sum(y_ijlt))
        total_wait_steps = 0.0   # 累加等待步数 (对应目标函数中 sum(w_ikt))
        last_move_direction: Optional[Tuple[int, int]] = None

        for i in range(len(self.sequence) - 1):
            node_i, time_i = self.sequence[i]     # 当前状态 (i, k, t)
            node_j, time_j = self.sequence[i+1]   # 下一个状态 (j, k, t+Δt)
            time_diff = time_j - time_i           # 状态间的时间差 (Δt)
            if time_diff < 0: return {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}

            if node_i == node_j: # 等待 (w_ikt=1)
                if time_diff > 0: total_wait_steps += float(time_diff)
            else: # 移动 (m_ijkt=1)
                ideal_time_steps = calculate_tij(node_i, node_j, v, delta_step, grid_map)
                if ideal_time_steps == float('inf'): return {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
                # 行驶成本分量
                total_travel_steps += float(ideal_time_steps)
                # 隐式等待
                actual_move_steps = time_diff
                if actual_move_steps > ideal_time_steps + 1e-6: total_wait_steps += float(actual_move_steps - ideal_time_steps)
                # 转弯成本分量 (y_ijlt=1)
                current_move_direction = (node_j[0] - node_i[0], node_j[1] - node_i[1])
                if last_move_direction is not None:
                    dx1, dy1 = last_move_direction; dx2, dy2 = current_move_direction
                    if (dx1**2 + dy1**2 > 1e-9) and (dx2**2 + dy2**2 > 1e-9):
                        cross_product = dx1 * dy2 - dx2 * dy1
                        if abs(cross_product) > 1e-9: total_turn_count += 1.0
                last_move_direction = current_move_direction

        # 最终加权成本 (公式 1)
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
            if len(self.sequence) > 4: seq_repr = f"[{self.sequence[0]}, {self.sequence[1]}, ..., {self.sequence[-2]}, {self.sequence[-1]}]"
            else: seq_repr = str(self.sequence)
        makespan = self.get_makespan() if self.sequence else -1
        return f"Path(AGV={self.agv_id}, Length={len(self)}, Makespan={makespan})"

# --- 解决方案类型别名 (保持 v8) ---
Solution = Dict[int, Path] # 解决方案是 AGV ID 到其路径的映射

# --- 辅助函数 calculate_tij (保持 v8) ---
def calculate_tij(node1: Node, node2: Node, v: float, delta_step: float, grid_map: 'GridMap') -> TimeStep:
    """
    计算从 node1 移动到相邻 node2 所需的整数时间步数。
    对应论文模型参数 v (速度) 和 δ_step (时间步长) 的应用。
    计算结果 t_ij 用于目标函数和约束。

    (实现逻辑保持 v8)
    """
    try: distance = grid_map.get_move_cost(node1, node2)
    except AttributeError: raise TypeError("grid_map 参数必须提供 get_move_cost 方法。")
    except Exception as e: print(f"错误: 调用 grid_map.get_move_cost({node1}, {node2}) 出错: {e}"); return float('inf')
    if distance == float('inf'): return float('inf')
    if v <= 0 or delta_step <= 0: raise ValueError("速度 v 和时间步长 delta_step 必须为正。")
    if distance < 1e-9: return 1
    time_real = distance / v; time_steps_float = time_real / delta_step
    time_steps_int = math.ceil(time_steps_float)
    return max(1, int(time_steps_int))

# --- 示例用法 (保持 v8) ---
if __name__ == '__main__':
    print("--- DataTypes 测试 (v9 - Enhanced Docs & Model Links) ---")
    # 使用 Task 的新 repr
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
    print(f"  Total: {costs.get('total', 'N/A'):.2f}") # 预期 6.70
    print(f"  Travel: {costs.get('travel', 'N/A'):.2f}") # 预期 4.00
    print(f"  Turn: {costs.get('turn', 'N/A'):.2f}") # 预期 0.30
    print(f"  Wait: {costs.get('wait', 'N/A'):.2f}") # 预期 2.40
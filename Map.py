# Map.py-v7
"""
定义地图环境类 GridMap。
该类负责表示仓储环境的二维栅格地图，包括静态障碍物及其膨胀区域。

与论文的关联:
- 地图表示 (模型 G=(V,E)): GridMap 对象代表了论文模型中定义的栅格地图 G=(V,E)，
                           其中节点 V 是所有 (x,y) 坐标，边 E 由相邻关系定义。
- 节点集合 (模型 V, V_o, V_s, V_p, V_f):
    - V: 由地图的 width 和 height 隐式定义所有节点。
    - V_o: 由 self.obstacles (包含原始障碍物 self.raw_obstacles 和膨胀缓冲区
           self.buffer_zones) 明确表示。
    - V_s (起始点集), V_p (目标/拣选站集): 不由 GridMap 类直接管理。这些节点属于 V 且
      不属于 V_o，它们的具体位置由外部的任务定义 (DataTypes.Task) 提供。
    - V_f (货架集) 和其他自由通行节点: 这些节点是 V 中除了 V_o, V_s, V_p 之外的
      所有有效节点。本类不显式区分它们，规划器主要视其为可通行区域。
- 障碍物膨胀 (假设 1): 构造函数中的 expansion_radius 参数和相关逻辑实现了论文
                       假设 1 中描述的障碍物膨胀处理。
- 移动成本 (模型 d_ij): get_move_cost 方法用于计算相邻节点间的移动成本 d_ij (通常是距离)，
                       是目标函数和规划器计算的基础。
- 节点有效性/邻居: is_valid 和 get_neighbors 方法用于路径规划器判断节点是否可用
                 (不在 V_o 内) 以及探索可能的移动 (状态转移)。

版本变更 (v6 -> v7):
- 更新了模块和类文档字符串，添加了关于 V_s, V_p, V_f 与 GridMap 类关系的说明。
- 保持了 v6 的实现逻辑和注释。
"""
import heapq
import math
from typing import List, Tuple, Set, Optional

# 定义节点类型别名 (对应论文模型中的节点索引 i, j, l)
Node = Tuple[int, int]

class GridMap:
    """
    表示栅格化地图环境 (对应模型 G=(V,E))。
    管理地图尺寸、障碍物 (V_o，含原始和膨胀)，并提供节点有效性、
    邻居查找和移动成本计算功能。不直接管理 V_s, V_p, V_f 节点类型。
    """

    # --- 构造函数 (保持 v6) ---
    def __init__(self, width: int, height: int, raw_obstacles: Optional[Set[Node]] = None, expansion_radius: int = 0):
        """
        初始化地图。

        Args:
            width (int): 地图宽度 (列数)。
            height (int): 地图高度 (行数)。
            raw_obstacles (Optional[Set[Node]]): 原始障碍物节点坐标集合。
            expansion_radius (int): 障碍物膨胀半径 (对应论文假设 1)。默认为 0。
        """
        # --- 参数验证 (保持 v6) ---
        if not isinstance(width, int) or width <= 0: raise ValueError("宽度必须是正整数。")
        if not isinstance(height, int) or height <= 0: raise ValueError("高度必须是正整数。")
        if raw_obstacles is not None and not isinstance(raw_obstacles, set): raise TypeError("原始障碍物必须是节点坐标 (x, y) 的集合。")
        if raw_obstacles:
            for obs in raw_obstacles:
                if not (isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[0], int) and isinstance(obs[1], int)): raise TypeError(f"原始障碍物坐标必须是 (int, int) 元组，发现: {obs}")
                if not (0 <= obs[0] < width and 0 <= obs[1] < height): raise ValueError(f"原始障碍物坐标 {obs} 超出地图边界 (宽度={width}, 高度={height})。")
        if not isinstance(expansion_radius, int) or expansion_radius < 0: raise ValueError("膨胀半径必须是非负整数。")

        # --- 实例属性赋值 (保持 v6) ---
        self.width = width
        self.height = height
        self.expansion_radius = expansion_radius # 存储膨胀半径 (论文假设 1)
        self.raw_obstacles: Set[Node] = raw_obstacles if raw_obstacles is not None else set()
        self.buffer_zones: Set[Node] = set()
        # 总障碍物集合 (包含原始和缓冲区)，对应模型 V_o
        self.obstacles: Set[Node] = set(self.raw_obstacles)

        # --- 执行障碍物膨胀 (实现论文假设 1) (保持 v6 逻辑) ---
        if self.expansion_radius > 0 and self.raw_obstacles:
            if __name__ != "__main__": print(f"  执行障碍物膨胀 (半径={self.expansion_radius})...")
            nodes_to_add_to_buffer = set()
            for ox, oy in self.raw_obstacles:
                for dx in range(-self.expansion_radius, self.expansion_radius + 1):
                    for dy in range(-self.expansion_radius, self.expansion_radius + 1):
                        if dx == 0 and dy == 0: continue
                        nx, ny = ox + dx, oy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                             if (nx, ny) not in self.raw_obstacles:
                                  nodes_to_add_to_buffer.add((nx, ny))
            self.buffer_zones = nodes_to_add_to_buffer
            self.obstacles.update(self.buffer_zones) # V_o = V_raw_obs U V_buffer
            if __name__ != "__main__": print(f"  膨胀完成，总障碍物 (V_o): {len(self.obstacles)} (原始: {len(self.raw_obstacles)}, 缓冲区: {len(self.buffer_zones)})")

    # --- 边界检查 (保持 v6) ---
    def is_within_bounds(self, x: int, y: int) -> bool:
        """检查坐标 (x, y) 是否在地图边界内。"""
        return 0 <= x < self.width and 0 <= y < self.height

    # --- 障碍物检查 (保持 v6) ---
    def is_obstacle(self, x: int, y: int) -> bool:
        """
        检查坐标 (x, y) 是否为总障碍物 (原始或缓冲区)。
        对应于检查节点是否属于模型中的 V_o 集合。
        """
        return (x, y) in self.obstacles

    def is_raw_obstacle(self, x: int, y: int) -> bool:
        """检查坐标 (x, y) 是否为原始障碍物。"""
        return (x, y) in self.raw_obstacles

    def is_buffer_zone(self, x: int, y: int) -> bool:
        """检查坐标 (x, y) 是否仅为膨胀缓冲区 (非原始障碍物)。"""
        return (x, y) in self.buffer_zones

    # --- 节点有效性检查 (保持 v6) ---
    def is_valid(self, x: int, y: int) -> bool:
        """
        检查坐标 (x, y) 是否有效：在边界内且非总障碍物。
        用于规划器判断节点是否可用（即不在 V_o 中）。
        """
        return self.is_within_bounds(x, y) and not self.is_obstacle(x, y)

    # --- 获取邻居 (保持 v6) ---
    def get_neighbors(self, node: Node) -> List[Node]:
        """
        获取给定节点的有效邻居列表（八向移动）。
        只返回在地图边界内且非总障碍物 (不在 V_o 中) 的邻居。
        用于规划器探索状态转移。
        """
        x, y = node
        possible_moves: List[Tuple[int, int]] = [
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), # 上下左右
            (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1) # 对角线
        ]
        valid_neighbors: List[Node] = []
        for move_x, move_y in possible_moves:
            if self.is_valid(move_x, move_y): # 检查不在 V_o 中
                valid_neighbors.append((move_x, move_y))
        return valid_neighbors

    # --- 获取移动成本 (保持 v6) ---
    def get_move_cost(self, node1: Node, node2: Node) -> float:
        """
        计算从 node1 移动到相邻 node2 的成本（欧几里得距离）。
        对应模型中的移动成本/距离 d_ij。
        如果移动无效（非邻居或目标是总障碍物 V_o），返回无穷大。
        """
        if not self.is_valid(node2[0], node2[1]): # 检查目标是否在 V_o 中
            return float('inf')
        dx = abs(node1[0] - node2[0]); dy = abs(node1[1] - node2[1])
        if dx <= 1 and dy <= 1:
            if dx == 0 and dy == 0: return 0.0
            else: return math.sqrt(dx**2 + dy**2) # d_ij
        else: return float('inf') # 非邻居

    # --- 对象表示 (保持 v6) ---
    def __repr__(self) -> str:
        """返回地图对象的可读字符串表示。"""
        return (f"GridMap(width={self.width}, height={self.height}, "
                f"total_obstacles(V_o)={len(self.obstacles)}, raw={len(self.raw_obstacles)}, "
                f"buffer={len(self.buffer_zones)}, expansion_radius={self.expansion_radius})")

# --- 示例用法 (保持 v6) ---
if __name__ == '__main__':
    print("--- GridMap 测试 (v7 - Enhanced Docs & Model Links) ---")
    raw_obs = {(2, 2)}
    print(f"原始障碍物: {raw_obs}")

    # 1. 膨胀半径为 1
    test_map_exp1 = GridMap(width=5, height=5, raw_obstacles=raw_obs, expansion_radius=1)
    print(f"\n测试膨胀半径为 1 (对应假设 1): {test_map_exp1}")
    print(f"  总障碍物 (obstacles / V_o): {test_map_exp1.obstacles}")
    print(f"  缓冲区 (buffer_zones): {test_map_exp1.buffer_zones}")
    print(f"\n  检查点类型:")
    print(f"    (2, 2): is_obstacle={test_map_exp1.is_obstacle(2, 2)} (T), is_raw={test_map_exp1.is_raw_obstacle(2, 2)} (T)")
    print(f"    (1, 1): is_obstacle={test_map_exp1.is_obstacle(1, 1)} (T), is_raw={test_map_exp1.is_raw_obstacle(1, 1)} (F), is_buffer={test_map_exp1.is_buffer_zone(1, 1)} (T)")
    print(f"    (0, 0): is_obstacle={test_map_exp1.is_obstacle(0, 0)} (F)")
    print(f"\n  检查有效性 (is_valid - 检查是否在 V_o 外):")
    print(f"    (2, 2) 有效? {test_map_exp1.is_valid(2, 2)} (F)")
    print(f"    (1, 1) 有效? {test_map_exp1.is_valid(1, 1)} (F)")
    print(f"    (0, 0) 有效? {test_map_exp1.is_valid(0, 0)} (T)")
    print(f"\n  检查邻居 (get_neighbors - 只返回非 V_o 邻居):")
    print(f"    节点 (0, 0) 的邻居: {test_map_exp1.get_neighbors((0, 0))}")
    print(f"    节点 (1, 0) 的邻居: {test_map_exp1.get_neighbors((1, 0))}")
    print(f"\n  检查移动成本 (get_move_cost - d_ij):")
    print(f"    (0,0) -> (1,0): {test_map_exp1.get_move_cost((0,0), (1,0)):.4f}")
    print(f"    (0,0) -> (1,1): {test_map_exp1.get_move_cost((0,0), (1,1)):.4f}") # inf
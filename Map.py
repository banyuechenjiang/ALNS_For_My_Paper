# Map.py -V10(修改后)
import heapq
import math
from typing import List, Tuple, Set, Optional
import traceback
from pathlib import Path
import json

# 定义节点类型别名
Node = Tuple[int, int]

class GridMap:
    """
    表示栅格化地图环境。
    """
    def __init__(self,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 raw_obstacles: Optional[Set[Node]] = None,
                 expansion_radius: int = 0,
                 grid_data: Optional[List[List[int]]] = None):
        """
        初始化地图。优先使用 grid_data 初始化。
        """
        raw_obstacles_internal: Set[Node] = set()
        self.grid: Optional[List[List[int]]] = None # 初始化 grid 属性

        if grid_data is not None:
            if not isinstance(grid_data, list) or not grid_data or not isinstance(grid_data[0], list):
                raise ValueError("grid_data 必须是有效的非空二维列表。")
            self.height = len(grid_data)
            self.width = len(grid_data[0])
            if self.height <= 0 or self.width <= 0:
                raise ValueError("grid_data 必须有正的宽度和高度。")
            self.grid = grid_data # <<<--- 存储 grid_data
            for r, row in enumerate(grid_data):
                if len(row) != self.width:
                    raise ValueError(f"grid_data 中所有行的长度必须等于宽度 {self.width} (行 {r} 长度为 {len(row)})。")
                for c, cell_type in enumerate(row):
                    if cell_type == 1:
                        raw_obstacles_internal.add((c, r))
        elif width is not None and height is not None:
            if not isinstance(width, int) or width <= 0: raise ValueError("宽度必须是正整数。")
            if not isinstance(height, int) or height <= 0: raise ValueError("高度必须是正整数。")
            self.width = width
            self.height = height
            if raw_obstacles is not None:
                if not isinstance(raw_obstacles, set): raise TypeError("原始障碍物必须是节点坐标 (x, y) 的集合。")
                for obs in raw_obstacles:
                    if not (isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[0], int) and isinstance(obs[1], int)): raise TypeError(f"原始障碍物坐标必须是 (int, int) 元组，发现: {obs}")
                    if not (0 <= obs[0] < self.width and 0 <= obs[1] < self.height): raise ValueError(f"原始障碍物坐标 {obs} 超出地图边界 (宽度={self.width}, 高度={self.height})。")
                raw_obstacles_internal = raw_obstacles
            # 注意：如果只提供 width/height，self.grid 将为 None
        else:
            raise ValueError("必须提供 grid_data 或同时提供 width 和 height 进行初始化。")

        self.expansion_radius = expansion_radius
        if not isinstance(expansion_radius, int) or expansion_radius < 0: raise ValueError("膨胀半径必须是非负整数。")
        self.raw_obstacles: Set[Node] = raw_obstacles_internal
        self.buffer_zones: Set[Node] = set()
        self.obstacles: Set[Node] = set(self.raw_obstacles)

        if self.expansion_radius > 0 and self.raw_obstacles:
            nodes_to_add_to_buffer = set()
            for ox, oy in self.raw_obstacles:
                min_dx, max_dx = -self.expansion_radius, self.expansion_radius
                min_dy, max_dy = -self.expansion_radius, self.expansion_radius
                for dx in range(min_dx, max_dx + 1):
                    nx = ox + dx
                    if 0 <= nx < self.width:
                         for dy in range(min_dy, max_dy + 1):
                             if dx == 0 and dy == 0: continue
                             ny = oy + dy
                             if 0 <= ny < self.height:
                                 node_to_check = (nx, ny)
                                 if node_to_check not in self.raw_obstacles:
                                     nodes_to_add_to_buffer.add(node_to_check)
            self.buffer_zones = nodes_to_add_to_buffer
            self.obstacles.update(self.buffer_zones)

    def is_within_bounds(self, x: int, y: int) -> bool:
        """检查坐标 (x, y) 是否在地图边界内。"""
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, x: int, y: int) -> bool:
        """检查坐标 (x, y) 是否为总障碍物 (原始或缓冲区)。"""
        return (x, y) in self.obstacles

    def is_raw_obstacle(self, x: int, y: int) -> bool:
        """检查坐标 (x, y) 是否为原始障碍物。"""
        return (x, y) in self.raw_obstacles

    def is_buffer_zone(self, x: int, y: int) -> bool:
        """检查坐标 (x, y) 是否仅为膨胀缓冲区。"""
        return (x, y) in self.buffer_zones

    def is_valid(self, x: int, y: int) -> bool:
        """检查坐标 (x, y) 是否有效：在边界内且非总障碍物。"""
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

    def get_neighbors(self, node: Node) -> List[Node]:
        """获取给定节点的有效邻居列表（八向移动）。"""
        x, y = node
        valid_neighbors: List[Node] = []
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                valid_neighbors.append((nx, ny))
        return valid_neighbors

    def get_move_cost(self, node1: Node, node2: Node) -> float:
        """计算从 node1 移动到相邻 node2 的成本（欧几里得距离）。"""
        if not self.is_valid(node2[0], node2[1]):
            return float('inf')
        dx = abs(node1[0] - node2[0]); dy = abs(node1[1] - node2[1])
        if dx <= 1 and dy <= 1:
            if dx == 0 and dy == 0: return 0.0
            else: return math.sqrt(dx**2 + dy**2)
        else:
            return float('inf')

    # <<<--- 新增方法 --->>>
    def get_node_type(self, x: int, y: int) -> int:
        """获取指定坐标 (x, y) 的节点类型。"""
        if self.grid is None:
            raise AttributeError("GridMap 未通过 grid_data 初始化，无法获取节点类型。")
        if not self.is_within_bounds(x, y):
            raise IndexError(f"坐标 ({x}, {y}) 超出地图边界 ({self.width}x{self.height})。")
        try:
            return self.grid[y][x] # grid[row][col] -> grid[y][x]
        except IndexError:
            raise IndexError(f"访问地图数据 grid[{y}][{x}] 时索引无效。")
    # <<<--- 新增方法结束 --->>>

    def __repr__(self) -> str:
        """返回地图对象的可读字符串表示。"""
        return (f"GridMap(width={self.width}, height={self.height}, "
                f"total_obstacles(V_o)={len(self.obstacles)}, raw={len(self.raw_obstacles)}, "
                f"buffer={len(self.buffer_zones)}, expansion_radius={self.expansion_radius})")

if __name__ == '__main__':
    print("--- GridMap 测试 (修改后) ---")
    map_json_path = "Map.json"; test_grid_data = None; grid_source = "硬编码 5x5"
    json_file = Path(map_json_path)
    if json_file.is_file():
        try:
            with open(json_file, 'r', encoding='utf-8') as f: map_json_content = json.load(f)
            if "dimensions" in map_json_content and "grid" in map_json_content and \
               isinstance(map_json_content["dimensions"], dict) and \
               "rows" in map_json_content["dimensions"] and "cols" in map_json_content["dimensions"] and \
               isinstance(map_json_content["grid"], list):
                test_grid_data = map_json_content["grid"]
                grid_source = f"Map.json ({map_json_content['dimensions']['cols']}x{map_json_content['dimensions']['rows']})"
                print(f"成功从 '{map_json_path}' 加载 grid_data。")
            else: print(f"警告: '{map_json_path}' 文件结构不符合预期，将回退。")
        except Exception as e: print(f"警告: 加载 '{map_json_path}' 出错: {e}。将回退。"); traceback.print_exc()
    if test_grid_data is None:
        test_grid_data = [[0, 0, 0, 0, 0],[0, 1, 0, 1, 0],[0, 0, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 0, 0],]
        grid_source = "硬编码 5x5"; print(f"使用硬编码 5x5 地图进行测试。")

    print(f"\n[1] 使用 {grid_source} grid_data 初始化 (膨胀半径=1)...")
    try:
        map_from_data = GridMap(grid_data=test_grid_data, expansion_radius=1)
        print(f"  地图对象: {map_from_data}")
        print(f"  测试 get_node_type(0, 0): {map_from_data.get_node_type(0, 0)} (应为 0)")
        obs_node_test = (1, 1) if grid_source == "硬编码 5x5" else (1, 19) # 假设 Map.json 的 (1,19) 是 1
        try: print(f"  测试 get_node_type{obs_node_test}: {map_from_data.get_node_type(*obs_node_test)}")
        except IndexError: print(f"  测试 get_node_type{obs_node_test}: 索引超出范围 (正常)。")
        print(f"  检查 {obs_node_test} (原始障碍物?): is_valid={map_from_data.is_valid(*obs_node_test)}")
    except Exception as e: print(f"  初始化或测试时发生错误: {e}"); traceback.print_exc()

    print("\n[2] 使用旧方式初始化 (不提供 grid_data)...")
    raw_obs_old = {(2, 2)}; map_old_way = GridMap(width=5, height=5, raw_obstacles=raw_obs_old, expansion_radius=0)
    print(f"  地图对象: {map_old_way}")
    try: node_type = map_old_way.get_node_type(1, 1); print(f"  测试 get_node_type(1, 1): {node_type}")
    except AttributeError as ae: print(f"  测试 get_node_type(1, 1): 触发 AttributeError '{ae}' (预期行为)。")
    except Exception as e: print(f"  测试 get_node_type(1, 1): 发生意外错误: {e}")
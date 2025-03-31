# InstanceGenerator.py-v12
"""
负责生成和加载用于仓储 AGV 路径规划研究的测试算例。
主要功能包括生成随机地图、生成随机任务以及加载手动设计的固定场景。

与论文的关联:
- 场景生成: 提供了生成符合论文中描述的两种核心测试场景的方法：
    - 手动设计场景 (对应论文实验部分可能描述的小规模固定场景)。
    - 随机生成场景 (对应论文实验部分用于测试算法在不同规模、
      复杂度下性能的随机场景，参数可配置)。
- 模型基础: 生成的 GridMap 对象定义了规划环境 (对应模型 G=(V,E))，
           其中障碍物 (V_o) 经过膨胀处理 (对应假设 1)。生成的 Task 列表
           定义了 AGV 的起始点 (s_k) 和目标点 (e_k) (对应模型参数和假设 5)。
- 节点类型 V_s, V_p, V_f (概念关联):
    - V_s (起始点集) 和 V_p (目标/拣选站集): 由本模块生成的 Task 列表
      中的 start_node 和 goal_node 隐式定义。每个任务的起点属于 V_s，
      终点属于 V_p。
    - V_f (货架集) 和其他自由通行节点: 是地图中除了障碍物 (V_o) 和
      任务指定的起终点 (V_s, V_p) 之外的所有有效节点。本模块不显式
      区分或生成 V_f。
- 地图连通性: 包含了检查地图连通性的功能，确保生成的随机地图对于
             路径规划是基本可行的。

版本变更 (v11 -> v12):
- 更新了模块和主要函数的文档字符串，明确 Task 如何隐式定义 V_s 和 V_p。
- 保留了 v11 的所有实现逻辑和功能。
"""
import random
import math
from typing import List, Tuple, Set, Optional, Dict
from collections import deque # 用于 BFS
import sys
import traceback

# --- 标准导入 (保持 v11) ---
try:
    from Map import GridMap, Node # 导入 v7
    from DataTypes import Task # 导入 v9
except ImportError as e:
    print(f"错误: 导入 InstanceGenerator 的依赖项失败: {e}")
    print("请确保 Map.py 和 DataTypes.py 在 Python 路径中。")
    # 定义临时的占位符类型以便静态分析，实际运行时如果导入失败会退出
    GridMap = type('GridMap', (object,), {})
    Task = type('Task', (object,), {})
    Node = tuple # 使用标准元组

# --- 地图连通性检查函数 (保持 v11) ---
def _is_connected(grid_map: 'GridMap') -> Tuple[bool, int]:
    """
    (内部辅助函数) 检查地图中的非障碍物节点是否完全连通（使用 BFS）。
    确保生成的随机地图具有基本的通路可行性。
    """
    if not isinstance(grid_map, GridMap): return False, 0 # 防御性检查
    valid_nodes: List[Node] = [(x, y) for x in range(grid_map.width) for y in range(grid_map.height) if grid_map.is_valid(x, y)]
    total_valid_nodes = len(valid_nodes)
    if total_valid_nodes <= 1: return True, total_valid_nodes
    start_node = valid_nodes[0]
    queue = deque([start_node])
    visited: Set[Node] = {start_node}
    reachable_nodes_count = 0
    while queue:
        current_node = queue.popleft()
        reachable_nodes_count += 1
        try: neighbors = grid_map.get_neighbors(current_node)
        except Exception as e: print(f"错误: 调用 grid_map.get_neighbors({current_node}) 出错: {e}"); return False, 0
        for neighbor in neighbors:
            if neighbor not in visited: visited.add(neighbor); queue.append(neighbor)
    is_fully_connected = (reachable_nodes_count == total_valid_nodes)
    return is_fully_connected, reachable_nodes_count

def get_largest_connected_component(grid_map: 'GridMap') -> Set[Node]:
    """
    (内部辅助函数) 找到地图中最大的非障碍物节点连通区域（使用 BFS）。
    用于确保随机生成的任务起点和终点在同一个主要活动区域内。
    """
    if not isinstance(grid_map, GridMap): return set()
    visited: Set[Node] = set()
    largest_component: Set[Node] = set()
    for x in range(grid_map.width):
        for y in range(grid_map.height):
            node = (x, y)
            if grid_map.is_valid(x, y) and node not in visited:
                current_component: Set[Node] = set()
                queue = deque([node])
                visited.add(node); current_component.add(node)
                while queue:
                    current_node = queue.popleft()
                    try: neighbors = grid_map.get_neighbors(current_node)
                    except Exception as e: print(f"错误: 调用 grid_map.get_neighbors({current_node}) 出错: {e}"); continue
                    for neighbor in neighbors:
                        if neighbor not in visited: visited.add(neighbor); current_component.add(neighbor); queue.append(neighbor)
                if len(current_component) > len(largest_component): largest_component = current_component
    return largest_component

# --- 地图生成 (保持 v11) ---
def generate_random_map(width: int, height: int, obstacle_ratio: float, expansion_radius: int = 0, ensure_connectivity: bool = True, min_lcc_ratio: float = 0.7) -> Optional['GridMap']:
    """
    生成具有随机障碍物的地图 (定义 V, V_o)，并可选地进行膨胀和连通性检查。
    对应论文中随机场景的地图生成部分。

    Args:
        width: 地图宽度。
        height: 地图高度。
        obstacle_ratio: 原始障碍物比例 (0 到 1)。
        expansion_radius: 障碍物膨胀半径 (对应论文假设 1)。
        ensure_connectivity: 是否确保最大连通分量 (LCC) 至少占有效节点的比例达到 min_lcc_ratio。
        min_lcc_ratio: LCC 最小比例要求。

    Returns:
        一个 GridMap 对象如果成功，否则 None。
    """
    # --- 输入验证 (保持 v11) ---
    if not (0 <= obstacle_ratio <= 1): print("错误: 障碍物比例必须在 0 到 1 之间。"); return None
    if width <= 0 or height <= 0: print("错误: 地图宽度和高度必须为正。"); return None
    if not isinstance(expansion_radius, int) or expansion_radius < 0: print("错误: 膨胀半径必须是非负整数。"); return None

    # --- 尝试生成地图直到满足条件 (保持 v11 逻辑) ---
    max_generation_attempts = 15
    for attempt in range(max_generation_attempts):
        raw_obstacles: Set[Node] = set()
        num_obstacles = int(width * height * obstacle_ratio)
        max_possible_obstacles = width * height - 2
        num_obstacles = min(num_obstacles, max_possible_obstacles)
        while len(raw_obstacles) < num_obstacles:
            x = random.randint(0, width - 1); y = random.randint(0, height - 1)
            if len(raw_obstacles) >= max_possible_obstacles: break
            raw_obstacles.add((x, y))
        try: grid_map = GridMap(width, height, raw_obstacles=raw_obstacles, expansion_radius=expansion_radius)
        except Exception as e: print(f"错误: (尝试 {attempt+1}) 创建 GridMap 对象失败: {e}"); traceback.print_exc(); continue
        if ensure_connectivity:
            lcc = get_largest_connected_component(grid_map)
            total_valid_nodes = width * height - len(grid_map.obstacles)
            if total_valid_nodes > 0:
                lcc_ratio_actual = len(lcc) / total_valid_nodes
                if lcc_ratio_actual >= min_lcc_ratio:
                    print(f"地图生成成功 (尝试 {attempt+1})，LCC 大小 {len(lcc)} / {total_valid_nodes} ({lcc_ratio_actual:.2%}) >= {min_lcc_ratio:.0%}")
                    return grid_map
            elif len(lcc) == 0: print(f"地图生成成功 (尝试 {attempt+1})，但没有有效节点。"); return grid_map
        else: print(f"地图生成成功 (尝试 {attempt+1})，未检查连通性。"); return grid_map
    print(f"错误: 无法在 {max_generation_attempts} 次尝试内生成满足条件的地图。"); return None

# --- 任务生成 (v12 - 更新文档) ---
def generate_random_tasks(grid_map: 'GridMap', num_agvs: int, use_lcc: bool = True) -> Optional[List[Task]]:
    """
    在给定地图上为指定数量的 AGV 生成随机任务 (Task 对象)。
    每个任务的 start_node 定义了该实例的 V_s (起始节点) 集合的一个元素，
    goal_node 定义了 V_p (目标节点) 集合的一个元素。
    确保起点和终点是有效节点（非 V_o）且互不相同，通常在最大连通分量 (LCC) 内。
    对应论文中随机场景的任务生成部分，满足假设 5 (任务已知)。

    Args:
        grid_map: 已生成的 GridMap 对象。
        num_agvs: 需要生成任务的 AGV 数量 (N)。
        use_lcc: 是否强制在最大连通分量内选择起终点。

    Returns:
        一个包含 Task 对象的列表如果成功，否则 None。
    """
    # --- 输入验证 (保持 v11) ---
    if not isinstance(grid_map, GridMap): print("错误: grid_map 必须是 GridMap 类型。"); return None
    if num_agvs <= 0: print("错误: AGV 数量必须为正。"); return None

    # --- 确定可用的节点池 (保持 v11 逻辑) ---
    valid_node_pool: List[Node] = []
    if use_lcc:
        lcc_nodes = get_largest_connected_component(grid_map)
        if len(lcc_nodes) < 2 * num_agvs:
            print(f"警告: 最大连通区域节点数 ({len(lcc_nodes)}) 不足以分配 {num_agvs} AGV 的起终点 ({2*num_agvs} 个)。回退到所有有效节点。")
            valid_node_pool = [(x, y) for x in range(grid_map.width) for y in range(grid_map.height) if grid_map.is_valid(x, y)]
            if len(valid_node_pool) < 2 * num_agvs: print(f"错误: 地图上所有有效节点 ({len(valid_node_pool)}) 也不足以分配 {num_agvs} AGV 的起终点。"); return None
        else: valid_node_pool = list(lcc_nodes)
    else: valid_node_pool = [(x, y) for x in range(grid_map.width) for y in range(grid_map.height) if grid_map.is_valid(x, y)]
    if len(valid_node_pool) < 2 * num_agvs: print(f"错误: 可用节点池 ({len(valid_node_pool)}) 不足以分配 {num_agvs} AGV 的起终点 ({2*num_agvs} 个)。"); return None

    # --- 尝试为每个 AGV 分配不冲突的起终点 (保持 v11 逻辑) ---
    tasks: List[Task] = []
    assigned_nodes: Set[Node] = set()
    max_total_attempts = num_agvs * len(valid_node_pool) * 10
    current_total_attempts = 0
    for agv_id in range(num_agvs):
        start_node: Optional[Node] = None; goal_node: Optional[Node] = None
        agv_attempts = 0; max_attempts_per_agv = len(valid_node_pool) * 10
        while agv_attempts < max_attempts_per_agv and current_total_attempts < max_total_attempts:
            agv_attempts += 1; current_total_attempts += 1
            try:
                available_nodes_for_pair = [n for n in valid_node_pool if n not in assigned_nodes]
                if len(available_nodes_for_pair) < 2: break
                chosen_pair = random.sample(available_nodes_for_pair, 2)
                start_node_try = chosen_pair[0]; goal_node_try = chosen_pair[1]
                if start_node_try not in assigned_nodes and goal_node_try not in assigned_nodes:
                    start_node = start_node_try; goal_node = goal_node_try
                    assigned_nodes.add(start_node); assigned_nodes.add(goal_node)
                    break
                else: start_node, goal_node = None, None
            except ValueError: break
            except Exception as e: print(f"错误: 在为 AGV {agv_id} 选择起点/终点时发生未知错误: {e}"); return None
        if start_node is not None and goal_node is not None: tasks.append(Task(agv_id=agv_id, start_node=start_node, goal_node=goal_node))
        else: print(f"错误: 无法为 AGV {agv_id} 生成不冲突的起点/终点（尝试次数耗尽）。任务生成失败。"); return None
    if len(tasks) == num_agvs: print(f"成功为所有 {num_agvs} 个 AGV 分配了任务 (定义了 V_s 和 V_p)。"); return tasks
    else: print(f"错误: 最终只成功生成了 {len(tasks)} / {num_agvs} 个任务。"); return None

# --- 固定算例加载 (v12 - 更新文档) ---
def load_fixed_scenario_1(expansion_radius: int = 0) -> Optional[Tuple['GridMap', List[Task]]]:
    """
    加载手动设计的固定算例场景 1 (10x10 地图, 3 个 AGV)。
    对应论文实验部分使用的特定小规模实例。
    任务定义中的 'start' 节点属于 V_s，'goal' 节点属于 V_p。

    Args:
        expansion_radius: 障碍物膨胀半径 (应与实验设置一致)。

    Returns:
        (GridMap, List[Task]) 如果成功，否则 None。
    """
    print(f"加载手动设计的固定算例场景 1 (10x10, expansion_radius={expansion_radius})...")
    if 'GridMap' not in globals() or 'Task' not in globals() or GridMap is None or Task is None: print("错误: GridMap 或 Task 类未正确加载。"); return None

    # --- 定义场景参数 (保持 v11) ---
    width = 10; height = 10
    raw_obstacles_s1: Set[Node] = {(3, 4), (6, 5)}
    tasks_data = [
        {'id': 0, 'start': (0, 1), 'goal': (9, 8)}, # start 属于 V_s, goal 属于 V_p
        {'id': 1, 'start': (1, 8), 'goal': (8, 1)}, # start 属于 V_s, goal 属于 V_p
        {'id': 2, 'start': (5, 0), 'goal': (5, 9)}  # start 属于 V_s, goal 属于 V_p
    ]
    num_agvs_s1 = len(tasks_data)
    print(f"  使用手动设计障碍物，原始数量: {len(raw_obstacles_s1)}")

    # --- 创建地图 (保持 v11) ---
    try: grid_map = GridMap(width, height, raw_obstacles=raw_obstacles_s1, expansion_radius=expansion_radius)
    except Exception as e: print(f"错误: 创建场景 1 地图失败: {e}"); traceback.print_exc(); return None

    # --- 创建并验证任务 (保持 v11) ---
    tasks: List[Task] = []; assigned_nodes_fixed: Set[Node] = set(); valid_tasks = True
    for t_data in tasks_data:
        task = Task(agv_id=t_data['id'], start_node=t_data['start'], goal_node=t_data['goal'])
        if not grid_map.is_valid(*task.start_node): print(f"错误: 场景 1 AGV {task.agv_id} 的起点 {task.start_node} 在膨胀后无效。"); valid_tasks = False; break
        if not grid_map.is_valid(*task.goal_node): print(f"错误: 场景 1 AGV {task.agv_id} 的终点 {task.goal_node} 在膨胀后无效。"); valid_tasks = False; break
        if task.start_node in assigned_nodes_fixed: print(f"错误: 场景 1 AGV {task.agv_id} 的起点 {task.start_node} 与之前的任务冲突。"); valid_tasks = False; break
        assigned_nodes_fixed.add(task.start_node)
        if task.goal_node in assigned_nodes_fixed: print(f"错误: 场景 1 AGV {task.agv_id} 的终点 {task.goal_node} 与之前的任务冲突。"); valid_tasks = False; break
        assigned_nodes_fixed.add(task.goal_node)
        tasks.append(task)
    if valid_tasks: print(f"手动设计场景 1 任务点 (定义了 V_s, V_p) 设置完成，共 {len(tasks)} 个任务。"); return grid_map, tasks
    else: return None

# --- 随机算例生成 (场景 2) (v12 - 更新文档) ---
def generate_scenario_2_instance(width: int, height: int, obstacle_ratio: float, num_agvs: int, expansion_radius: int = 0) -> Optional[Tuple['GridMap', List[Task]]]:
    """
    生成论文中定义的随机算例场景 2 的一个实例。
    组合了随机地图生成 (定义 V, V_o) 和随机任务生成 (定义 V_s, V_p)。

    Args:
        width: 地图宽度。
        height: 地图高度。
        obstacle_ratio: 原始障碍物比例。
        num_agvs: AGV 数量。
        expansion_radius: 障碍物膨胀半径。

    Returns:
        (GridMap, List[Task]) 如果成功，否则 None。
    """
    print(f"生成随机算例场景 2 实例 ({width}x{height}, obs_ratio={obstacle_ratio}, agvs={num_agvs}, expansion_radius={expansion_radius})...")
    if 'GridMap' not in globals() or 'Task' not in globals() or GridMap is None or Task is None: print("错误: GridMap 或 Task 类未正确加载。"); return None

    # 1. 生成随机地图 (确保连通性) (保持 v11)
    grid_map = generate_random_map(width, height, obstacle_ratio, expansion_radius=expansion_radius, ensure_connectivity=True, min_lcc_ratio=0.7)
    if not grid_map: print("错误: 生成随机地图失败。"); return None

    # 2. 在生成的地图上生成随机任务 (通常在 LCC 内) (保持 v11)
    tasks = generate_random_tasks(grid_map, num_agvs, use_lcc=True)
    if not tasks: print("错误: 生成随机任务失败。"); return None

    # 成功生成地图和任务
    print(f"成功生成随机地图 (膨胀后障碍物: {len(grid_map.obstacles)}) 和 {len(tasks)} 个随机任务 (定义了 V_s, V_p)。")
    return grid_map, tasks

# --- 示例用法 (保持 v11) ---
if __name__ == '__main__':
    print("--- Instance Generator 测试 (v12 - Enhanced Docs & Node Type Links) ---")
    test_expansion_radius = 1
    print(f"\n[1] 测试加载手动设计的场景 1 (膨胀半径={test_expansion_radius})...")
    scenario1_data = load_fixed_scenario_1(expansion_radius=test_expansion_radius)
    if scenario1_data:
        map1, tasks1 = scenario1_data; print(f"  地图: {map1}"); print(f"  任务数量: {len(tasks1)}");
        if tasks1: print(f"  第一个任务 (Start in Vs, Goal in Vp): {tasks1[0]}")
        is_c1, count_c1 = _is_connected(map1); print(f"  场景 1 地图是否完全连通? {is_c1} (可达: {count_c1})")
    else: print("  加载手动设计场景 1 失败。")
    test_obs_ratio_s2 = 0.1
    print(f"\n[2] 测试生成随机场景 2 实例 (膨胀半径={test_expansion_radius}, obs_ratio={test_obs_ratio_s2})...")
    scenario2_instance = generate_scenario_2_instance(width=12, height=12, obstacle_ratio=test_obs_ratio_s2, num_agvs=5, expansion_radius=test_expansion_radius)
    if scenario2_instance:
        map2, tasks2 = scenario2_instance; print(f"  地图: {map2}"); print(f"  任务数量: {len(tasks2)}");
        if tasks2: print(f"  第一个任务 (Start in Vs, Goal in Vp): {tasks2[0]}")
    else: print("  生成随机场景 2 实例失败。")
    test_obs_ratio_conn = 0.08
    print(f"\n[3&4] 测试连通性检查 (膨胀半径={test_expansion_radius}, obs_ratio={test_obs_ratio_conn})...")
    map_conn_test = generate_random_map(width=10, height=10, obstacle_ratio=test_obs_ratio_conn, expansion_radius=test_expansion_radius, ensure_connectivity=True, min_lcc_ratio=0.7)
    if map_conn_test:
         print(f"  生成地图 (10x10, {test_obs_ratio_conn*100:.0f}% obs, radius={test_expansion_radius}): {map_conn_test}")
         is_c, count_c = _is_connected(map_conn_test); lcc_c = get_largest_connected_component(map_conn_test); total_valid_c = 10*10 - len(map_conn_test.obstacles); lcc_ratio = len(lcc_c) / total_valid_c if total_valid_c > 0 else 0
         print(f"    是否完全连通? {is_c} (可达: {count_c})"); print(f"    LCC 大小: {len(lcc_c)} / {total_valid_c} ({lcc_ratio:.2%}) (要求 >= 70%)")
    else: print("  生成用于连通性测试的随机地图失败。")
# InstanceGenerator.py-v15 (返回 AGV 速度信息)
"""
负责生成和加载用于仓储 AGV 路径规划研究的测试算例。
主要功能包括从 JSON 文件加载地图、生成随机任务 (使用 Tasks.py-v5 逻辑)
以及加载手动设计的固定场景。

与论文的关联:
- 场景生成:
    - `load_fixed_scenario_1`: 加载固定场景，并返回关联的 AGV 速度信息。
    - `generate_scenario_2_instance`: 加载地图、生成随机任务，并返回关联的 AGV 速度信息。
- 模型基础:
    - GridMap 对象定义环境。
    - Task 列表定义起终点。
    - **新增**: 返回 AGV 速度字典，支持异构性实验。
- 节点类型关联: 任务生成现在更明确地关联区域类型 (4->2, 2->3, 3->4)。

版本变更 (v14 -> v15):
- **修改**: `load_fixed_scenario_1` 和 `generate_scenario_2_instance` 的返回值
           从 `Tuple[GridMap, List[Task]]` 更改为
           `Tuple[GridMap, List[Task], Dict[int, float]]`，增加了 AGV 速度字典。
- **修改**: `if __name__ == '__main__':` 部分更新，演示如何处理和生成返回的速度信息。
- **保持**: 核心的地图加载和任务生成调用逻辑不变。
- **依赖**: Map.py(v9+), DataTypes.py(v10+), Tasks.py(v5+)。
"""
import random
import math
from typing import List, Tuple, Set, Optional, Dict
from collections import deque
import sys
import traceback
import json # 用于读取 JSON
from pathlib import Path # 用于处理文件路径

# --- 标准导入 ---
try:
    # 依赖 v9+ 的 Map
    from Map import GridMap, Node
    # 依赖 v10+ 的 DataTypes
    from DataTypes import Task
    # --- 导入 v6 的任务生成函数 ---
    from Tasks import generate_tasks_goods_to_person_v6
except ImportError as e:
    print(f"错误: 导入 InstanceGenerator 的依赖项失败: {e}")
    print("请确保 Map.py(v9+), DataTypes.py(v10+), Tasks.py(v5+) 都在 Python 路径中。")
    GridMap = type('GridMap', (object,), {})
    Task = type('Task', (object,), {})
    Node = tuple
    # 定义一个占位符函数，以防 Tasks 导入失败
    def generate_tasks_goods_to_person_v6(*args, **kwargs) -> Optional[List[Task]]:
        print("错误: Tasks.py 未能正确导入，无法生成任务！")
        return None

# --- JSON 地图加载辅助函数 (与 v14 相同) ---
def _load_map_from_json(json_file_path: str) -> Optional[Tuple[int, int, List[List[int]]]]:
    """从 JSON 文件加载地图数据。"""
    file_path = Path(json_file_path)
    if not file_path.is_file():
        print(f"错误: 地图 JSON 文件未找到: {json_file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "dimensions" not in data or "grid" not in data:
            print(f"错误: JSON 文件 {json_file_path} 缺少 'dimensions' 或 'grid' 键。")
            return None
        dims = data["dimensions"]; grid_data = data["grid"]
        if "rows" not in dims or "cols" not in dims:
            print(f"错误: JSON 文件 {json_file_path} 的 'dimensions' 缺少 'rows' 或 'cols'。")
            return None
        height = dims["rows"]; width = dims["cols"]
        if not isinstance(height, int) or height <= 0 or not isinstance(width, int) or width <= 0:
            print(f"错误: JSON 文件 {json_file_path} 的 'rows' 和 'cols' 必须是正整数。")
            return None
        if not isinstance(grid_data, list) or len(grid_data) != height:
            print(f"错误: JSON 文件 {json_file_path} 的 'grid' 数据行数 ({len(grid_data)}) 与 'rows' ({height}) 不匹配。")
            return None
        if not all(isinstance(row, list) and len(row) == width for row in grid_data):
            print(f"错误: JSON 文件 {json_file_path} 的 'grid' 数据中存在行长度不等于 'cols' ({width}) 的情况。")
            return None
        return width, height, grid_data
    except json.JSONDecodeError as e:
        print(f"错误: 解析 JSON 文件 {json_file_path} 失败: {e}")
        return None
    except Exception as e:
        print(f"错误: 读取或处理 JSON 文件 {json_file_path} 时发生未知错误: {e}"); traceback.print_exc()
        return None

# --- 提取障碍物辅助函数 (与 v14 相同) ---
def _parse_grid_for_obstacles(grid_data: List[List[int]]) -> Set[Node]:
    """从 grid 数据中提取原始障碍物 (类型 1)。"""
    obstacles: Set[Node] = set()
    height = len(grid_data)
    if height == 0: return obstacles
    width = len(grid_data[0])
    for r in range(height):
        for c in range(width):
            if grid_data[r][c] == 1: obstacles.add((c, r))
    return obstacles

# --- 任务生成包装器 (与 v14 相同) ---
def generate_random_tasks(grid_map: 'GridMap', grid_data: List[List[int]], num_agvs: int) -> Optional[List[Task]]:
    """
    在给定地图上为 AGV 生成随机任务的包装器。
    调用 Tasks.py 中的 generate_tasks_goods_to_person_v6 实现。
    """
    if not isinstance(grid_map, GridMap): print("错误: grid_map 必须是 GridMap 类型。"); return None
    if num_agvs <= 0: print("错误: AGV 数量必须为正。"); return None
    print(f"  任务生成: 调用 Tasks.py (v5) 为 {num_agvs} 个 AGV 生成 G2P 任务...")
    try:
        tasks = generate_tasks_goods_to_person_v6(
            grid_map=grid_map,
            grid_data=grid_data,
            num_agvs=num_agvs
        )
        if tasks is None:
             print("错误: Tasks.py 返回 None，任务生成失败。")
             return None
        elif len(tasks) < num_agvs:
             print(f"警告: Tasks.py 最终只生成了 {len(tasks)} / {num_agvs} 个任务。")
             return tasks
        else:
             print(f"成功从 Tasks.py (v5) 获取了 {len(tasks)} 个任务。")
             return tasks
    except Exception as e:
        print(f"错误: 调用 Tasks.py 中的任务生成函数时发生异常: {e}")
        traceback.print_exc()
        return None

# --- 固定算例加载 (v15 - 返回速度字典) ---
def load_fixed_scenario_1(
    json_map_file: str = "Map.json",
    expansion_radius: int = 0,
    # 新增: 用于演示的速度参数，实际应用中速度可能在外部定义
    assign_speeds: bool = True,
    avg_speed: float = 1.0,
    fast_speed: float = 1.2,
    slow_speed: float = 0.8
) -> Optional[Tuple['GridMap', List[Task], Dict[int, float]]]:
    """
    加载固定算例场景 1。
    返回 GridMap, 任务列表, 以及 AGV 速度字典。
    """
    print(f"加载固定算例场景 1 (地图: {json_map_file}, expansion={expansion_radius})...")
    if 'GridMap' not in globals() or 'Task' not in globals() or GridMap is None or Task is None:
        print("错误: GridMap 或 Task 类未正确加载。"); return None

    map_data = _load_map_from_json(json_map_file)
    if map_data is None: return None
    width, height, grid_data = map_data

    try:
        grid_map = GridMap(grid_data=grid_data, expansion_radius=expansion_radius)
    except Exception as e:
        print(f"错误: 创建场景 1 地图失败: {e}"); traceback.print_exc(); return None

    # --- 手动定义与 Map.json (20x20) 兼容的固定任务 ---
    tasks_fixed_s1: List[Task] = []
    # 示例任务 (请根据你的 Map.json 调整!)
    fixed_task_data = [
        {'id': 0, 'start': (1, 19), 'goal': (1, 0)},
        {'id': 1, 'start': (4, 19), 'goal': (4, 0)},
        {'id': 2, 'start': (7, 19), 'goal': (7, 0)},
        {'id': 3, 'start': (10, 19), 'goal': (10, 0)},
        {'id': 4, 'start': (13, 19), 'goal': (13, 0)},
    ]
    assigned_nodes_fixed: Set[Node] = set()
    valid_tasks = True
    agv_ids_in_scenario: List[int] = [] # 收集场景中的 AGV ID
    for t_data in fixed_task_data:
        task = Task(agv_id=t_data['id'], start_node=t_data['start'], goal_node=t_data['goal'])
        if not grid_map.is_valid(*task.start_node): print(f"错误(固定任务): AGV {task.agv_id} 起点 {task.start_node} 无效。"); valid_tasks = False; break
        if not grid_map.is_valid(*task.goal_node): print(f"错误(固定任务): AGV {task.agv_id} 终点 {task.goal_node} 无效。"); valid_tasks = False; break
        if task.start_node in assigned_nodes_fixed: print(f"错误(固定任务): AGV {task.agv_id} 起点 {task.start_node} 冲突。"); valid_tasks = False; break
        assigned_nodes_fixed.add(task.start_node)
        tasks_fixed_s1.append(task)
        agv_ids_in_scenario.append(task.agv_id)

    if not valid_tasks:
        print("错误: 手动定义的场景 1 任务存在无效或冲突，无法加载。"); return None
    print(f"手动定义场景 1 任务完成，共 {len(tasks_fixed_s1)} 个任务。")

    # --- 生成 AGV 速度字典 ---
    agv_speeds: Dict[int, float] = {}
    if assign_speeds:
        print("  为固定场景 AGV 分配速度...")
        num_agvs_fixed = len(agv_ids_in_scenario)
        agv_ids_shuffled = list(agv_ids_in_scenario); random.shuffle(agv_ids_shuffled)
        num_fast = num_agvs_fixed // 2
        for i, agv_id in enumerate(agv_ids_shuffled):
            agv_speeds[agv_id] = fast_speed if i < num_fast else slow_speed
        print(f"  速度分配 (示例): {agv_speeds}")
    else:
        print("  跳过速度分配，所有 AGV 将使用默认速度。")
        # 可以选择返回空字典，让调用者处理，或填充默认速度
        for agv_id in agv_ids_in_scenario:
            agv_speeds[agv_id] = avg_speed # 填充平均/默认速度

    # --- 返回地图、任务和速度字典 ---
    return grid_map, tasks_fixed_s1, agv_speeds

# --- 随机算例生成 (场景 2) (v15 - 返回速度字典) ---
def generate_scenario_2_instance(
    json_map_file: str = "Map.json",
    num_agvs: int = 5,
    expansion_radius: int = 0,
    # 新增: 用于演示的速度参数
    assign_speeds: bool = True,
    avg_speed: float = 1.0,
    fast_speed: float = 1.2,
    slow_speed: float = 0.8
) -> Optional[Tuple['GridMap', List[Task], Dict[int, float]]]:
    """
    生成场景 2 实例。
    返回 GridMap, 任务列表, 以及 AGV 速度字典。
    """
    print(f"生成随机算例场景 2 实例 (地图: {json_map_file}, agvs={num_agvs}, expansion={expansion_radius})...")
    if 'GridMap' not in globals() or 'Task' not in globals() or GridMap is None or Task is None:
        print("错误: GridMap 或 Task 类未正确加载。"); return None

    map_data = _load_map_from_json(json_map_file)
    if map_data is None: return None
    width, height, grid_data = map_data

    try:
        grid_map = GridMap(grid_data=grid_data, expansion_radius=expansion_radius)
    except Exception as e:
        print(f"错误: 创建场景 2 地图失败: {e}"); traceback.print_exc(); return None

    # --- 调用包装器函数生成任务 ---
    tasks = generate_random_tasks(grid_map, grid_data, num_agvs)
    if not tasks:
        print("错误: 生成随机任务失败。"); return None

    # --- 生成 AGV 速度字典 ---
    agv_speeds: Dict[int, float] = {}
    agv_ids_in_scenario = [task.agv_id for task in tasks] # 从生成的任务获取 AGV ID
    if assign_speeds:
        print("  为随机场景 AGV 分配速度...")
        num_agvs_actual = len(agv_ids_in_scenario)
        agv_ids_shuffled = list(agv_ids_in_scenario); random.shuffle(agv_ids_shuffled)
        num_fast = num_agvs_actual // 2
        for i, agv_id in enumerate(agv_ids_shuffled):
            agv_speeds[agv_id] = fast_speed if i < num_fast else slow_speed
        print(f"  速度分配 (示例): {agv_speeds}")
    else:
        print("  跳过速度分配，所有 AGV 将使用默认速度。")
        for agv_id in agv_ids_in_scenario:
            agv_speeds[agv_id] = avg_speed # 填充平均/默认速度

    print(f"成功生成地图 (来自 {json_map_file}) 和 {len(tasks)} 个随机任务。")
    # --- 返回地图、任务和速度字典 ---
    return grid_map, tasks, agv_speeds

# --- 示例用法 (v15 - 处理返回的速度字典) ---
if __name__ == '__main__':
    print("--- Instance Generator 测试 (v15 - 返回 AGV 速度信息) ---")
    test_expansion_radius = 0
    map_json_path = "Map.json"

    if not Path(map_json_path).is_file():
        print(f"\n错误: 找不到地图文件 '{map_json_path}'。测试无法继续。")
    else:
        # --- 测试加载固定场景 ---
        print(f"\n[1] 测试加载固定场景 1 (地图: {map_json_path}, 膨胀半径={test_expansion_radius})...")
        # 调用新版函数，接收速度字典
        scenario1_data = load_fixed_scenario_1(
            json_map_file=map_json_path,
            expansion_radius=test_expansion_radius,
            assign_speeds=True # 在这里演示分配速度
        )
        if scenario1_data:
            map1, tasks1, speeds1 = scenario1_data # 解包三个返回值
            print(f"  地图: {map1}")
            print(f"  固定任务数量: {len(tasks1)}")
            if tasks1: print(f"  第一个固定任务: {tasks1[0]}")
            print(f"  返回的 AGV 速度字典 (固定场景): {speeds1}") # 打印速度字典
        else: print("  加载固定场景 1 失败。")

        # --- 测试生成场景 2 (使用 Tasks.py-v5 逻辑) ---
        num_agvs_s2 = 10 # 测试生成 10 个 AGV 的任务
        print(f"\n[2] 测试生成随机场景 2 实例 (地图: {map_json_path}, AGVs={num_agvs_s2}, 膨胀半径={test_expansion_radius})...")
        # 调用新版函数，接收速度字典
        scenario2_instance = generate_scenario_2_instance(
            json_map_file=map_json_path,
            num_agvs=num_agvs_s2,
            expansion_radius=test_expansion_radius,
            assign_speeds=True # 在这里演示分配速度
        )
        if scenario2_instance:
            map2, tasks2, speeds2 = scenario2_instance # 解包三个返回值
            print(f"  地图: {map2}")
            print(f"  随机任务数量: {len(tasks2)}")
            if tasks2:
                print(f"  第一个随机任务: {tasks2[0]}")
                print("  前 5 个随机任务:")
                for i, t in enumerate(tasks2[:5]):
                    print(f"    AGV {t.agv_id}: {t.start_node} -> {t.goal_node}")
            print(f"  返回的 AGV 速度字典 (随机场景): {speeds2}") # 打印速度字典
        else:
            print("  生成随机场景 2 实例失败。")
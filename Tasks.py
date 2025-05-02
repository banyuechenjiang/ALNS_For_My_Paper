# Tasks.py (v6 - Orthogonal Access Points Only, No Comments)
import random
from typing import List, Tuple, Set, Optional, Dict, Sequence
from enum import Enum, auto
import traceback

try:
    from Map import GridMap, Node
    from DataTypes import Task
except ImportError as e:
    print(f"错误: 导入 Tasks 依赖项失败: {e}")
    GridMap = type('GridMap', (object,), {'width': 0, 'height': 0, 'is_valid': lambda s, x, y: False})
    Node = Tuple[int, int]
    Task = type('Task', (object,), {'agv_id': 0, 'start_node': (0,0), 'goal_node': (0,0)})

class TaskType(Enum):
    GOTO_STORAGE_FROM_DEPOT = auto()
    GOTO_PICKING_FROM_STORAGE = auto()
    GOTO_DEPOT_FROM_PICKING = auto()

def _identify_zone_access_points(
    grid_data: List[List[int]],
    grid_map: GridMap
) -> Dict[str, Set[Node]]:
    access_points: Dict[str, Set[Node]] = {'2': set(), '3': set(), '4': set()}
    rows = len(grid_data); cols = len(grid_data[0]) if rows > 0 else 0
    if rows == 0 or cols == 0: return access_points

    for r in range(rows):
        for c in range(cols):
            node_type = grid_data[r][c]
            zone_key = str(node_type)
            if zone_key in access_points:
                # --- v6 Modification: Only check orthogonal neighbors ---
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # Orthogonal moves only
                # --- End v6 Modification ---
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid_data[nr][nc] == 0:
                        neighbor_node: Node = (nc, nr)
                        if grid_map.is_valid(*neighbor_node):
                            access_points[zone_key].add(neighbor_node)
    return access_points

def _find_valid_start_goal_pair(
    start_zone_key: str,
    goal_zone_key: str,
    available_nodes: Dict[str, List[Node]],
    assigned_start_nodes: Set[Node],
    assigned_unique_goal_nodes: Dict[int, Set[Node]],
    ensure_unique_starts: bool = True
) -> Tuple[Optional[Node], Optional[Node]]:
    start_nodes_pool = available_nodes.get(start_zone_key)
    goal_nodes_pool = available_nodes.get(goal_zone_key)

    if not start_nodes_pool or not goal_nodes_pool:
        return None, None

    potential_starts = list(start_nodes_pool)
    random.shuffle(potential_starts)

    for start_node in potential_starts:
        if ensure_unique_starts and start_node in assigned_start_nodes:
            continue

        potential_goals = list(goal_nodes_pool)
        random.shuffle(potential_goals)
        goal_zone_type = int(goal_zone_key)
        needs_goal_unique = goal_zone_type in assigned_unique_goal_nodes

        for goal_node in potential_goals:
            if start_node == goal_node:
                continue
            goal_unique_ok = (not needs_goal_unique) or (goal_node not in assigned_unique_goal_nodes[goal_zone_type])
            if goal_unique_ok:
                if ensure_unique_starts:
                    assigned_start_nodes.add(start_node)
                if needs_goal_unique:
                    assigned_unique_goal_nodes[goal_zone_type].add(goal_node)
                return start_node, goal_node
    return None, None

def generate_tasks_goods_to_person_v6( # Renamed to v6
    grid_map: GridMap,
    grid_data: List[List[int]],
    num_agvs: int,
    ensure_unique_starts: bool = True
) -> List[Task]:
    if num_agvs <= 0: return []
    print(f"\n--- 开始生成任务 (v6 - 仅正交接入点) for {num_agvs} AGVs ---")

    access_points = _identify_zone_access_points(grid_data, grid_map) # Uses modified v6 function
    required_zones = ['2', '3', '4']
    for zone in required_zones:
        if not access_points.get(zone):
            print(f"错误: 地图中未找到区域 {zone} 的有效正交接入点，无法生成 G2P 任务。")
            return []
        print(f"  区域 {zone} 找到 {len(access_points[zone])} 个正交接入点。") # Updated message

    available_nodes: Dict[str, List[Node]] = {
        k: random.sample(list(v), len(v)) for k, v in access_points.items()
    }

    assigned_start_nodes: Set[Node] = set()
    assigned_unique_goal_nodes: Dict[int, Set[Node]] = {3: set(), 4: set()}
    tasks: List[Task] = []
    agv_ids_available = list(range(num_agvs))
    random.shuffle(agv_ids_available)

    num_core_types = 3
    base_count = num_agvs // num_core_types
    remainder = num_agvs % num_core_types
    target_counts = {
        TaskType.GOTO_STORAGE_FROM_DEPOT: base_count,
        TaskType.GOTO_PICKING_FROM_STORAGE: base_count,
        TaskType.GOTO_DEPOT_FROM_PICKING: base_count
    }
    types_for_remainder = list(target_counts.keys())
    for i in range(remainder):
        target_counts[types_for_remainder[i]] += 1

    print(f"  目标任务数: 4->2 ({target_counts[TaskType.GOTO_STORAGE_FROM_DEPOT]}), "
          f"2->3 ({target_counts[TaskType.GOTO_PICKING_FROM_STORAGE]}), "
          f"3->4 ({target_counts[TaskType.GOTO_DEPOT_FROM_PICKING]})")

    current_counts = {t_type: 0 for t_type in target_counts}
    unassigned_agv_ids = list(agv_ids_available)

    task_types_priority = list(target_counts.keys())
    random.shuffle(task_types_priority)

    agv_idx = 0
    while agv_idx < len(agv_ids_available):
        agv_id = agv_ids_available[agv_idx]
        assigned_this_agv = False
        for task_type in task_types_priority:
            if current_counts[task_type] < target_counts[task_type]:
                start_key, goal_key = '', ''
                if task_type == TaskType.GOTO_STORAGE_FROM_DEPOT:   start_key, goal_key = '4', '2'
                elif task_type == TaskType.GOTO_PICKING_FROM_STORAGE: start_key, goal_key = '2', '3'
                elif task_type == TaskType.GOTO_DEPOT_FROM_PICKING:   start_key, goal_key = '3', '4'
                else: continue

                start_node, goal_node = _find_valid_start_goal_pair(
                    start_key, goal_key, available_nodes,
                    assigned_start_nodes, assigned_unique_goal_nodes,
                    ensure_unique_starts
                )

                if start_node and goal_node:
                    tasks.append(Task(agv_id, start_node, goal_node))
                    current_counts[task_type] += 1
                    unassigned_agv_ids.remove(agv_id)
                    assigned_this_agv = True
                    break

        if not assigned_this_agv:
            print(f"  警告: 未能为 AGV {agv_id} 在当前轮次找到合适的任务 (可能因接入点不足)。")
            agv_idx += 1
        else:
             agv_idx += 1

    tasks.sort(key=lambda t: t.agv_id)
    print(f"--- 任务生成完成 (v6 - 正交接入点) ---")
    print(f"  成功生成 {len(tasks)} 个任务。")
    if unassigned_agv_ids:
        print(f"  警告: {len(unassigned_agv_ids)} 个 AGV 未能成功分配任务: {sorted(unassigned_agv_ids)}")
    final_counts_str = ", ".join([f"{tt.name}: {current_counts[tt]}" for tt in target_counts])
    print(f"  最终任务类型分布: {final_counts_str}")

    return tasks

if __name__ == '__main__':
    print("--- Tasks.py 测试 (v6 - 仅正交接入点) ---")
    import json
    from pathlib import Path

    map_json_path = "Map.json"
    grid_map_instance = None
    grid_data_instance = None
    try:
        json_file = Path(map_json_path)
        if json_file.is_file():
            with open(json_file, 'r', encoding='utf-8') as f: map_json_content = json.load(f)
            if "grid" in map_json_content:
                grid_data_instance = map_json_content["grid"]
                # Need GridMap from Map.py (v10+)
                from Map import GridMap
                grid_map_instance = GridMap(grid_data=grid_data_instance, expansion_radius=1) # Use expansion
                print(f"成功从 '{map_json_path}' 加载地图并初始化 GridMap (膨胀半径=1)。")
            else: print(f"错误: '{map_json_path}' 缺少 'grid' 数据。")
        else: print(f"错误: 地图文件 '{map_json_path}' 不存在。")
    except ImportError: print("错误: 无法导入 Map.py 中的 GridMap。")
    except Exception as e: print(f"错误: 加载地图或初始化 GridMap 时出错: {e}"); traceback.print_exc()

    if grid_map_instance and grid_data_instance:
        num_test_agvs = 8
        try:
            # Call the v6 function
            generated_tasks = generate_tasks_goods_to_person_v6(
                grid_map_instance,
                grid_data_instance,
                num_test_agvs
            )

            if generated_tasks:
                print("\n--- 生成的任务列表 ---")
                task_type_counts_check = {TaskType.GOTO_STORAGE_FROM_DEPOT: 0, TaskType.GOTO_PICKING_FROM_STORAGE: 0, TaskType.GOTO_DEPOT_FROM_PICKING: 0}
                access_points_check = _identify_zone_access_points(grid_data_instance, grid_map_instance)

                for task in generated_tasks:
                    start_zone = '?'; goal_zone = '?'
                    for zone, points in access_points_check.items():
                        if task.start_node in points: start_zone = zone
                        if task.goal_node in points: goal_zone = zone
                    task_flow = f"{start_zone}->{goal_zone}"
                    task_type_guess = "未知"
                    if task_flow == "4->2": task_type_guess = TaskType.GOTO_STORAGE_FROM_DEPOT.name; task_type_counts_check[TaskType.GOTO_STORAGE_FROM_DEPOT] += 1
                    elif task_flow == "2->3": task_type_guess = TaskType.GOTO_PICKING_FROM_STORAGE.name; task_type_counts_check[TaskType.GOTO_PICKING_FROM_STORAGE] += 1
                    elif task_flow == "3->4": task_type_guess = TaskType.GOTO_DEPOT_FROM_PICKING.name; task_type_counts_check[TaskType.GOTO_DEPOT_FROM_PICKING] += 1
                    print(f"  AGV {task.agv_id}: {task.start_node} -> {task.goal_node} (推断类型: {task_flow} / {task_type_guess})")
                print("\n--- 任务类型统计 (基于推断) ---")
                for tt, count in task_type_counts_check.items(): print(f"  {tt.name}: {count}")
            else: print("\n未能生成任何任务。")
        except Exception as e: print(f"错误: 调用任务生成函数时出错: {e}"); traceback.print_exc()
    else: print("\n无法继续测试，因为地图加载或 GridMap 初始化失败。")
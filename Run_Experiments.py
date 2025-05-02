# Run_Experiments.py (v21 - ALNS Only, Visualize Best Instance)

import random
import time
import csv
import os
import sys
import traceback
import pathlib
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import copy
from collections import deque, defaultdict, Counter # 确保导入

import pathlib
# --- 修改: 从 InstanceGenerator 导入辅助函数，从 Tasks 导入 v6 函数 ---
from InstanceGenerator import _load_map_from_json
from Tasks import generate_tasks_goods_to_person_v6
# --- ---------------------------------------------------------- ---
from DataTypes import Task, Path as AgentPath, Solution, CostDict, Node, TimeStep, State, DynamicObstacles, calculate_tij
from Planner import TWAStarPlanner
from ALNS import ALNS # 需要 v44+
# --- 移除 Benchmark 导入 ---
# from Benchmark import PrioritizedPlanner
# --- --------------------- ---
from Map import GridMap # 需要 v10+

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    _visual_libs_available = True
    try:
        # 尝试设置支持中文的字体
        font_list = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'PingFang SC', 'sans-serif']
        matplotlib.rcParams['font.sans-serif'] = font_list
        matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    except Exception as font_e:
        print(f"警告: 设置中文字体时可能出现问题: {font_e}")
except ImportError:
    _visual_libs_available = False
    print("警告: Matplotlib 未安装，可视化功能将被禁用。")

# --- 常量定义 ---
MAP_FILE = "Map.json"
NUM_AGVS = 8
NUM_INSTANCES = 3 # 实例数调整为 3
EXPANSION_RADIUS = 0
# --- 更新: 结果目录和文件名前缀 ---
RESULTS_DIR = f"experiment_results_v21_agv{NUM_AGVS}_taskV6_alnsOnly_visBest"
INSTANCE_PREFIX = f"map_agv{NUM_AGVS}_taskV6_alnsOnly_visBest"

AVG_SPEED = 1.0; FAST_SPEED = 1.3; SLOW_SPEED = 0.7

COMMON_ALGORITHM_PARAMS = {
    'max_time': 1000, 'cost_weights': (1.0, 0.3, 0.8),
    'delta_step': 1.0, 'v': AVG_SPEED,
}
# --- 使用调整后的 ALNS 参数 (鼓励探索, 提高冲突容忍) ---
ALNS_SPECIFIC_PARAMS = {
    'alns_max_iterations': 800, 'alns_initial_temp': 450.0, 'alns_cooling_rate': 0.995,
    'alns_segment_size': 100, 'alns_weight_update_rate': 0.15,
    'alns_sigma1': 35.0, 'alns_sigma2': 18.0, 'alns_sigma3': 5.0, 
    'alns_removal_percentage_min': 0.15, 'alns_removal_percentage_max': 0.45, # 移除范围
    'alns_regret_k': 4, 'alns_regret_max_attempts': 15, # 增强 regret
    'alns_no_improvement_limit': 150, 'alns_no_improvement_bbox_disable': 50, #  BBox 禁用阈值
    'conflict_history_decay': 0.8, 'buffer': 1,
    'alns_planner_time_limit_factor': 5.0, 'alns_regret_planner_time_limit_abs': 0.15,
    'wait_threshold': 6, 'deadlock_max_wait': 200, 'edge_wait_threshold': 4, # 大幅增加死锁等待容忍
    'alns_verbose_output': False, 'alns_debug_weights': False,
    'alns_record_history': True, 'alns_plot_convergence': True,
}
# --- 移除 BENCHMARK_SPECIFIC_PARAMS ---

# --- 实验分组定义 (仅 ALNS) ---
EXPERIMENTAL_GROUPS = [
    {"name": "Baseline", "desc": "同质+ID优先", "use_hetero": False, "prio": "id"},
    {"name": "Control1", "desc": "异构+ID优先", "use_hetero": True, "prio": "id"},
    {"name": "Experimental", "desc": "异构+速度优先", "use_hetero": True, "prio": "speed"}
]

# --- 辅助函数 (保持不变) ---
def create_results_dir(dir_path: str): pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
def calculate_makespan(solution: Optional[Solution]) -> TimeStep:
    if not solution: return -1
    max_t = -1
    for path in solution.values():
        if isinstance(path, AgentPath) and path.sequence: max_t = max(max_t, path.get_makespan())
    return max_t
def generate_simple_initial_solution(grid_map: GridMap, tasks: List[Task], planner: TWAStarPlanner, cost_weights: Tuple[float, float, float], agv_speeds: Dict[int, float], delta_step: float, max_time: int, planner_time_limit: Optional[float]) -> Tuple[Optional[Solution], Optional[CostDict]]:
    print("    生成简化初始解 (独立规划)...", end="", flush=True)
    initial_solution: Solution = {}; alpha, beta, gamma_wait = cost_weights
    total_cost_dict: CostDict = {'total': 0.0, 'travel': 0.0, 'turn': 0.0, 'wait': 0.0}
    inf_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
    all_success = True; start_g = time.perf_counter()
    for task in tasks:
        agv_id = task.agv_id; empty_dynamic_obstacles: DynamicObstacles = {}; path: Optional[AgentPath] = None
        agv_speed = agv_speeds.get(agv_id, 1.0)
        if not grid_map.is_valid(*task.start_node) or not grid_map.is_valid(*task.goal_node): print(f"\n      错误: AGV {agv_id} 任务 {task.start_node}->{task.goal_node} 起点或终点无效。"); all_success = False; break
        try: path = planner.plan(grid_map=grid_map, task=task, dynamic_obstacles=empty_dynamic_obstacles, max_time=max_time, cost_weights=cost_weights, agv_speed=agv_speed, delta_step=delta_step, start_time=0, time_limit=planner_time_limit)
        except Exception as e: print(f"\n      错误: 调用 Planner 为 AGV {agv_id} 任务 {task.start_node}->{task.goal_node} 时发生异常: {e}"); traceback.print_exc(); all_success = False; break
        if path and path.sequence:
            initial_solution[agv_id] = path
            try:
                path_cost_dict = path.get_cost(grid_map, alpha, beta, gamma_wait, agv_speed, delta_step)
                if path_cost_dict.get('total', float('inf')) == float('inf'): print(f"\n      错误: AGV {agv_id} 任务 {task.start_node}->{task.goal_node} 成本为 Inf。"); all_success = False; break
                for key in total_cost_dict: total_cost_dict[key] += path_cost_dict.get(key, 0.0)
            except Exception as e: print(f"\n      错误: 计算 AGV {agv_id} 任务 {task.start_node}->{task.goal_node} 成本时出错: {e}"); all_success = False; break
        else: print(f"\n      错误：AGV {agv_id} 任务 {task.start_node}->{task.goal_node} 独立规划失败！"); all_success = False; break
    duration_g = time.perf_counter() - start_g
    if all_success and len(initial_solution) == len(tasks):
        cost_val = total_cost_dict.get('total', float('inf'))
        if cost_val != float('inf'): print(f" 成功 (耗时 {duration_g:.2f}s, 组合成本 {cost_val:.2f})"); return initial_solution, total_cost_dict
        else: print(f" 失败 (最终成本为 Inf, 耗时 {duration_g:.2f}s)"); return None, inf_dict
    else: print(f" 失败 (耗时 {duration_g:.2f}s)"); return None, inf_dict
def visualize_paths(grid_data: Optional[List[List[int]]], solution: Optional[Solution], output_filename: str, title_suffix: str):
    if not _visual_libs_available: print("警告: Matplotlib 不可用，跳过可视化。"); return
    if solution is None: print("警告: 解决方案为 None，跳过可视化。"); return
    if grid_data is None: print("警告: 地图 grid_data 为 None，跳过可视化。"); return
    rows, cols = len(grid_data), len(grid_data[0]) if grid_data else 0
    if rows == 0 or cols == 0: print("警告: 地图数据为空，跳过可视化。"); return
    fig, ax = plt.subplots(figsize=(max(8, cols * 0.6), max(6, rows * 0.6)))
    map_node_colors = {0: 'white', 1: 'black', 2: 'lightgrey', 3: 'lightblue', 4: 'lightgreen'}
    node_types_desc = {0: "通道", 1: "障碍物", 2: "存储位", 3: "拣选站", 4: "停靠区"}
    map_legend_elements = []
    for node_type in sorted(map_node_colors.keys()):
        color = map_node_colors[node_type]; map_legend_elements.append(mpatches.Patch(color=color, label=f"{node_type}: {node_types_desc[node_type]}"))
        for r in range(rows):
            for c in range(cols):
                if grid_data[r][c] == node_type: ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='grey', linewidth=0.3))
    num_agvs = len(solution); cmap = plt.get_cmap('tab10', max(10, num_agvs)); agv_legend_elements = []
    for i, (agv_id, path) in enumerate(sorted(solution.items())):
        if not isinstance(path, AgentPath) or not path.sequence: continue
        path_nodes = [state[0] for state in path.sequence]; x_coords, y_coords = zip(*path_nodes); agv_color = cmap(i % cmap.N)
        line, = ax.plot(x_coords, y_coords, marker='.', markersize=4, linestyle='-', linewidth=1.2, color=agv_color, alpha=0.8)
        start_node, goal_node = path_nodes[0], path_nodes[-1]
        ax.plot(start_node[0], start_node[1], 'o', markersize=7, color=agv_color, markeredgecolor='black', label=f'_nolegend_start_{agv_id}')
        ax.plot(goal_node[0], goal_node[1], 's', markersize=7, color=agv_color, markeredgecolor='black', label=f'_nolegend_goal_{agv_id}')
        agv_legend_elements.append(Line2D([0], [0], color=agv_color, lw=2, label=f'AGV {agv_id}'))
    ax.set_xlim(-0.5, cols - 0.5); ax.set_ylim(-0.5, rows - 0.5); ax.set_xticks(np.arange(-0.5, cols, 1), minor=True); ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.set_xticks(np.arange(0, cols, 5)); ax.set_yticks(np.arange(0, rows, 5)); ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
    ax.set_xlabel("列"); ax.set_ylabel("行"); ax.set_title(f"最终路径 ({title_suffix})", fontsize=12); ax.set_aspect('equal', adjustable='box'); ax.invert_yaxis()
    combined_legend_elements = map_legend_elements + agv_legend_elements
    lgd = ax.legend(handles=combined_legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0., title="图例", fontsize=8)
    try: plt.subplots_adjust(right=0.78); plt.savefig(output_filename, dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight'); print(f"      路径可视化图像已保存到: {os.path.abspath(output_filename)}")
    except Exception as e: print(f"      错误: 保存可视化图像时出错: {e}")
    finally: plt.close(fig)

# --- 主执行逻辑 ---
if __name__ == "__main__":
    print("===================================================================")
    print(f"=== 仓储 AGV 路径规划对比实验 (v21 - 仅 ALNS, 仅可视化最佳实例, AGV={NUM_AGVS}) ===")
    print("===================================================================")

    map_file_path = pathlib.Path(MAP_FILE)
    if not map_file_path.is_file(): print(f"错误: 地图文件 '{MAP_FILE}' 未找到！"); sys.exit(1)

    create_results_dir(RESULTS_DIR)
    alns_details_subdir = os.path.join(RESULTS_DIR, "alns_run_details")
    create_results_dir(alns_details_subdir)
    visualization_subdir = os.path.join(RESULTS_DIR, "visualizations")
    create_results_dir(visualization_subdir) # 可视化目录仍然创建，但只放最佳实例的图

    all_results: List[Dict[str, Any]] = []
    instance_run_details: Dict[int, Dict[str, Any]] = defaultdict(dict) # 存储每个实例的运行详情

    print(f"\n开始运行 {NUM_INSTANCES} 个随机任务实例 (使用 Tasks v6 生成单任务)...")
    total_start_time = time.perf_counter()

    # --- 外层: 实例循环 ---
    for i in range(NUM_INSTANCES):
        instance_num = i + 1
        print(f"\n--- 处理实例 {instance_num}/{NUM_INSTANCES} ---")
        instance_start_time = time.perf_counter()
        instance_id_base = f"{INSTANCE_PREFIX}_inst{instance_num}"
        grid_map_instance: Optional[GridMap] = None; tasks_instance: Optional[List[Task]] = None
        agv_speeds_hetero: Dict[int, float] = {}; agv_speeds_homo: Dict[int, float] = {}
        grid_data_for_vis: Optional[List[List[int]]] = None
        try:
            # 加载地图和生成任务
            map_data = _load_map_from_json(MAP_FILE); width, height, grid_data = map_data
            grid_data_for_vis = grid_data; grid_map_instance = GridMap(grid_data=grid_data, expansion_radius=EXPANSION_RADIUS)
            tasks_instance = generate_tasks_goods_to_person_v6(grid_map=grid_map_instance, grid_data=grid_data, num_agvs=NUM_AGVS)
            if not tasks_instance or len(tasks_instance) < NUM_AGVS: raise ValueError(f"任务生成失败 (v6), 预期 {NUM_AGVS}, 实际 {len(tasks_instance) if tasks_instance else 0}")
            print(f"  加载地图并成功生成 {len(tasks_instance)} 个任务 (Tasks v6)。")
            # 生成速度字典
            agv_ids_instance = [t.agv_id for t in tasks_instance]; agv_speeds_homo = {agv_id: AVG_SPEED for agv_id in agv_ids_instance}
            agv_ids_shuffled = list(agv_ids_instance); random.shuffle(agv_ids_shuffled); num_fast = len(agv_ids_instance) // 2
            for idx, agv_id in enumerate(agv_ids_shuffled): agv_speeds_hetero[agv_id] = FAST_SPEED if idx < num_fast else SLOW_SPEED
            # 存储实例详情用于可能的后续可视化
            instance_run_details[instance_num]['grid_data'] = grid_data_for_vis
        except Exception as e: print(f"  错误: 实例 {instance_num} 数据生成失败: {e}，跳过。"); traceback.print_exc(); continue

        # --- 内层: 实验分组循环 (仅 ALNS) ---
        for group_config in EXPERIMENTAL_GROUPS:
            group_name = group_config["name"]; print(f"\n  运行实验组: {group_name} ({group_config['desc']})")
            current_agv_speeds = agv_speeds_hetero if group_config["use_hetero"] else agv_speeds_homo
            current_priority = group_config["prio"]; speed_type_desc = "异构" if group_config["use_hetero"] else "同质"

            # --- 运行 ALNS ---
            alns_solution: Optional[Solution] = None; alns_cost: Optional[CostDict] = None
            alns_duration: Optional[float] = None; alns_makespan: Optional[TimeStep] = None
            init_source = "None"

            print(f"    运行 ALNS (速度: {speed_type_desc}, 优先级: {current_priority})...", end="", flush=True)
            alns_start_time = time.perf_counter(); initial_solution_for_alns: Optional[Solution] = None

            # --- 始终尝试生成简单初始解 ---
            print(" 尝试生成简单独立解...", end="", flush=True)
            planner_simple = TWAStarPlanner() # 需要一个独立的 planner 实例
            simple_sol, _ = generate_simple_initial_solution(
                grid_map_instance, tasks_instance, planner_simple,
                COMMON_ALGORITHM_PARAMS['cost_weights'], current_agv_speeds,
                COMMON_ALGORITHM_PARAMS['delta_step'], COMMON_ALGORITHM_PARAMS['max_time'],
                ALNS_SPECIFIC_PARAMS.get('alns_planner_time_limit_factor', 5.0) * 0.5 # 使用较短的时间限制
            )
            if simple_sol is not None:
                initial_solution_for_alns = simple_sol
                init_source = "SimpleIndependent"
                print(" 成功.", end="", flush=True)
            else:
                print(" 失败 (简单解也无法生成).")
                init_source = "Failed"
            # --- -------------------------- ---

            if initial_solution_for_alns is not None:
                try:
                    planner_alns = TWAStarPlanner() # ALNS 使用自己的 planner 实例
                    alns_instance_id = f"{instance_id_base}_{group_name}_ALNS"
                    init_params = {**COMMON_ALGORITHM_PARAMS, **ALNS_SPECIFIC_PARAMS}
                    init_params['skip_internal_initial_solution'] = True # 明确告知 ALNS 跳过内部生成

                    alns_instance = ALNS(
                        grid_map=grid_map_instance, tasks=tasks_instance, planner=planner_alns,
                        instance_identifier=alns_instance_id, results_dir=alns_details_subdir,
                        agv_speeds=current_agv_speeds, priority_strategy=current_priority,
                        **init_params
                    )
                    # 运行 ALNS，传入外部初始解
                    alns_solution, alns_duration_run, alns_cost = alns_instance.run(external_initial_solution=initial_solution_for_alns)
                    alns_duration = time.perf_counter() - alns_start_time

                    if alns_solution and alns_cost and alns_cost.get('total', float('inf')) != float('inf'):
                        alns_makespan = calculate_makespan(alns_solution)
                        print(f" 成功 (耗时 {alns_duration:.2f}s, Cost {alns_cost['total']:.2f}, Makespan {alns_makespan if alns_makespan != -1 else 'N/A'})")
                        # 存储 ALNS 运行详情用于可能的后续可视化
                        instance_run_details[instance_num][f"{group_name}_ALNS"] = {'solution': alns_solution, 'desc': f"Inst {instance_num} ALNS ({group_config['desc']})"}
                    else:
                        print(f" 失败 (耗时 {alns_duration:.2f}s)")
                except Exception as e:
                    alns_duration = time.perf_counter() - alns_start_time
                    print(f" 运行时异常: {e} (耗时 {alns_duration:.2f}s)")
                    traceback.print_exc()
            else:
                # 如果连简单初始解都生成失败
                alns_duration = time.perf_counter() - alns_start_time
                print(f" 失败 (无可用初始解，耗时 {alns_duration:.2f}s)")

            # 记录结果 (无论成功失败)
            result_entry_alns = {
                "instance": instance_num, "group": group_name, "algorithm": "ALNS",
                "priority": current_priority, "speed_type": speed_type_desc,
                "total_cost": alns_cost['total'] if alns_cost else None,
                "travel_cost": alns_cost['travel'] if alns_cost else None,
                "turn_cost": alns_cost['turn'] if alns_cost else None,
                "wait_cost": alns_cost['wait'] if alns_cost else None,
                "cpu_time": alns_duration, "makespan": alns_makespan,
                "success": alns_solution is not None,
                "init_source": init_source
            }
            all_results.append(result_entry_alns)

        instance_duration = time.perf_counter() - instance_start_time
        print(f"--- 实例 {instance_num} 处理完成 (耗时 {instance_duration:.2f}s) ---")

    # --- 所有实例处理完毕 ---
    total_run_duration = time.perf_counter() - total_start_time
    print(f"\n所有 {NUM_INSTANCES} 个实例处理完毕，总耗时: {total_run_duration:.2f}s")

    # --- 结果聚合与分析 ---
    print("\n--- 实验结果聚合与分析 ---")
    if not all_results: print("错误: 没有收集到任何实验结果。"); sys.exit(1)

    # --- 查找并展示全局最佳 ALNS 结果 ---
    successful_alns_runs = [r for r in all_results if r.get('success') and r.get('algorithm') == 'ALNS']
    best_alns_run_overall: Optional[Dict[str, Any]] = None
    best_instance_num: Optional[int] = None # 存储最佳实例编号

    if successful_alns_runs:
        def get_sort_key(run_result):
            cost = run_result.get('total_cost', float('inf')); cost = float('inf') if cost is None else cost
            makespan = run_result.get('makespan', float('inf')); makespan = float('inf') if makespan is None or makespan == -1 else makespan
            return (cost, makespan)
        best_alns_run_overall = min(successful_alns_runs, key=get_sort_key)
        best_instance_num = best_alns_run_overall.get('instance') # 获取最佳实例编号
        print("--- 全局最佳 ALNS 结果 (基于最低总成本，次优为最短 Makespan) ---")
        print(f"  来源实例: Instance {best_alns_run_overall.get('instance', 'N/A')}")
        print(f"  实验组:   {best_alns_run_overall.get('group', 'N/A')}")
        # print(f"  算法:     ALNS") # 明确是 ALNS
        print(f"  优先级:   {best_alns_run_overall.get('priority', 'N/A')}")
        print(f"  速度类型: {best_alns_run_overall.get('speed_type', 'N/A')}")
        print("-" * 60)
        print(f"  总成本:   {best_alns_run_overall.get('total_cost', 'N/A'):.2f}")
        print(f"    - 行驶: {best_alns_run_overall.get('travel_cost', 'N/A'):.2f}")
        print(f"    - 转弯: {best_alns_run_overall.get('turn_cost', 'N/A'):.2f}")
        print(f"    - 等待: {best_alns_run_overall.get('wait_cost', 'N/A'):.2f}")
        print(f"  Makespan: {best_alns_run_overall.get('makespan', 'N/A')}")
        print(f"  CPU 时间: {best_alns_run_overall.get('cpu_time', 'N/A'):.3f} s")
        print(f"  初始解来源: {best_alns_run_overall.get('init_source', 'N/A')}")
        print("-" * 60)
    else: print("--- 未找到任何成功的 ALNS 运行结果 ---")

    # --- 仅可视化最佳实例的 ALNS 结果 ---
    if best_instance_num is not None and best_instance_num in instance_run_details:
        print(f"\n--- 可视化最佳实例 (Instance {best_instance_num}) 的 ALNS 结果 ---")
        grid_data_best_vis = instance_run_details[best_instance_num].get('grid_data')
        if grid_data_best_vis:
            best_instance_vis_dir = os.path.join(visualization_subdir, f"{INSTANCE_PREFIX}_inst{best_instance_num}_BEST")
            create_results_dir(best_instance_vis_dir) # 创建最佳实例的可视化目录

            # 查找该实例下的所有 ALNS 运行详情
            alns_runs_for_best_instance = {k: v for k, v in instance_run_details[best_instance_num].items() if '_ALNS' in k}

            if alns_runs_for_best_instance:
                for run_key, run_data in alns_runs_for_best_instance.items():
                    solution_to_vis = run_data.get('solution')
                    desc_to_vis = run_data.get('desc')
                    if solution_to_vis and desc_to_vis:
                        vis_filename = os.path.join(best_instance_vis_dir, f"{run_key}_paths.png")
                        visualize_paths(grid_data_best_vis, solution_to_vis, vis_filename, desc_to_vis)
            else:
                print(f"警告: 找到了最佳实例 {best_instance_num}，但未能找到其 ALNS 运行详情进行可视化。")
        else:
            print(f"警告: 找到了最佳实例 {best_instance_num}，但未能获取其地图数据进行可视化。")

    elif best_alns_run_overall:
        print("警告: 找到了最佳 ALNS 运行结果，但无法获取其详细运行数据进行可视化。")
    # --- 结束: 仅可视化最佳实例 ---

    # 保存详细结果到 CSV (仅 ALNS)
    detailed_results_file = os.path.join(RESULTS_DIR, f"{INSTANCE_PREFIX}_detailed_results.csv")
    try:
        fieldnames_set = set(); [fieldnames_set.update(res.keys()) for res in all_results]
        fieldnames = sorted(list(fieldnames_set))
        with open(detailed_results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore'); writer.writeheader(); writer.writerows(all_results)
        print(f"\n详细结果已保存到: {detailed_results_file}")
    except Exception as e: print(f"错误: 无法写入详细结果文件 '{detailed_results_file}': {e}")

    # 计算统计摘要 (仅 ALNS)
    summary_results: Dict[str, Dict[str, List[Optional[float]]]] = {} # Key is group name
    total_counts: Dict[str, int] = defaultdict(int)
    for result in all_results:
        if result['algorithm'] != 'ALNS': continue # 只统计 ALNS
        key = result['group']; total_counts[key] += 1
        if key not in summary_results:
            summary_results[key] = {metric: [] for metric in ['total_cost', 'travel_cost', 'turn_cost', 'wait_cost', 'cpu_time', 'makespan']}
            summary_results[key]['success_count'] = 0; summary_results[key]['priority'] = result['priority']
            summary_results[key]['speed_type'] = result['speed_type']; summary_results[key]['init_sources'] = []
        summary_results[key]['cpu_time'].append(result.get('cpu_time'))
        summary_results[key]['init_sources'].append(result.get('init_source', 'Unknown'))
        if result['success']:
            summary_results[key]['success_count'] += 1
            for metric in ['total_cost', 'travel_cost', 'turn_cost', 'wait_cost', 'makespan']: value = result.get(metric); summary_results[key][metric].append(float(value) if value is not None else None)
        else:
            for metric in ['total_cost', 'travel_cost', 'turn_cost', 'wait_cost', 'makespan']: summary_results[key][metric].append(None)

    # 格式化并打印统计摘要表 (仅 ALNS)
    summary_stats: List[Dict[str, Any]] = []
    print("\n--- ALNS 实验结果摘要 (均值 ± 标准差) ---")
    header_format = "{:<15} {:<6} {:<7} {:<8} {:<20} {:<15} {:<15} {:<15} {:<20} {:<15}" # 移除 Algorithm 列
    header_labels = ["Group", "Prio", "Speed", "Success", "Total Cost", "Travel Cost", "Turn Cost", "Wait Cost", "CPU Time (s)", "Makespan"]
    header_width = sum([15, 6, 7, 8, 20, 15, 15, 15, 20, 15]) + len(header_labels) -1
    print("-" * header_width); print(header_format.format(*header_labels)); print("-" * header_width)
    sorted_keys = sorted(summary_results.keys(), key=lambda x: ['Baseline', 'Control1', 'Experimental'].index(x))
    for key in sorted_keys:
        group = key; data = summary_results[key]; run_count = total_counts[key]
        success_count = data['success_count']; success_rate = f"{success_count}/{run_count}" if run_count > 0 else "0/0"
        priority = data.get('priority', 'N/A'); speed_type = data.get('speed_type', 'N/A')
        stats_row = {"group": group, "priority": priority, "speed_type": speed_type, "success_rate": success_rate}
        output_row = [f"{group:<15}", f"{priority:<6}", f"{speed_type:<7}", f"{success_rate:<8}"]
        total_cost_values = [v for v in data.get('total_cost', []) if v is not None]; tc_mean, tc_std = (np.mean(total_cost_values), np.std(total_cost_values)) if total_cost_values else (float('nan'), float('nan'))
        stats_row[f"total_cost_mean"], stats_row[f"total_cost_std"] = tc_mean, tc_std; tc_output_str = "{mean:.2f} ± {std:.2f}".format(mean=tc_mean, std=tc_std) if not np.isnan(tc_mean) else "N/A"; output_row.append(f"{tc_output_str:<20}")
        for cost_comp in ['travel_cost', 'turn_cost', 'wait_cost']: values_comp = [v for v in data.get(cost_comp, []) if v is not None]; comp_mean = np.mean(values_comp) if values_comp else float('nan'); stats_row[f"{cost_comp}_mean"] = comp_mean; comp_output_str = f"{comp_mean:.2f}" if not np.isnan(comp_mean) else "N/A"; output_row.append(f"{comp_output_str:<15}")
        cpu_times = [v for v in data.get('cpu_time', []) if v is not None]; cpu_mean, cpu_std = (np.mean(cpu_times), np.std(cpu_times)) if cpu_times else (float('nan'), float('nan'))
        stats_row[f"cpu_time_mean"], stats_row[f"cpu_time_std"] = cpu_mean, cpu_std; cpu_output_str = "{mean:.3f} ± {std:.3f}".format(mean=cpu_mean, std=cpu_std) if not np.isnan(cpu_mean) else "N/A"; output_row.append(f"{cpu_output_str:<20}")
        makespan_values = [v for v in data.get('makespan', []) if v is not None]; mk_mean, mk_std = (np.mean(makespan_values), np.std(makespan_values)) if makespan_values else (float('nan'), float('nan'))
        stats_row[f"makespan_mean"], stats_row[f"makespan_std"] = mk_mean, mk_std; mk_output_str = "{mean:.1f} ± {std:.1f}".format(mean=mk_mean, std=mk_std) if not np.isnan(mk_mean) else "N/A"; output_row.append(f"{mk_output_str:<15}")
        init_sources_count = Counter(data.get('init_sources', [])); stats_row['init_source_summary'] = dict(init_sources_count)
        summary_stats.append(stats_row); print(header_format.format(*output_row))
    print("-" * header_width)

    # 保存统计摘要到 CSV (仅 ALNS)
    summary_results_file = os.path.join(RESULTS_DIR, f"{INSTANCE_PREFIX}_summary_stats.csv")
    try:
        if summary_stats:
            preferred_order = ["group", "priority", "speed_type", "success_rate", "total_cost_mean", "total_cost_std", "travel_cost_mean", "turn_cost_mean", "wait_cost_mean", "cpu_time_mean", "cpu_time_std", "makespan_mean", "makespan_std", "init_source_summary"]
            fieldnames = [f for f in preferred_order if f in summary_stats[0]] + sorted([f for f in summary_stats[0] if f not in preferred_order])
            with open(summary_results_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore'); writer.writeheader(); writer.writerows(summary_stats)
            print(f"统计摘要已保存到: {summary_results_file}")
    except Exception as e: print(f"错误: 无法写入统计摘要文件 '{summary_results_file}': {e}")

    print("\n实验运行结束。")
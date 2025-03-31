# Visualizer.py-v4
"""
静态路径可视化工具。
- v4: 修复 MatplotlibDeprecationWarning for get_cmap。
- 保持 v3 的其他功能（手动场景 1，膨胀半径集成）。
- *** 提供绝对完整的代码，无省略。 ***
"""
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# v4: 不直接导入 cm，使用 plt.colormaps 或 plt.get_cmap
# import matplotlib.cm as cm
import numpy as np
from pathlib import Path as FilePath
from typing import Dict, Any, Optional, List, Tuple
import sys
import traceback
import random

# --- 标准导入当前项目模块 ---
try:
    from Map import GridMap, Node # 导入 v5
    from DataTypes import Task, Solution, Path, CostDict
    from Planner import TWAStarPlanner # 导入 v13
    from ALNS import ALNS # 导入 v29
    from Benchmark import PrioritizedPlanner # 导入 v6
    from InstanceGenerator import load_fixed_scenario_1, generate_scenario_2_instance # 导入 v10
except ImportError as e:
    print(f"错误: 导入 Visualizer 的依赖项失败: {e}")
    print("请确保所有必需的 .py 文件都在 Python 路径中且为最新版本。")
    sys.exit(1)

# --- 配置 (与 v3 相同) ---
VIS_CONFIG: Dict[str, Any] = {
    "expansion_radius": 1, "cost_weights": (1.0, 0.3, 0.8), "agv_v": 1.0, "delta_step": 1.0, "max_time_horizon": 400, "planner_buffer": 0, "default_planner_timeout": 30.0,
    "alns_max_iterations": 100, "alns_initial_temp": 10.0, "alns_cooling_rate": 0.98, "alns_segment_size": 20, "alns_weight_update_rate": 0.18, "alns_sigma1": 15.0, "alns_sigma2": 8.0, "alns_sigma3": 3.0, "alns_removal_percentage_min": 0.2, "alns_removal_percentage_max": 0.4, "alns_regret_k": 3, "alns_no_improvement_limit": 40, "alns_planner_time_limit_factor": 4.0, "alns_regret_planner_time_limit_abs": 0.1, "alns_verbose_output": False, "alns_debug_weights": False,
    "conflict_wait_threshold": 6, "conflict_deadlock_max_wait": 10,
    "scenario2_map_width": 12, "scenario2_map_height": 12, "scenario2_obstacle_ratio": 0.1, "scenario2_num_agvs": 5,
    "results_base_dir": "visualization_results_v4", # 更新版本号
}

# --- 辅助函数：创建结果目录 (与 v3 相同) ---
def setup_vis_results_dir(base_dir: str) -> FilePath:
    results_path = FilePath(base_dir); results_path.mkdir(parents=True, exist_ok=True); print(f"可视化结果将保存到: {results_path}"); return results_path

# --- 函数 1: 加载实例数据 (与 v3 相同) ---
def load_instance_data(scenario_num: int, instance_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Optional[Tuple[GridMap, List[Task]]]:
    print(f"--- 加载场景 {scenario_num} 实例数据 ---");
    if config is None: print("错误: 需要提供配置参数 (config) 以获取膨胀半径。"); return None
    expansion_radius = config.get('expansion_radius', 0); print(f"  使用膨胀半径: {expansion_radius}")
    if scenario_num == 1: instance_data = load_fixed_scenario_1(expansion_radius=expansion_radius);
    elif scenario_num == 2: print("警告: 每次运行将生成一个新的随机场景 2 实例。"); instance_data = generate_scenario_2_instance(width=config['scenario2_map_width'], height=config['scenario2_map_height'], obstacle_ratio=config['scenario2_obstacle_ratio'], num_agvs=config['scenario2_num_agvs'], expansion_radius=expansion_radius)
    else: print(f"错误: 不支持的场景编号 {scenario_num}。"); return None
    if not instance_data: print(f"错误: 加载场景 {scenario_num} 失败。")
    return instance_data

# --- 函数 2: 获取解决方案 (与 v3 相同) ---
def get_solution(algorithm_name: str, instance_data: Tuple[GridMap, List[Task]], config: Dict[str, Any]) -> Optional[Solution]:
    print(f"--- 运行 {algorithm_name} 以获取路径解 ---"); grid_map, tasks = instance_data; solution: Optional[Solution] = None; cost_dict: CostDict = {}
    try:
        if algorithm_name.upper() == "ALNS":
            planner_instance = TWAStarPlanner(); alns_params = {k.replace('alns_', ''): v for k, v in config.items() if k.startswith('alns_')}
            alns_params.update({'max_time': config['max_time_horizon'], 'cost_weights': config['cost_weights'], 'v': config['agv_v'], 'delta_step': config['delta_step'], 'buffer': config['planner_buffer'], 'wait_threshold': config['conflict_wait_threshold'], 'deadlock_max_wait': config['conflict_deadlock_max_wait'], 'verbose': config['alns_verbose_output'], 'debug_weights': config['alns_debug_weights']})
            alns_runner = ALNS(grid_map=grid_map, tasks=tasks, planner=planner_instance, **alns_params); print("  正在运行 ALNS..."); solution, _, cost_dict = alns_runner.run();
            if solution: print("  ALNS 运行成功。")
            else: print("  ALNS 运行失败或未找到解。")
        elif algorithm_name.upper() == "BENCHMARK":
            planner_instance = TWAStarPlanner(); v_bench = config['agv_v']; delta_step_bench = config['delta_step']
            benchmark_runner = PrioritizedPlanner(grid_map=grid_map, tasks=tasks, planner=planner_instance, v=v_bench, delta_step=delta_step_bench); print("  正在运行 Benchmark..."); solution, _, cost_dict = benchmark_runner.plan(cost_weights=config['cost_weights'], max_time=config['max_time_horizon'], time_limit_per_agent=config['default_planner_timeout'])
            if solution: print("  Benchmark 运行成功。")
            else: print("  Benchmark 运行失败或未找到解。")
        else: print(f"错误: 未知算法名称 '{algorithm_name}'"); return None
    except Exception as e: print(f"!!!!!!!!!! 运行算法时发生错误 !!!!!!!!!!"); print(f"Algorithm={algorithm_name}"); print(f"错误: {type(e).__name__}: {e}"); traceback.print_exc(); return None
    if solution is not None and not isinstance(solution, dict): print(f"错误: 算法返回的 Solution 不是字典类型: {type(solution)}"); return None
    if solution is not None:
        for agv_id, path in solution.items():
            if path is not None and not isinstance(path, Path): print(f"错误: Solution 中 AGV {agv_id} 的值不是 Path 类型或 None: {type(path)}"); return None
    return solution

# --- 函数 3: 可视化静态路径 (v4: 修复 get_cmap 警告) ---
def visualize_static_paths(grid_map: GridMap, solution: Solution, tasks: List[Task], filename: str):
    print(f"--- 开始生成静态路径图: {filename} ---");
    if not solution: print("错误: 没有有效的解决方案可以可视化。"); return
    valid_agv_ids_in_solution = {agv_id for agv_id, path in solution.items() if path and path.sequence}
    if not valid_agv_ids_in_solution: print("错误: 解决方案中没有有效的 AGV 路径。"); return
    num_agvs_to_plot = len(valid_agv_ids_in_solution)

    fig, ax = plt.subplots(figsize=(10, 10)); ax.set_xlim(-0.5, grid_map.width - 0.5); ax.set_ylim(-0.5, grid_map.height - 0.5); ax.set_aspect('equal', adjustable='box'); ax.invert_yaxis(); ax.set_xticks(np.arange(grid_map.width)); ax.set_yticks(np.arange(grid_map.height)); ax.set_xlabel("X coordinate"); ax.set_ylabel("Y coordinate"); ax.set_title(f"Static AGV Paths ({num_agvs_to_plot} AGVs)"); ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')
    obstacle_color = 'black'; free_space_color = 'white'
    for r in range(grid_map.height):
        for c in range(grid_map.width): color = obstacle_color if grid_map.is_obstacle(c, r) else free_space_color; rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=0.5, edgecolor='grey', facecolor=color, alpha=0.9 if color==obstacle_color else 0.1); ax.add_patch(rect)

    # v4: 使用新的 API 获取颜色映射
    try: cmap = plt.colormaps.get_cmap('tab10')
    except AttributeError: cmap = plt.get_cmap('tab10')
    num_colors_needed = max(10, len(tasks)) # 确保至少有 10 种颜色
    # 使用与之前类似的逻辑分配颜色
    colors = {task.agv_id: cmap(i % cmap.N) for i, task in enumerate(tasks)}

    start_marker = 'o'; goal_marker = 's'; marker_size = 8
    for task in tasks:
        if task.agv_id in valid_agv_ids_in_solution:
            agv_id = task.agv_id; start_node = task.start_node; goal_node = task.goal_node; color = colors.get(agv_id, 'grey')
            ax.plot(start_node[0], start_node[1], marker=start_marker, color=color, markersize=marker_size, linestyle='None', label=f'AGV {agv_id} Start', zorder=5)
            ax.text(start_node[0], start_node[1], f'S{agv_id}', color='white', fontsize=6, ha='center', va='center', zorder=6, fontweight='bold')
            ax.plot(goal_node[0], goal_node[1], marker=goal_marker, color=color, markersize=marker_size, linestyle='None', label=f'AGV {agv_id} Goal', zorder=5)
            ax.text(goal_node[0], goal_node[1], f'G{agv_id}', color='white', fontsize=6, ha='center', va='center', zorder=6, fontweight='bold')
    path_linewidth = 1.5; path_alpha = 0.8; legend_handles = []
    for agv_id, path in solution.items():
        if path and path.sequence:
            color = colors.get(agv_id, 'grey'); nodes_in_path = [state[0] for state in path.sequence];
            if not nodes_in_path: continue
            x_coords = [node[0] for node in nodes_in_path]; y_coords = [node[1] for node in nodes_in_path]
            line, = ax.plot(x_coords, y_coords, color=color, linewidth=path_linewidth, alpha=path_alpha, label=f'AGV {agv_id} Path', zorder=4)
            legend_handles.append(patches.Patch(color=color, label=f'AGV {agv_id}'))
    if legend_handles: ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    try: save_path = FilePath(filename); save_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"静态路径图已保存到: {save_path}")
    except Exception as e: print(f"错误: 保存图像 '{filename}' 失败: {e}")
    plt.close(fig)

# --- 主执行块 (与 v3 相同) ---
if __name__ == "__main__":
    print("--- 开始静态路径可视化 (v4 - Fixed get_cmap Warning) ---")
    scenario_to_visualize = 1; algorithm_to_visualize = "ALNS"; instance_id_s2 = None
    vis_config = VIS_CONFIG; vis_results_dir = setup_vis_results_dir(vis_config['results_base_dir'])
    if scenario_to_visualize == 1: instance_id_str = "ManualFixed_10x10_ExpR"+str(vis_config['expansion_radius'])
    elif scenario_to_visualize == 2: instance_id_str = f"Random_{vis_config['scenario2_map_width']}x{vis_config['scenario2_map_height']}_Obs{vis_config['scenario2_obstacle_ratio']*100:.0f}p_ExpR{vis_config['expansion_radius']}"; instance_id_str += f"_{instance_id_s2}" if instance_id_s2 else "_new"
    else: instance_id_str = "Unknown"
    instance_data = load_instance_data(scenario_to_visualize, instance_id=instance_id_s2, config=vis_config)
    if instance_data:
        solution = get_solution(algorithm_to_visualize, instance_data, vis_config)
        if solution is not None:
             output_filename = vis_results_dir / f"static_paths_Scen{scenario_to_visualize}_{instance_id_str}_{algorithm_to_visualize}.png"
             visualize_static_paths(instance_data[0], solution, instance_data[1], str(output_filename))
        else: print("错误: 未能获取有效的解决方案，无法生成路径可视化。")
    else: print("错误: 加载实例数据失败，可视化终止。")
    print("\n--- 可视化流程结束 ---")
# Animator.py-v4
"""
为单个算例和算法生成的解决方案创建动态可视化动画。
- v4: 添加动态轨迹线显示 AGV 已走过的路径。
      减小 AGV 圆点标记的大小。
- 保持 v3 的 TypeError 修复。
"""

import time
import sys
import traceback
import random
import copy
from pathlib import Path as FilePathALN
from typing import Dict, Any, Optional, List, Tuple # 确保导入 typing

# --- 尝试导入依赖库 (与 v3 相同) ---
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle, Circle
    # v4: 导入 Line2D
    from matplotlib.lines import Line2D
    _visual_libs_available = True
except ImportError:
    print("错误: 动画生成需要 numpy 和 matplotlib。")
    print("请使用 'pip install numpy matplotlib' 安装。")
    _visual_libs_available = False

# --- 导入项目核心模块 (与 v3 相同) ---
try:
    from Map import GridMap
    from DataTypes import Task, Solution, Path, CostDict, TimeStep, Node
    from Planner import TWAStarPlanner
    from ALNS import ALNS
    from Benchmark import PrioritizedPlanner
    from InstanceGenerator import load_fixed_scenario_1, generate_scenario_2_instance
    _project_libs_available = True
except ImportError as e:
    print(f"错误: 导入项目核心模块失败: {e}")
    print("请确保 Map.py, DataTypes.py, Planner.py, ALNS.py, Benchmark.py, InstanceGenerator.py 在 Python 路径中。")
    _project_libs_available = False

# --- 配置 (与 v3 相同) ---
DEFAULT_ANIMATOR_CONFIG: Dict[str, Any] = {
    # 1. 场景与算法选择
    "scenario_to_animate": 1,
    "algorithm_to_animate": "ALNS",
    "instance_id_s2": None,
    "instance_run_id_s2": 1,

    # 2. 动画参数
    "animation_interval_ms": 150,
    "output_filename_base": "agv_animation",
    "output_format": "gif",
    "save_animation": True,
    "results_dir": "animations",
    "show_time_on_plot": True,

    # 3. 共享环境/算法参数
    "expansion_radius": 1,
    "cost_weights": [1.0, 0.3, 0.8],
    "agv_v": 1.0,
    "delta_step": 1.0,
    "max_time_horizon": 400,
    "planner_buffer": 0,
    "default_planner_timeout": 30.0,

    # --- ALNS 特定参数 ---
    "alns_max_iterations": 100,
    "alns_initial_temp": 10.0,
    "alns_cooling_rate": 0.98,
    "alns_segment_size": 20,
    "alns_weight_update_rate": 0.18,
    "alns_sigma1": 15.0, "alns_sigma2": 8.0, "alns_sigma3": 3.0,
    "alns_removal_percentage_min": 0.2, "alns_removal_percentage_max": 0.4,
    "alns_regret_k": 3,
    "alns_no_improvement_limit": 40,
    "alns_planner_time_limit_factor": 4.0,
    "alns_regret_planner_time_limit_abs": 0.1,
    "alns_verbose_output": False,
    "alns_debug_weights": False,
    "alns_record_history": False,
    "alns_plot_convergence": False,

    # --- 冲突解决参数 (如果 ALNS 需要) ---
    "conflict_wait_threshold": 6,
    "conflict_deadlock_max_wait": 10,

    # --- 场景 2 特定参数 ---
    "scenario2_map_width": 12,
    "scenario2_map_height": 12,
    "scenario2_obstacle_ratio": 0.1,
    "scenario2_num_agvs": 5,

    # 4. 可视化细节
    # v4: 减小 AGV 标记大小
    "agv_marker_size_radius": 0.25, # 使用半径代替硬编码大小
    "trajectory_linewidth": 1.5,
    "trajectory_alpha": 0.8,
    "obstacle_color": 'black',
    "buffer_color": '#a9a9a9',
    "start_color": 'green',
    "goal_color": 'red',
    "grid_line_style": '--',
    "grid_line_alpha": 0.3,
    "plot_title": "AGV Path Animation",
    "use_random_agv_colors": True,
    "default_agv_color": 'blue',
    "color_map": "tab10",
}

# --- 动画类 ---
class Animator:
    def __init__(self, config: Dict[str, Any]):
        """初始化 Animator。"""
        if not _visual_libs_available or not _project_libs_available:
            raise ImportError("缺少必要的库或项目模块，无法创建 Animator。")

        self.config = config
        self.grid_map: Optional[GridMap] = None
        self.tasks: Optional[List[Task]] = None
        self.solution: Optional[Solution] = None
        self.max_time: TimeStep = 0
        self.agv_ids: List[int] = []
        self.agv_colors: Dict[int, Any] = {}

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.agv_patches: Dict[int, Circle] = {}
        # v4: 添加轨迹线存储
        self.trajectory_lines: Dict[int, Line2D] = {}
        self.start_markers: Optional[PatchCollection] = None
        self.goal_markers: Optional[PatchCollection] = None
        self.time_text: Optional[plt.Text] = None
        self.animation = None

        # 场景和算法配置 (与 v3 相同)
        self.scenario = config['scenario_to_animate']
        self.algorithm = config['algorithm_to_animate']
        self.instance_id_s2 = config.get('instance_id_s2')
        self.instance_run_id_s2 = config.get('instance_run_id_s2', 1)
        self.expansion_radius = config['expansion_radius']

    # --- _load_instance (与 v3 相同) ---
    def _load_instance(self) -> bool:
        """加载指定的场景实例。"""
        print(f"\n--- 加载场景 {self.scenario} 实例 ---")
        instance_data: Optional[Tuple[GridMap, List[Task]]] = None
        instance_id_str = ""

        if self.scenario == 1:
            instance_data = load_fixed_scenario_1(expansion_radius=self.expansion_radius)
            instance_id_str = f"Fixed_10x10_ExpR{self.expansion_radius}"
        elif self.scenario == 2:
            s2_width = self.config['scenario2_map_width']
            s2_height = self.config['scenario2_map_height']
            s2_obs_ratio = self.config['scenario2_obstacle_ratio']
            s2_num_agvs = self.config['scenario2_num_agvs']
            instance_id_base = f"Random_{s2_width}x{s2_height}_Obs{s2_obs_ratio*100:.0f}p_AGV{s2_num_agvs}_ExpR{self.expansion_radius}"
            instance_id_str = f"{instance_id_base}_#{self.instance_run_id_s2}_Anim"
            print(f"生成随机实例: {instance_id_str}")
            instance_data = generate_scenario_2_instance(
                width=s2_width, height=s2_height, obstacle_ratio=s2_obs_ratio,
                num_agvs=s2_num_agvs, expansion_radius=self.expansion_radius
            )
        else:
            print(f"错误: 未知的场景编号 {self.scenario}")
            return False

        if instance_data:
            self.grid_map, self.tasks = instance_data
            self.agv_ids = sorted([task.agv_id for task in self.tasks])
            print(f"实例加载/生成成功: {instance_id_str}")
            print(f"  地图: {self.grid_map}")
            print(f"  任务数量: {len(self.tasks)}")
            self.config['_instance_id_resolved'] = instance_id_str
            return True
        else:
            print(f"错误: 加载或生成场景 {self.scenario} 实例失败。")
            return False

    # --- _get_solution (与 v3 相同) ---
    def _get_solution(self) -> bool:
        """重新运行指定算法以获取解决方案。"""
        if not self.grid_map or not self.tasks:
            print("错误: 地图或任务未加载，无法获取解决方案。")
            return False

        print(f"\n--- 运行算法 '{self.algorithm}' 获取解决方案 ---")
        solution: Optional[Solution] = None
        cost_dict: CostDict = {'total': float('inf')}
        cpu_time: float = 0.0
        start_time = time.time()

        try:
            if self.algorithm == "ALNS":
                planner_instance = TWAStarPlanner()
                alns_params = {k.replace('alns_', ''): v for k, v in self.config.items() if k.startswith('alns_')}
                alns_params['max_time'] = self.config['max_time_horizon']
                alns_params['cost_weights'] = self.config['cost_weights']
                alns_params['v'] = self.config['agv_v']
                alns_params['delta_step'] = self.config['delta_step']
                alns_params['buffer'] = self.config['planner_buffer']
                alns_params['wait_threshold'] = self.config['conflict_wait_threshold']
                alns_params['deadlock_max_wait'] = self.config['conflict_deadlock_max_wait']
                alns_params['instance_identifier'] = self.config.get('_instance_id_resolved', 'anim_instance') + f"_{self.algorithm}"
                temp_results_path = FilePath(self.config['results_dir']) / "_temp_alns_run"
                try:
                    temp_results_path.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    print(f"警告: 无法创建 ALNS 临时目录 '{temp_results_path}': {e}")
                alns_params['results_dir'] = str(temp_results_path)
                alns_runner = ALNS(grid_map=self.grid_map, tasks=self.tasks, planner=planner_instance, **alns_params)
                solution, cpu_time, cost_dict = alns_runner.run()

            elif self.algorithm == "Benchmark":
                planner_instance = TWAStarPlanner()
                benchmark_runner = PrioritizedPlanner(
                    grid_map=self.grid_map, tasks=self.tasks, planner=planner_instance,
                    v=self.config['agv_v'], delta_step=self.config['delta_step']
                )
                solution, cpu_time, cost_dict = benchmark_runner.plan(
                    cost_weights=self.config['cost_weights'],
                    max_time=self.config['max_time_horizon'],
                    time_limit_per_agent=self.config['default_planner_timeout']
                )
            else:
                print(f"错误: 未知算法 '{self.algorithm}'")
                return False

        except Exception as e:
            print(f"错误: 运行算法 '{self.algorithm}' 时发生异常: {e}")
            traceback.print_exc()
            return False

        run_duration = time.time() - start_time
        print(f"算法运行耗时: {run_duration:.2f}s (内部计时: {cpu_time:.4f}s)")

        if solution and cost_dict.get('total', float('inf')) != float('inf'):
            self.solution = solution
            self.max_time = 0
            for path in self.solution.values():
                if path and path.sequence:
                    self.max_time = max(self.max_time, path.get_makespan())
            cost_str = f"{cost_dict.get('total', 0):.2f}"
            print(f"解决方案获取成功，总成本: {cost_str}, 最大时间步: {self.max_time}")
            return True
        else:
            print(f"错误: 算法 '{self.algorithm}' 未能找到有效的解决方案。")
            self.solution = None
            return False

    # v4: 修改 _setup_plot 以创建轨迹线对象
    def _setup_plot(self):
        """设置 Matplotlib 图形和坐标轴，并初始化 AGV 标记和轨迹线。"""
        if not self.grid_map or not self.tasks: return False

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        width = self.grid_map.width
        height = self.grid_map.height

        # --- 绘制地图背景、障碍物、起点/终点 (与 v3 相同) ---
        self.ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        self.ax.grid(which='minor', color=self.config['obstacle_color'], linestyle=self.config['grid_line_style'], linewidth=0.5, alpha=self.config['grid_line_alpha'])
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, width - 0.5); self.ax.set_ylim(-0.5, height - 0.5)
        self.ax.set_aspect('equal', adjustable='box')

        obstacle_patches = []
        buffer_patches = []
        if self.grid_map.obstacles:
            for obs_node in self.grid_map.obstacles:
                is_raw = obs_node in self.grid_map.raw_obstacles
                color = self.config['obstacle_color'] if is_raw else self.config['buffer_color']
                patch = Rectangle((obs_node[0] - 0.5, obs_node[1] - 0.5), 1, 1, color=color)
                if is_raw: obstacle_patches.append(patch)
                else: buffer_patches.append(patch)
            if buffer_patches: self.ax.add_collection(PatchCollection(buffer_patches, match_original=True))
            if obstacle_patches: self.ax.add_collection(PatchCollection(obstacle_patches, match_original=True))

        start_patches = []; goal_patches = []
        task_nodes = set()
        for task in self.tasks:
            start_node = task.start_node; goal_node = task.goal_node
            task_nodes.add(start_node); task_nodes.add(goal_node)
            start_patches.append(Rectangle((start_node[0] - 0.4, start_node[1] - 0.4), 0.8, 0.8, color=self.config['start_color'], alpha=0.7))
            goal_patches.append(Circle((goal_node[0], goal_node[1]), 0.4, color=self.config['goal_color'], alpha=0.7))
        self.start_markers = PatchCollection(start_patches, match_original=True)
        self.goal_markers = PatchCollection(goal_patches, match_original=True)
        self.ax.add_collection(self.start_markers); self.ax.add_collection(self.goal_markers)
        # ---------------------------------------------------------------

        # --- 初始化 AGV 颜色、标记(v4:更小) 和轨迹线 ---
        cmap = plt.get_cmap(self.config['color_map'])
        color_indices = np.linspace(0, 1, len(self.agv_ids))
        agv_marker_radius = self.config.get('agv_marker_size_radius', 0.25) # 使用配置半径

        for i, agv_id in enumerate(self.agv_ids):
            # 获取颜色
            color = cmap(color_indices[i]) if self.config['use_random_agv_colors'] else self.config['default_agv_color']
            self.agv_colors[agv_id] = color

            # 创建 AGV 圆点标记 (v4: 使用配置半径，zorder更高)
            patch = Circle((-1, -1), agv_marker_radius, color=color, zorder=10) # zorder=10, 在线之上
            self.agv_patches[agv_id] = patch
            self.ax.add_patch(patch)

            # v4: 创建轨迹线对象
            line, = self.ax.plot([], [], color=color,
                                 linewidth=self.config.get('trajectory_linewidth', 1.5),
                                 alpha=self.config.get('trajectory_alpha', 0.8),
                                 zorder=5) # zorder=5, 在标记之下
            self.trajectory_lines[agv_id] = line
        # ----------------------------------------------------------

        # --- 初始化时间文本和标题 (与 v3 相同) ---
        if self.config['show_time_on_plot']:
            self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, verticalalignment='top', fontsize=12)

        instance_name = self.config.get('_instance_id_resolved', 'Unknown Instance')
        title = f"{self.config['plot_title']}\nAlgorithm: {self.algorithm}, Instance: {instance_name}"
        self.ax.set_title(title, wrap=True)
        self.fig.tight_layout()
        # -----------------------------------------
        return True

    # --- _get_position (与 v3 相同) ---
    def _get_position(self, agv_id: int, time_step: TimeStep) -> Optional[Node]:
        """获取 AGV 在特定时间步的位置。"""
        if not self.solution or agv_id not in self.solution: return None
        path = self.solution[agv_id]
        if not path or not path.sequence: return None
        current_node: Optional[Node] = None
        found_idx = -1
        for idx, (node, t) in enumerate(path.sequence):
             if t >= time_step:
                  found_idx = idx
                  break
        if found_idx != -1:
             if path.sequence[found_idx][1] == time_step: current_node = path.sequence[found_idx][0]
             elif found_idx > 0: current_node = path.sequence[found_idx - 1][0]
             else: current_node = path.sequence[0][0]
        else: current_node = path.sequence[-1][0]
        return current_node

    # v4: 修改 _update_animation 以更新轨迹线
    def _update_animation(self, frame: int) -> Tuple:
        """更新动画的每一帧，包括 AGV 位置和轨迹线。"""
        time_step = frame
        updated_artists: List[Any] = [] # 使用列表存储

        for agv_id in self.agv_ids:
            # --- 更新 AGV 圆点标记 ---
            patch = self.agv_patches.get(agv_id)
            if patch:
                position = self._get_position(agv_id, time_step)
                if position:
                    patch.center = (position[0], position[1]) # position 是 (x, y)
                    patch.set_visible(True)
                    updated_artists.append(patch)
                else:
                    patch.set_visible(False)
            # ------------------------

            # --- v4: 更新轨迹线 ---
            line = self.trajectory_lines.get(agv_id)
            if line and self.solution:
                path_obj = self.solution.get(agv_id)
                if path_obj and path_obj.sequence:
                    # 获取时间 <= frame 的路径段节点
                    segment_nodes = [state[0] for state in path_obj.sequence if state[1] <= frame]

                    if len(segment_nodes) > 1: # 至少需要两个点才能画线
                        # segment_nodes 是 [(x0, y0), (x1, y1), ...]
                        x_coords = [node[0] for node in segment_nodes]
                        y_coords = [node[1] for node in segment_nodes]
                        line.set_data(x_coords, y_coords)
                    else: # 如果只有一个点或没有点，清空线条
                        line.set_data([], [])
                    updated_artists.append(line) # 必须将线添加到更新列表
            # -----------------------

        # --- 更新时间文本 (不变) ---
        if self.time_text:
            self.time_text.set_text(f'Time: {time_step * self.config["delta_step"]:.1f}')
            updated_artists.append(self.time_text)
        # --------------------------

        return tuple(updated_artists) # 返回所有更新过的 artists 的元组

    # v4: 修改 run_and_animate 以使用新的 init_func
    def run_and_animate(self):
        """执行加载、求解和动画生成的完整流程。"""
        if not self._load_instance(): return
        if not self._get_solution(): return
        if not self._setup_plot(): return

        if self.max_time is None or self.max_time < 0: # 允许 max_time 为 0
             print("警告: 计算出的最大时间步无效或为负数，无法生成动画。")
             return

        print(f"\n--- 生成动画 (共 {self.max_time + 1} 帧) ---")
        num_frames = self.max_time + 1
        interval = self.config['animation_interval_ms']

        # v4: 定义 init_func 以正确初始化 blit
        def init_func():
             """初始化动画帧，清空轨迹线。"""
             artists = list(self.agv_patches.values()) \
                       + list(self.trajectory_lines.values())
             if self.time_text:
                 artists.append(self.time_text)
                 self.time_text.set_text('') # 初始化时间文本
             # 清空所有轨迹线的初始数据
             for line in self.trajectory_lines.values():
                 line.set_data([], [])
             # 设置 AGV 标记初始不可见或在 (-1,-1)
             for patch in self.agv_patches.values():
                  patch.center = (-1, -1)
             return tuple(artists)

        self.animation = animation.FuncAnimation(
            self.fig, self._update_animation, init_func=init_func, frames=num_frames,
            interval=interval, blit=True, repeat=False
        )

        # --- 保存动画 (与 v3 相同) ---
        if self.config['save_animation']:
            output_dir = FilePath(self.config['results_dir'])
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"错误: 无法创建动画输出目录 '{output_dir}': {e}")
                return

            instance_name = self.config.get('_instance_id_resolved', 'unknown_instance')
            filename_base = self.config['output_filename_base']
            output_filename = f"{filename_base}_{instance_name}_{self.algorithm}.gif"
            output_path = output_dir / output_filename
            writer_name = 'pillow'

            try:
                print(f"尝试使用 '{writer_name}' 保存 GIF 动画到: {output_path}")
                def progress_callback(current_frame, total_frames):
                     if total_frames > 0:
                          percent = 100.0 * current_frame / total_frames
                          # 减少打印频率
                          if current_frame % max(1, total_frames // 20) == 0 or current_frame == total_frames - 1:
                               print(f"  保存进度: {percent:.1f}% ({current_frame+1}/{total_frames})", end='\r')
                          if current_frame == total_frames - 1:
                               print("\n保存完成。")

                self.animation.save(str(output_path), writer=writer_name, dpi=150, progress_callback=progress_callback)
                print(f"动画已保存到: {output_path.resolve()}")
            except ImportError:
                print("错误: 保存 GIF 需要 Pillow 库。请运行 'pip install Pillow'。")
                print("将尝试显示动画...")
                plt.show()
            except Exception as e:
                print(f"错误: 保存动画失败 ({type(e).__name__})。")
                print(f"  错误信息: {e}")
                traceback.print_exc()
                print("将尝试显示动画...")
                plt.show()
        else:
            print("显示动画...")
            plt.show()

# --- 入口点 (与 v3 相同) ---
if __name__ == "__main__":
    if not _visual_libs_available or not _project_libs_available:
        print("检测到缺少必要的库或项目模块，无法运行动画生成器。")
        sys.exit(1)

    print("====== AGV 动画生成器 v4 ======")
    config_to_use = DEFAULT_ANIMATOR_CONFIG.copy()

    try:
        animator = Animator(config_to_use)
        animator.run_and_animate()
    except Exception as main_e:
        print(f"\n!!!!!! 动画生成过程中发生未处理的错误 !!!!!!")
        print(f"{type(main_e).__name__}: {main_e}")
        traceback.print_exc()

    print("\n====== 动画生成器运行结束 ======")
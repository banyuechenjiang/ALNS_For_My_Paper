# Animator.py-v4.2
"""
为单个算例和算法生成的解决方案创建动态可视化动画。
- v4.2: 添加命令行参数解析，支持加载外部 JSON 配置文件 (如 base_experiment_config_with_bbox.json)
        以覆盖默认配置，命令行参数具有最高优先级。
- v4.1: 修复了因别名修改导致的 NameError (FilePath -> FilePathALN)。
- v4: 添加动态轨迹线显示 AGV 已走过的路径。
      减小 AGV 圆点标记的大小。
- 保持 v3 的 TypeError 修复。
"""

import time
import sys
import traceback
import random
import copy
import json         # <--- 导入 json
import argparse     # <--- 导入 argparse
from pathlib import Path as FilePathALN
from typing import Dict, Any, Optional, List, Tuple

# --- 尝试导入依赖库 ---
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.lines import Line2D
    _visual_libs_available = True
except ImportError:
    print("错误: 动画生成需要 numpy 和 matplotlib。")
    print("请使用 'pip install numpy matplotlib' 安装。")
    _visual_libs_available = False

# --- 导入项目核心模块 ---
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

# --- 配置 ---
DEFAULT_ANIMATOR_CONFIG: Dict[str, Any] = {
    # 1. 场景与算法选择 (可被覆盖)
    "scenario_to_animate": 1,
    "algorithm_to_animate": "ALNS",
    "instance_id_s2": None,
    "instance_run_id_s2": 1,

    # 2. 动画参数 (可被覆盖)
    "animation_interval_ms": 150,
    "output_filename_base": "agv_animation",
    "output_format": "gif",
    "save_animation": True,
    "results_dir": "animations",
    "show_time_on_plot": True,

    # 3. 共享环境/算法参数 (可被覆盖)
    "expansion_radius": 1,
    "cost_weights": [1.0, 0.3, 0.8],
    "agv_v": 1.0,
    "delta_step": 1.0,
    "max_time_horizon": 400,
    "planner_buffer": 0, # TWA* buffer (通常不应在 ALNS 中使用)
    "default_planner_timeout": 30.0, # Benchmark 的超时

    # --- ALNS 特定参数 (可被覆盖) ---
    "alns_max_iterations": 100,
    "alns_initial_temp": 10.0,
    "alns_cooling_rate": 0.98,
    "alns_segment_size": 20,
    "alns_weight_update_rate": 0.18,
    "alns_sigma1": 15.0, "alns_sigma2": 8.0, "alns_sigma3": 3.0,
    "alns_removal_percentage_min": 0.2, "alns_removal_percentage_max": 0.4,
    "alns_regret_k": 3,
    "alns_no_improvement_limit": 40, # ALNS 早停参数
    "alns_planner_time_limit_factor": 4.0,
    "alns_regret_planner_time_limit_abs": 0.1,
    "alns_verbose_output": False, # 减少 ALNS 运行时的打印
    "alns_debug_weights": False,
    "alns_record_history": False, # 动画生成不需要记录历史
    "alns_plot_convergence": False,

    # --- 冲突解决参数 (可被覆盖) ---
    "conflict_wait_threshold": 6,
    "conflict_deadlock_max_wait": 10, # 默认使用这个 key

    # --- 场景 2 特定参数 (可被覆盖) ---
    "scenario2_map_width": 12,
    "scenario2_map_height": 12,
    "scenario2_obstacle_ratio": 0.1,
    "scenario2_num_agvs": 5,

    # 4. 可视化细节 (通常保持默认)
    "agv_marker_size_radius": 0.25,
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

# --- 动画类 (保持与 v4.1 相同，核心逻辑不变) ---
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
        self.trajectory_lines: Dict[int, Line2D] = {}
        self.start_markers: Optional[PatchCollection] = None
        self.goal_markers: Optional[PatchCollection] = None
        self.time_text: Optional[plt.Text] = None
        self.animation = None

        # 从最终配置中提取场景/算法信息
        self.scenario = config['scenario_to_animate']
        self.algorithm = config['algorithm_to_animate']
        self.instance_id_s2 = config.get('instance_id_s2')
        self.instance_run_id_s2 = config.get('instance_run_id_s2', 1)
        self.expansion_radius = config['expansion_radius']

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
            # 使用 instance_id_s2 或 instance_run_id_s2 命名
            inst_suffix = self.instance_id_s2 if self.instance_id_s2 is not None else self.instance_run_id_s2
            instance_id_str = f"{instance_id_base}_#{inst_suffix}_Anim"
            print(f"生成随机实例: {instance_id_str}")
            # 注意：如果需要复现特定的随机实例，InstanceGenerator 可能需要支持种子
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
            self.config['_instance_id_resolved'] = instance_id_str # 用于标题
            return True
        else:
            print(f"错误: 加载或生成场景 {self.scenario} 实例失败。")
            return False

    def _get_solution(self) -> bool:
        """重新运行指定算法以获取解决方案 (使用 self.config 中的参数)。"""
        if not self.grid_map or not self.tasks:
            print("错误: 地图或任务未加载，无法获取解决方案。")
            return False

        print(f"\n--- 运行算法 '{self.algorithm}' 获取解决方案 (使用当前配置) ---")
        solution: Optional[Solution] = None
        cost_dict: CostDict = {'total': float('inf')}
        cpu_time: float = 0.0
        start_time = time.time()

        try:
            if self.algorithm == "ALNS":
                planner_instance = TWAStarPlanner()
                # 从 self.config 构建 ALNS 参数字典
                alns_params = {k.replace('alns_', ''): v for k, v in self.config.items() if k.startswith('alns_')}
                alns_params['max_time'] = self.config['max_time_horizon']
                alns_params['cost_weights'] = self.config['cost_weights']
                alns_params['v'] = self.config['agv_v']
                alns_params['delta_step'] = self.config['delta_step']
                # 优先使用 config 中的 planner_buffer (即使 ALNS 不应该依赖它)
                alns_params['buffer'] = self.config.get('planner_buffer', 0)
                alns_params['wait_threshold'] = self.config['conflict_wait_threshold']
                # 优先使用 config 中的 deadlock_max_wait (处理 JSON 中可能的 key 不同)
                alns_params['deadlock_max_wait'] = self.config.get('conflict_deadlock_max_wait',
                                                                     self.config.get('alns_deadlock_max_wait', 10)) # 后备
                alns_params['instance_identifier'] = self.config.get('_instance_id_resolved', 'anim_instance') + f"_{self.algorithm}"
                temp_results_path = FilePathALN(self.config.get('results_dir', 'animations')) / "_temp_alns_run"
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
                    # Benchmark 不使用 planner_buffer 或 deadlock 参数
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

    def _setup_plot(self):
        """设置 Matplotlib 图形和坐标轴，并初始化 AGV 标记和轨迹线。"""
        # --- 这部分代码与 v4.1 完全相同 ---
        if not self.grid_map or not self.tasks: return False
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        width = self.grid_map.width; height = self.grid_map.height
        self.ax.set_xticks(np.arange(-0.5, width, 1), minor=True); self.ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        self.ax.grid(which='minor', color=self.config['obstacle_color'], linestyle=self.config['grid_line_style'], linewidth=0.5, alpha=self.config['grid_line_alpha'])
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, width - 0.5); self.ax.set_ylim(-0.5, height - 0.5); self.ax.set_aspect('equal', adjustable='box')
        obstacle_patches = []; buffer_patches = []
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
        for task in self.tasks:
            start_node = task.start_node; goal_node = task.goal_node
            start_patches.append(Rectangle((start_node[0] - 0.4, start_node[1] - 0.4), 0.8, 0.8, color=self.config['start_color'], alpha=0.7))
            goal_patches.append(Circle((goal_node[0], goal_node[1]), 0.4, color=self.config['goal_color'], alpha=0.7))
        self.start_markers = PatchCollection(start_patches, match_original=True); self.goal_markers = PatchCollection(goal_patches, match_original=True)
        self.ax.add_collection(self.start_markers); self.ax.add_collection(self.goal_markers)
        cmap = plt.get_cmap(self.config['color_map']); color_indices = np.linspace(0, 1, len(self.agv_ids)); agv_marker_radius = self.config.get('agv_marker_size_radius', 0.25)
        for i, agv_id in enumerate(self.agv_ids):
            color = cmap(color_indices[i]) if self.config['use_random_agv_colors'] else self.config['default_agv_color']
            self.agv_colors[agv_id] = color
            patch = Circle((-1, -1), agv_marker_radius, color=color, zorder=10); self.agv_patches[agv_id] = patch; self.ax.add_patch(patch)
            line, = self.ax.plot([], [], color=color, linewidth=self.config.get('trajectory_linewidth', 1.5), alpha=self.config.get('trajectory_alpha', 0.8), zorder=5); self.trajectory_lines[agv_id] = line
        if self.config['show_time_on_plot']: self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, verticalalignment='top', fontsize=12)
        instance_name = self.config.get('_instance_id_resolved', 'Unknown Instance')
        title = f"{self.config['plot_title']}\nAlgorithm: {self.algorithm}, Instance: {instance_name}"
        self.ax.set_title(title, wrap=True); self.fig.tight_layout()
        return True

    def _get_position(self, agv_id: int, time_step: TimeStep) -> Optional[Node]:
        """获取 AGV 在特定时间步的位置。"""
        # --- 这部分代码与 v4.1 完全相同 ---
        if not self.solution or agv_id not in self.solution: return None
        path = self.solution[agv_id];
        if not path or not path.sequence: return None
        current_node: Optional[Node] = None; found_idx = -1
        for idx, (node, t) in enumerate(path.sequence):
             if t >= time_step: found_idx = idx; break
        if found_idx != -1:
             if path.sequence[found_idx][1] == time_step: current_node = path.sequence[found_idx][0]
             elif found_idx > 0: current_node = path.sequence[found_idx - 1][0]
             else: current_node = path.sequence[0][0]
        else: current_node = path.sequence[-1][0]
        return current_node

    def _update_animation(self, frame: int) -> Tuple:
        """更新动画的每一帧，包括 AGV 位置和轨迹线。"""
        # --- 这部分代码与 v4.1 完全相同 ---
        time_step = frame; updated_artists: List[Any] = []
        for agv_id in self.agv_ids:
            patch = self.agv_patches.get(agv_id)
            if patch:
                position = self._get_position(agv_id, time_step)
                if position: patch.center = (position[0], position[1]); patch.set_visible(True); updated_artists.append(patch)
                else: patch.set_visible(False)
            line = self.trajectory_lines.get(agv_id)
            if line and self.solution:
                path_obj = self.solution.get(agv_id)
                if path_obj and path_obj.sequence:
                    segment_nodes = [state[0] for state in path_obj.sequence if state[1] <= frame]
                    if len(segment_nodes) > 1: x_coords = [node[0] for node in segment_nodes]; y_coords = [node[1] for node in segment_nodes]; line.set_data(x_coords, y_coords)
                    else: line.set_data([], [])
                    updated_artists.append(line)
        if self.time_text: self.time_text.set_text(f'Time: {time_step * self.config["delta_step"]:.1f}'); updated_artists.append(self.time_text)
        return tuple(updated_artists)

    def run_and_animate(self):
        """执行加载、求解和动画生成的完整流程。"""
        # --- 这部分代码与 v4.1 基本相同，只是路径构造使用 FilePathALN ---
        if not self._load_instance(): return
        if not self._get_solution(): return
        if not self._setup_plot(): return
        if self.max_time is None or self.max_time < 0: print("警告: 计算出的最大时间步无效或为负数，无法生成动画。"); return
        print(f"\n--- 生成动画 (共 {self.max_time + 1} 帧) ---"); num_frames = self.max_time + 1; interval = self.config['animation_interval_ms']
        def init_func():
             artists = list(self.agv_patches.values()) + list(self.trajectory_lines.values());
             if self.time_text: artists.append(self.time_text); self.time_text.set_text('')
             for line in self.trajectory_lines.values(): line.set_data([], [])
             for patch in self.agv_patches.values(): patch.center = (-1, -1)
             return tuple(artists)
        self.animation = animation.FuncAnimation(self.fig, self._update_animation, init_func=init_func, frames=num_frames, interval=interval, blit=True, repeat=False)
        if self.config['save_animation']:
            output_dir = FilePathALN(self.config['results_dir'])
            try: output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e: print(f"错误: 无法创建动画输出目录 '{output_dir}': {e}"); return
            instance_name = self.config.get('_instance_id_resolved', 'unknown_instance'); filename_base = self.config['output_filename_base']
            output_filename = f"{filename_base}_{instance_name}_{self.algorithm}.gif"; output_path = output_dir / output_filename; writer_name = 'pillow'
            try:
                print(f"尝试使用 '{writer_name}' 保存 GIF 动画到: {output_path}")
                def progress_callback(current_frame, total_frames):
                     if total_frames > 0:
                          percent = 100.0 * current_frame / total_frames
                          if current_frame % max(1, total_frames // 20) == 0 or current_frame == total_frames - 1: print(f"  保存进度: {percent:.1f}% ({current_frame+1}/{total_frames})", end='\r')
                          if current_frame == total_frames - 1: print("\n保存完成。")
                self.animation.save(str(output_path), writer=writer_name, dpi=150, progress_callback=progress_callback)
                print(f"动画已保存到: {output_path.resolve()}")
            except ImportError: print("错误: 保存 GIF 需要 Pillow 库。请运行 'pip install Pillow'。\n将尝试显示动画..."); plt.show()
            except Exception as e: print(f"错误: 保存动画失败 ({type(e).__name__})。\n  错误信息: {e}"); traceback.print_exc(); print("将尝试显示动画..."); plt.show()
        else: print("显示动画..."); plt.show()

# --- 入口点 ---
if __name__ == "__main__":
    if not _visual_libs_available or not _project_libs_available:
        print("检测到缺少必要的库或项目模块，无法运行动画生成器。")
        sys.exit(1)

    print("====== AGV 动画生成器 v4.2 ======") # 更新版本号
    config_to_use = DEFAULT_ANIMATOR_CONFIG.copy()

    # --- 添加命令行参数解析 ---
    parser = argparse.ArgumentParser(description="生成 AGV 路径动画")
    parser.add_argument(
        "--config",
        type=str,
        help="指定一个 JSON 配置文件路径 (例如 base_experiment_config_with_bbox.json)，"
             "其内容将覆盖默认配置。"
    )
    parser.add_argument("--scenario", type=int, choices=[1, 2], help="覆盖配置中的场景编号 (1 or 2)")
    parser.add_argument("--algorithm", type=str, choices=["ALNS", "Benchmark"], help="覆盖配置中的算法名称")
    parser.add_argument("--instance_id_s2", type=int, help="指定场景2的实例ID/运行编号 (用于命名和识别)")
    # 添加更多需要的命令行覆盖参数...
    # 例如: parser.add_argument("--max_iter", type=int, help="覆盖 ALNS 最大迭代次数")

    args = parser.parse_args()

    # --- 加载指定的配置文件 (如果提供) ---
    if args.config:
        config_path = FilePathALN(args.config)
        if config_path.is_file():
            print(f"\n--- 加载指定的运行配置: {config_path} ---")
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                # 更新基础配置，加载的配置优先
                config_to_use.update(loaded_config)
                print("JSON 配置加载成功。")
                # 处理潜在的 key 不一致 (示例: deadlock_max_wait)
                if 'alns_deadlock_max_wait' in config_to_use and 'conflict_deadlock_max_wait' not in loaded_config:
                     config_to_use['conflict_deadlock_max_wait'] = config_to_use['alns_deadlock_max_wait']
                     print("  (已将 JSON 中的 'alns_deadlock_max_wait' 映射到 'conflict_deadlock_max_wait')")
                # 可以忽略 JSON 中多余的 'buffer' 键，因为 'planner_buffer' 会被优先使用

            except Exception as e:
                print(f"错误: 加载配置文件 '{config_path}' 失败: {e}")
                print("将继续使用默认配置 (或后续命令行覆盖)。")
        else:
            print(f"警告: 配置文件 '{config_path}' 未找到。将继续使用默认配置 (或后续命令行覆盖)。")

    # --- 允许命令行参数覆盖加载的配置或默认值 (最高优先级) ---
    overridden_params = []
    if args.scenario is not None:
        config_to_use['scenario_to_animate'] = args.scenario
        overridden_params.append(f"场景={args.scenario}")
    if args.algorithm is not None:
        config_to_use['algorithm_to_animate'] = args.algorithm
        overridden_params.append(f"算法={args.algorithm}")
    if args.instance_id_s2 is not None:
         config_to_use['instance_id_s2'] = args.instance_id_s2
         config_to_use['instance_run_id_s2'] = args.instance_id_s2 # 保持一致
         overridden_params.append(f"场景2实例ID={args.instance_id_s2}")
    # if args.max_iter is not None:
    #      config_to_use['alns_max_iterations'] = args.max_iter
    #      overridden_params.append(f"ALNS最大迭代={args.max_iter}")

    if overridden_params:
        print(f"--- 命令行参数覆盖: {', '.join(overridden_params)} ---")
    # -----------------------------------------------------------

    print("\n--- 使用最终生效的配置参数 ---")
    # 打印关键配置以供确认
    print(f"  Scenario: {config_to_use['scenario_to_animate']}")
    print(f"  Algorithm: {config_to_use['algorithm_to_animate']}")
    if config_to_use['scenario_to_animate'] == 2:
        print(f"  Scenario 2 Instance ID/Run: {config_to_use.get('instance_id_s2', config_to_use.get('instance_run_id_s2'))}")
    if config_to_use['algorithm_to_animate'] == 'ALNS':
        print(f"  ALNS Max Iterations: {config_to_use['alns_max_iterations']}")
        print(f"  ALNS No Improvement Limit: {config_to_use['alns_no_improvement_limit']}")
    print(f"  Expansion Radius: {config_to_use['expansion_radius']}")
    print(f"  Cost Weights (alpha, beta, gamma_wait): {config_to_use['cost_weights']}")
    print("--------------------------------")


    try:
        # 使用最终确定的配置创建 Animator
        animator = Animator(config_to_use)
        animator.run_and_animate()
    except Exception as main_e:
        print(f"\n!!!!!! 动画生成过程中发生未处理的错误 !!!!!!")
        print(f"{type(main_e).__name__}: {main_e}")
        traceback.print_exc()

    print("\n====== 动画生成器运行结束 ======")
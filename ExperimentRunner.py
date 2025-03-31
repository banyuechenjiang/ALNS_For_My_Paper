# ExperimentRunner.py-v25
"""
实验运行与结果记录模块。
负责协调加载实验场景、运行指定算法 (ALNS 或 Benchmark)、收集性能指标，
并将结果保存到 CSV 文件。

与论文的关联:
- 实验执行: 作为执行论文实验部分描述的对比测试的脚本框架。
- 场景加载: 调用 InstanceGenerator (论文相关算例生成器) 加载固定或随机场景。
- 算法调用: 调用 ALNS (论文核心算法) 和 Benchmark (论文基准算法) 的 `run`/`plan` 方法。
- 结果记录: 记录的性能指标 (成本、时间、Makespan) 用于评估算法性能，
           其中成本指标直接关联论文的目标函数 (公式 1)。
- 配置管理: 支持通过 JSON 文件配置实验参数 (地图、AGV数量、算法参数等)，
           便于复现论文中的实验设置。

版本变更 (v24 -> v25):
- 添加了详细的模块和函数文档字符串，明确与论文实验框架和模型/算法的关联。
- 在关键逻辑处添加了内联注释，解释其作用。
- 清理了不再需要的注释。
- 确保代码风格一致性和完整性。
- 更新了依赖项版本说明 (ALNS v37, Benchmark v7, etc.)
"""
import time
import csv
import os
import sys
import traceback
import json
import argparse
from pathlib import Path as FilePath
from typing import Dict, Any, Optional, List, Tuple

# --- 标准导入 (使用更新后的模块) ---
try:
    # 依赖 v5 的 Map
    from Map import GridMap, Node
    # 依赖 v8 的 DataTypes
    from DataTypes import Task, Solution, CostDict
    # 依赖 v14 的 Planner
    from Planner import TWAStarPlanner
    # 依赖 v37 的 ALNS
    from ALNS import ALNS
    # 依赖 v7 的 Benchmark
    from Benchmark import PrioritizedPlanner
    # 依赖 v11 的 InstanceGenerator
    from InstanceGenerator import load_fixed_scenario_1, generate_scenario_2_instance
except ImportError as e:
    print(f"错误: 导入 ExperimentRunner 的依赖项失败: {e}")
    print("请确保所有模块 (Map-v5, DataTypes-v8, Planner-v14, ALNS-v37, Benchmark-v7, InstanceGenerator-v11) 都在 Python 路径中。")
    sys.exit(1)

# --- 默认配置参数 (与 v24 相同，但更新版本号) ---
DEFAULT_CONFIG: Dict[str, Any] = {
    "run_alns": True,
    "run_benchmark": True,
    "verbose_runner": True,
    "num_random_instances_s2": 3,
    "results_base_dir": "experiment_results_v25", # 更新版本号
    "expansion_radius": 1, # 对应论文假设 1
    "default_planner_timeout": 30.0,
    "cost_weights": (1.0, 0.3, 0.8), # (alpha, beta, gamma_wait) 对应模型公式 1
    "agv_v": 1.0, # 对应模型参数 v
    "delta_step": 1.0, # 对应模型参数 delta_step
    "max_time_horizon": 400, # 对应模型参数 T_max
    "planner_buffer": 0, # 对应论文 TWA* 的 buffer 参数 (如果实现)
    # ALNS 参数 (对应论文 3.8 节)
    "alns_max_iterations": 100,
    "alns_initial_temp": 10.0,
    "alns_cooling_rate": 0.98,
    "alns_segment_size": 20,
    "alns_weight_update_rate": 0.18,
    "alns_sigma1": 15.0, "alns_sigma2": 8.0, "alns_sigma3": 3.0,
    "alns_removal_percentage_min": 0.2,
    "alns_removal_percentage_max": 0.4,
    "alns_regret_k": 3,
    "alns_no_improvement_limit": 40,
    "alns_planner_time_limit_factor": 4.0,
    "alns_regret_planner_time_limit_abs": 0.1,
    "alns_verbose_output": False,
    "alns_debug_weights": False,
    "alns_record_history": True, # 控制 ALNS 是否记录内部历史
    "alns_plot_convergence": True, # 控制 ALNS 是否绘制收敛图
    # 冲突解决参数 (对应论文 3.6, 3.8 节)
    "conflict_wait_threshold": 6,
    "conflict_deadlock_max_wait": 10,
    # 场景 2 参数 (对应论文随机场景描述)
    "scenario2_map_width": 12,
    "scenario2_map_height": 12,
    "scenario2_obstacle_ratio": 0.1,
    "scenario2_num_agvs": 5,
    # 实验控制参数 (非论文内容，用于脚本自身)
    "run_scenario_1": True,
    "run_scenario_2": True,
}

# --- 辅助函数：创建结果目录 ---
def setup_results_dir(base_dir: str) -> FilePath:
    """
    (内部辅助) 创建或确认用于保存本次实验结果的目录。
    兼容由 SensitivityAnalysisRunner 调用时直接使用参数目录。

    Args:
        base_dir: 基础结果目录路径。

    Returns:
        最终使用的结果目录路径。
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = FilePath(base_dir)

    # 检查 base_dir 是否是 SensitivityRunner 创建的参数目录
    if results_path.exists() and results_path.name.startswith("param_"):
        final_path = results_path
        print(f"使用预设的参数目录: {final_path.resolve()}")
    else:
        # 否则，在 base_dir 下创建时间戳子目录
        final_path = results_path / f"experiment_{timestamp}"
        print(f"创建新的时间戳目录: {final_path.resolve()}")

    try:
        final_path.mkdir(parents=True, exist_ok=True)
        print(f"最终结果将保存到: {final_path.resolve()}")
        return final_path
    except OSError as e:
        print(f"错误: 无法创建结果目录 '{final_path}': {e}")
        sys.exit(1)

# --- 辅助函数：写入 CSV 结果 ---
def initialize_csv(filepath: FilePath) -> Tuple[Optional[csv.DictWriter], Optional[Any]]:
    """(内部辅助) 初始化 CSV 文件并写入表头。"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # 使用 'utf-8-sig' 编码确保 Excel 正确识别 UTF-8
        csvfile = open(filepath, 'w', newline='', encoding='utf-8-sig')
        # 定义 CSV 列名 (包含了算法参数和性能指标)
        fieldnames = [
            # 标识信息
            'Algorithm', 'Scenario', 'InstanceID', 'RunID',
            # 结果指标
            'Success', 'TotalCost', 'TravelCost', 'TurnCost', 'WaitCost',
            'CPUTime_s', 'Makespan',
            # 实例参数
            'NumAGVs_Param', 'ObstacleRatio_Param', 'ExpansionRadius_Param',
            # 模型/环境参数
            'CostWeightAlpha', 'CostWeightBeta', 'CostWeightGamma',
            # ALNS 特定参数 (用于记录，方便分析)
            'ALNS_MaxIter_Param', 'ALNS_NoImproveLimit_Param',
            'ALNS_CoolingRate_Param', 'ALNS_InitialTemp_Param',
            'ALNS_RegretK_Param', 'ALNS_RemovalMax_Param', 'ALNS_RemovalMin_Param',
            'ALNS_RegretLimitAbs_Param', 'ALNS_TimeFactor_Param',
            # 敏感性分析参数 (如果由 SensitivityRunner 调用)
            'ConfigParamName', 'ConfigParamValue'
        ]
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        return csv_writer, csvfile
    except IOError as e:
        print(f"错误: 无法打开或写入 CSV 文件 '{filepath}': {e}")
        return None, None

def write_result_to_csv(writer: csv.DictWriter, result_data: Dict[str, Any]):
    """(内部辅助) 将单次运行的结果写入 CSV 文件。"""
    if writer:
        try:
            # 确保所有字段都存在于字典中，缺失则填 'N/A'
            row_to_write = {field: result_data.get(field, 'N/A') for field in writer.fieldnames}
            writer.writerow(row_to_write)
        except Exception as e:
            print(f"错误: 写入 CSV 时出错: {e}")
            print("数据:", result_data)

# --- 单次实验运行函数 ---
def run_single_experiment(
    algorithm_name: str, scenario_num: int, instance_id: str, run_id: int,
    instance_data: Tuple['GridMap', List['Task']], config: Dict[str, Any],
    param_name: Optional[str] = None, # 用于敏感性分析
    param_value: Optional[Any] = None # 用于敏感性分析
) -> Dict[str, Any]:
    """
    运行指定算法在给定实例上一次，并返回结果字典。
    这是执行核心算法调用并收集性能指标的地方。

    Args:
        algorithm_name: 要运行的算法名称 ("ALNS" 或 "Benchmark")。
        scenario_num: 场景编号 (1 或 2)。
        instance_id: 实例的唯一标识符。
        run_id: 运行编号 (主要用于区分同一场景配置的多次随机运行)。
        instance_data: 包含 GridMap 和 Task 列表的元组。
        config: 包含所有配置参数的字典。
        param_name: (可选) 敏感性分析中变化的参数名。
        param_value: (可选) 敏感性分析中变化的参数值。

    Returns:
        一个包含运行结果和相关参数的字典。
    """
    grid_map, tasks = instance_data
    start_time = time.perf_counter() # 记录本次运行开始时间
    # --- 初始化结果字典 (包含所有要记录的字段) ---
    result_data: Dict[str, Any] = {
        'Algorithm': algorithm_name, 'Scenario': scenario_num, 'InstanceID': instance_id, 'RunID': run_id,
        'Success': False, # 默认失败
        'TotalCost': float('inf'), 'TravelCost': float('inf'), 'TurnCost': float('inf'), 'WaitCost': float('inf'), # 对应公式 1
        'CPUTime_s': 0.0, 'Makespan': float('inf'),
        'NumAGVs_Param': len(tasks),
        'ObstacleRatio_Param': config.get('scenario2_obstacle_ratio', 'N/A') if scenario_num == 2 else 'N/A',
        'ExpansionRadius_Param': config.get('expansion_radius', 'N/A'),
        'CostWeightAlpha': config.get('cost_weights', [None]*3)[0],
        'CostWeightBeta': config.get('cost_weights', [None]*3)[1],
        'CostWeightGamma': config.get('cost_weights', [None]*3)[2],
        'ALNS_MaxIter_Param': config.get('alns_max_iterations', 'N/A') if algorithm_name == 'ALNS' else 'N/A',
        'ALNS_NoImproveLimit_Param': config.get('alns_no_improvement_limit', 'N/A') if algorithm_name == 'ALNS' else 'N/A',
        'ALNS_CoolingRate_Param': config.get('alns_cooling_rate', 'N/A') if algorithm_name == 'ALNS' else 'N/A',
        'ALNS_InitialTemp_Param': config.get('alns_initial_temp', 'N/A') if algorithm_name == 'ALNS' else 'N/A',
        'ALNS_RegretK_Param': config.get('alns_regret_k', 'N/A') if algorithm_name == 'ALNS' else 'N/A',
        'ALNS_RemovalMax_Param': config.get('alns_removal_percentage_max', 'N/A') if algorithm_name == 'ALNS' else 'N/A',
        'ALNS_RemovalMin_Param': config.get('alns_removal_percentage_min', 'N/A') if algorithm_name == 'ALNS' else 'N/A',
        'ALNS_RegretLimitAbs_Param': config.get('alns_regret_planner_time_limit_abs', 'N/A') if algorithm_name == 'ALNS' else 'N/A',
        'ALNS_TimeFactor_Param': config.get('alns_planner_time_limit_factor', 'N/A') if algorithm_name == 'ALNS' else 'N/A',
        'ConfigParamName': param_name if param_name else 'N/A',
        'ConfigParamValue': param_value if param_value else 'N/A',
    }

    final_solution: Optional[Solution] = None
    best_cost_dict: CostDict = {'total': float('inf'), 'travel': float('inf'), 'turn': float('inf'), 'wait': float('inf')}
    cpu_time_reported: float = 0.0 # 算法内部报告的 CPU 时间

    # --- 调用核心算法 ---
    try:
        # --- 调用 ALNS (对应论文核心算法) ---
        if algorithm_name == "ALNS":
            # 准备 ALNS 初始化所需的参数
            planner_instance = TWAStarPlanner() # 创建规划器实例
            alns_init_args = {
                'instance_identifier': instance_id, # 传递实例标识符
                'results_dir': str(config['_current_run_results_dir']), # 传递结果目录
                # 传递所有以 'alns_' 开头的参数，以及其他 ALNS 需要的参数
                **{k: v for k, v in config.items() if k.startswith('alns_')}, # ALNS 特定参数
                'max_time': config['max_time_horizon'], # 环境参数
                'cost_weights': config['cost_weights'],
                'v': config['agv_v'],
                'delta_step': config['delta_step'],
                'buffer': config['planner_buffer'],
                'wait_threshold': config['conflict_wait_threshold'], # 冲突解决参数
                'deadlock_max_wait': config['conflict_deadlock_max_wait'],
            }
            # 使用 v37 的 ALNS
            alns_runner = ALNS(grid_map=grid_map, tasks=tasks, planner=planner_instance, **alns_init_args)
            # 执行 ALNS 的 run 方法
            final_solution, cpu_time_reported, best_cost_dict = alns_runner.run()

        # --- 调用 Benchmark (对应论文基准算法) ---
        elif algorithm_name == "Benchmark":
            planner_instance = TWAStarPlanner() # 创建规划器实例
            # 使用 v7 的 Benchmark
            benchmark_runner = PrioritizedPlanner(
                grid_map=grid_map, tasks=tasks, planner=planner_instance,
                v=config['agv_v'], delta_step=config['delta_step']
            )
            # 执行 Benchmark 的 plan 方法
            final_solution, cpu_time_reported, best_cost_dict = benchmark_runner.plan(
                cost_weights=config['cost_weights'],
                max_time=config['max_time_horizon'],
                time_limit_per_agent=config['default_planner_timeout']
            )
        else:
            print(f"错误: 未知算法名称 '{algorithm_name}'")

    except Exception as e:
        # 捕获算法运行中的任何异常
        print(f"!!!!!!!!!! [ {time.strftime('%H:%M:%S')} ] 运行实验时发生严重错误 !!!!!!!!!!")
        print(f"Alg={algorithm_name}, Scen={scenario_num}, Inst={instance_id}, Run={run_id}")
        print(f"Param={param_name}, Value={param_value}")
        print(f"错误: {type(e).__name__}: {e}")
        traceback.print_exc()
        result_data['Success'] = False
        result_data['CPUTime_s'] = time.perf_counter() - start_time # 记录总时间
        return result_data

    # --- 处理和记录结果 ---
    run_duration_total = time.perf_counter() - start_time # 本次运行总墙上时间
    result_data['CPUTime_s'] = cpu_time_reported # 记录算法内部报告的 CPU 时间

    # 检查解是否有效 (找到解且成本有效)
    if final_solution and best_cost_dict.get('total', float('inf')) != float('inf'):
        result_data['Success'] = True
        # 记录成本分量 (对应公式 1)
        result_data['TotalCost'] = best_cost_dict.get('total', float('inf'))
        result_data['TravelCost'] = best_cost_dict.get('travel', float('inf'))
        result_data['TurnCost'] = best_cost_dict.get('turn', float('inf'))
        result_data['WaitCost'] = best_cost_dict.get('wait', float('inf'))
        # 计算并记录 Makespan (完成所有任务的最晚时间)
        max_makespan = 0
        for path in final_solution.values():
            if path and path.sequence: max_makespan = max(max_makespan, path.get_makespan())
        result_data['Makespan'] = max_makespan
        status = "Success"
    else:
        result_data['Success'] = False
        status = "Failed"

    # 打印本次运行的简要结果 (如果启用 verbose)
    if config.get('verbose_runner', False):
        cost_str = f"{result_data['TotalCost']:.2f}" if result_data['Success'] else "Inf"
        time_str = f"{result_data['CPUTime_s']:.4f}s (Wall: {run_duration_total:.2f}s)"
        makespan_str = str(result_data['Makespan']) if result_data['Success'] else "Inf"
        param_info = f" (Param: {param_name}={param_value})" if param_name else ""
        print(f"<<< [ {time.strftime('%H:%M:%S')} ] 完成运行: {algorithm_name} | 实例: {instance_id}{param_info}")
        print(f"    结果: {status}, Cost={cost_str}, Time={time_str}, Makespan={makespan_str}")

    return result_data


# --- 主函数 (v25 - 增强文档) ---
def main(config_override: Optional[Dict[str, Any]] = None):
    """
    主实验流程控制函数。
    负责加载配置、设置结果目录、循环运行场景和算法、记录结果到 CSV。
    """
    script_start_time = time.time() # 记录脚本整体开始时间
    print(f"实验开始 @ {time.strftime('%Y%m%d_%H%M%S')}")

    # --- 1. 参数解析和配置加载 ---
    parser = argparse.ArgumentParser(description="运行仓储 AGV 路径规划实验")
    # 支持通过命令行 --config 加载 JSON 配置文件
    parser.add_argument('--config', type=str, help="指定要加载的 JSON 配置文件路径")
    args = parser.parse_args()
    config = DEFAULT_CONFIG.copy() # 使用默认配置作为基础
    # 如果提供了配置文件路径，尝试加载并覆盖默认配置
    if args.config:
        try:
            config_path = FilePath(args.config)
            if config_path.is_file():
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f); config.update(loaded_config)
                    print(f"已从 '{args.config}' 加载配置。")
            else: print(f"警告: 配置文件 '{args.config}' 未找到，将使用默认配置。")
        except Exception as e: print(f"错误: 加载配置文件 '{args.config}' 时出错: {e}\n将使用默认配置。")
    # 如果直接传递了配置字典 (例如由 SensitivityRunner 调用)，则覆盖
    if config_override:
         config.update(config_override); print("已应用直接传递的配置覆盖。")

    # --- 2. 设置结果目录 ---
    # setup_results_dir 会处理普通运行和 SensitivityRunner 调用的情况
    results_dir_path = setup_results_dir(config['results_base_dir'])
    # 将最终使用的结果目录路径存入 config，供内部使用 (如 ALNS 保存历史文件)
    config['_current_run_results_dir'] = results_dir_path

    # --- 3. 初始化 CSV 文件 ---
    # 确定 CSV 文件名 (允许 SensitivityRunner 指定固定名称)
    default_csv_filename = f"experiment_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_filename = config.get('_csv_filename', default_csv_filename)
    csv_filepath = results_dir_path / csv_filename # CSV 文件保存在最终结果目录下
    csv_writer, csvfile = initialize_csv(csv_filepath)
    if not csv_writer or not csvfile: print("错误: 无法初始化 CSV 文件，实验终止。"); return

    # 打印最终使用的配置参数
    print("\n--- 使用的最终配置参数 ---")
    for key, value in config.items():
        # 不打印内部使用的临时键
        if not key.startswith('_'): print(f"  {key}: {value}")
    print("-------------------------")

    # 获取敏感性分析参数信息 (如果存在)
    config_param_name = config.get('_config_param_name', None)
    config_param_value = config.get('_config_param_value', None)

    # --- 4. 运行实验 ---
    try:
        # --- 运行场景 1 (手动设计的固定场景) ---
        if config.get("run_scenario_1", DEFAULT_CONFIG["run_scenario_1"]):
            print(f"\n========== [SCENARIO 1 START] @ {time.strftime('%H:%M:%S')} (对应论文固定场景) ==========")
            exp_r_s1 = config['expansion_radius']
            instance_id_s1 = f"Fixed_10x10_ExpR{exp_r_s1}"
            # 加载场景 1 实例
            instance_s1_data = load_fixed_scenario_1(expansion_radius=exp_r_s1)
            if instance_s1_data:
                # 运行 ALNS (如果配置允许)
                if config.get('run_alns', DEFAULT_CONFIG['run_alns']):
                    print(f"\n>>> [ {time.strftime('%H:%M:%S')} ] 开始运行: ALNS | 场景: 1 | 实例: {instance_id_s1} | Run: 1")
                    result_alns_s1 = run_single_experiment("ALNS", 1, instance_id_s1, 1, instance_s1_data, config, config_param_name, config_param_value)
                    write_result_to_csv(csv_writer, result_alns_s1); csvfile.flush() # 写入并刷新缓冲区
                # 运行 Benchmark (如果配置允许)
                if config.get('run_benchmark', DEFAULT_CONFIG['run_benchmark']):
                    print(f"\n>>> [ {time.strftime('%H:%M:%S')} ] 开始运行: Benchmark | 场景: 1 | 实例: {instance_id_s1} | Run: 1")
                    result_bench_s1 = run_single_experiment("Benchmark", 1, instance_id_s1, 1, instance_s1_data, config, config_param_name, config_param_value)
                    write_result_to_csv(csv_writer, result_bench_s1); csvfile.flush()
            else: print(f"警告: 无法加载场景 1 实例数据 ({instance_id_s1})。跳过场景 1。")
            print(f"========== [SCENARIO 1 END] @ {time.strftime('%H:%M:%S')} ==========")
        else: print("\n跳过场景 1 (根据配置)。")

        # --- 运行场景 2 (随机生成的场景) ---
        if config.get("run_scenario_2", DEFAULT_CONFIG["run_scenario_2"]):
            # 获取场景 2 配置参数
            s2_width = config['scenario2_map_width']; s2_height = config['scenario2_map_height']
            s2_obs_ratio = config['scenario2_obstacle_ratio']; s2_num_agvs = config['scenario2_num_agvs']
            s2_exp_r = config['expansion_radius']; num_instances = config['num_random_instances_s2']
            print(f"\n========== [SCENARIO 2 START] @ {time.strftime('%H:%M:%S')} (对应论文随机场景, N={num_instances}) ==========")
            successful_s2_runs = 0
            # 循环生成并运行指定数量的随机实例
            for i in range(num_instances):
                instance_num = i + 1
                print(f"\n--- [ {time.strftime('%H:%M:%S')} ] Generating Scenario 2 Instance {instance_num}/{num_instances} ---")
                # 生成实例 ID
                instance_id_s2_base = f"Random_{s2_width}x{s2_height}_Obs{s2_obs_ratio*100:.0f}p_AGV{s2_num_agvs}_ExpR{s2_exp_r}"
                instance_id_s2 = f"{instance_id_s2_base}_#{instance_num}"
                # 生成随机实例数据
                instance_s2_data = generate_scenario_2_instance(
                    width=s2_width, height=s2_height, obstacle_ratio=s2_obs_ratio,
                    num_agvs=s2_num_agvs, expansion_radius=s2_exp_r
                )
                if instance_s2_data:
                    successful_s2_runs += 1
                    # 运行 ALNS
                    if config.get('run_alns', DEFAULT_CONFIG['run_alns']):
                        print(f"\n>>> [ {time.strftime('%H:%M:%S')} ] 开始运行: ALNS | 场景: 2 | 实例: {instance_id_s2} | Run: {instance_num}")
                        result_alns_s2 = run_single_experiment("ALNS", 2, instance_id_s2, instance_num, instance_s2_data, config, config_param_name, config_param_value)
                        write_result_to_csv(csv_writer, result_alns_s2); csvfile.flush()
                    # 运行 Benchmark
                    if config.get('run_benchmark', DEFAULT_CONFIG['run_benchmark']):
                        print(f"\n>>> [ {time.strftime('%H:%M:%S')} ] 开始运行: Benchmark | 场景: 2 | 实例: {instance_id_s2} | Run: {instance_num}")
                        result_bench_s2 = run_single_experiment("Benchmark", 2, instance_id_s2, instance_num, instance_s2_data, config, config_param_name, config_param_value)
                        write_result_to_csv(csv_writer, result_bench_s2); csvfile.flush()
                else: print(f"警告: 无法生成场景 2 实例 {instance_num} ({instance_id_s2}) 的数据。跳过此实例。")
            print(f"--- Scenario 2: Successfully generated and ran {successful_s2_runs}/{num_instances} instances ---")
            print(f"========== [SCENARIO 2 END] @ {time.strftime('%H:%M:%S')} ==========")
        else: print("\n跳过场景 2 (根据配置)。")

    finally:
        # --- 5. 收尾工作 ---
        # 关闭 CSV 文件
        if csvfile:
             print(f"\n结果已写入 CSV 文件: {csv_filepath.resolve()}")
             csvfile.close()
        script_end_time = time.time()
        total_script_duration = script_end_time - script_start_time
        print(f"\n实验完成 @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {total_script_duration:.2f} 秒")

# --- 脚本入口 ---
if __name__ == "__main__":
    # 直接运行时，config_override 为 None，将使用默认配置或命令行指定的配置
    main(config_override=None)
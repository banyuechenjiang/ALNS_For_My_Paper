# SensitivityAnalysisRunner.py-v5
"""
自动化运行参数敏感性分析实验并可视化结果。
- v5: 修正 collect_and_aggregate_results 函数，在 param_* 子目录内
       **递归地**查找目标 CSV 文件名，以适应 ExperimentRunner 可能创建的
       额外时间戳子目录。
- 保持与 ExperimentRunner.py-v24 的兼容性。
"""
import subprocess
import json
import itertools
import os
import shutil
import time
import sys
from pathlib import Path as FilePath
import copy
from typing import Dict, Any, Optional, List, Tuple

# --- 尝试导入可选库 (不变) ---
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    _visual_libs_available = True
except ImportError:
    _visual_libs_available = False
    print("警告: 未找到 pandas 或 matplotlib。结果收集和绘图功能将被禁用。")
    print("请使用 'pip install pandas matplotlib' 安装。")

# --- 配置 (不变) ---
CONFIG_SENSITIVITY: Dict[str, Any] = {
    "parameter_ranges": {
        "alns_cooling_rate": [0.95, 0.98, 0.99, 0.995],
        "alns_removal_percentage_max": [0.3, 0.4, 0.5, 0.6],
        "alns_no_improvement_limit": [20, 40, 60, 80],
    },
    "base_config_path": "base_experiment_config.json",
    "experiment_runner_script": "ExperimentRunner.py",
    "results_base_dir": "sensitivity_analysis_results",
    "scenario_to_run": 2,
    "num_instances_per_setting": 5,
    "run_alns_only": True,
    "run_benchmark_in_er": False,
    "run_scenario_1_in_er": False,
    "run_scenario_2_in_er": True,
    "disable_alns_history_in_er": True,
    "disable_alns_plotting_in_er": True,
    "fixed_csv_filename_in_er": "results.csv",
}

# --- 辅助函数 (run_command, setup_analysis_directories, load_base_config, generate_temp_config_path - 不变) ---
def run_command(command: List[str], cwd: Optional[str] = None) -> bool:
    """运行外部命令并捕获输出。"""
    print(f"\n--- 执行命令 ---\n{' '.join(command)}\n-----------------")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd, encoding='utf-8', errors='ignore')
        # 只在有内容时打印 stdout/stderr
        output = process.stdout.strip()
        error_output = process.stderr.strip()
        if output:
            print("命令输出:")
            print(output)
        if error_output:
            print("命令错误输出 (可能包含警告):")
            print(error_output)
        print("-----------------")
        return True
    except FileNotFoundError:
        print(f"错误: 命令未找到: {command[0]}。请确保它在 PATH 中或提供了正确路径。")
        return False
    except subprocess.CalledProcessError as e:
        print(f"错误: 命令执行失败，返回码: {e.returncode}")
        print("命令输出:")
        print(e.stdout)
        print("命令错误输出:")
        print(e.stderr)
        print("-----------------")
        return False
    except Exception as e:
        print(f"错误: 执行命令时发生未知错误: {e}")
        return False

def setup_analysis_directories(base_dir: str) -> Tuple[Optional[FilePath], Optional[FilePath]]:
    """创建分析结果主目录和临时配置目录。"""
    results_dir = FilePath(base_dir)
    temp_config_dir = results_dir / "_temp_configs"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        temp_config_dir.mkdir(parents=True, exist_ok=True)
        print(f"敏感性分析结果将保存在: {results_dir.resolve()}")
        return results_dir, temp_config_dir
    except OSError as e:
        print(f"错误: 无法创建目录 '{results_dir}' 或 '{temp_config_dir}': {e}")
        return None, None

def load_base_config(config_path: str) -> Optional[Dict[str, Any]]:
    """加载基础 JSON 配置文件。"""
    path = FilePath(config_path)
    if not path.is_file():
        print(f"错误: 基础配置文件 '{config_path}' 未找到。")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            base_config = json.load(f)
        print(f"已加载基础配置: {config_path}")
        return base_config
    except json.JSONDecodeError as e:
        print(f"错误: 解析基础配置文件 '{config_path}' 失败: {e}")
        return None
    except Exception as e:
        print(f"错误: 读取基础配置文件 '{config_path}' 时出错: {e}")
        return None

def generate_temp_config_path(temp_dir: FilePath, param_name: str, param_value: Any) -> FilePath:
    """生成临时的配置文件路径。"""
    safe_value = str(param_value).replace('.', '_').replace('-', 'neg')
    filename = f"temp_config_{param_name}_{safe_value}.json"
    return temp_dir / filename

# --- 结果收集与聚合函数 (v5: 递归查找 CSV) ---
def collect_and_aggregate_results(results_base_dir: FilePath, expected_csv_filename: str) -> Optional[pd.DataFrame]:
    """
    收集所有实验运行的 CSV 结果并进行聚合。
    v5: 在 param_* 子目录内递归查找目标 CSV 文件名。
    """
    if not _visual_libs_available:
        print("依赖库 pandas 未加载，无法收集和聚合结果。")
        return None
    if not results_base_dir.is_dir():
        print(f"错误: 结果目录 '{results_base_dir}' 不存在或不是一个目录，无法收集数据。")
        return None

    all_data = []
    processed_files = 0

    print(f"\n--- 开始在 '{results_base_dir}' 下的 'param_*' 子目录内递归查找 '{expected_csv_filename}' ---")
    param_subdirs = [d for d in results_base_dir.iterdir() if d.is_dir() and d.name.startswith("param_")]
    print(f"发现 {len(param_subdirs)} 个参数子目录。")

    if not param_subdirs:
        print("警告: 未找到任何 'param_*' 子目录。请检查 ExperimentRunner 是否创建了正确的目录结构。")
        return None

    for param_dir in param_subdirs:
        print(f"  检查目录: {param_dir.name}")
        # v5: 使用 rglob 进行递归查找
        found_csv_files = list(param_dir.rglob(expected_csv_filename))

        if not found_csv_files:
            print(f"    警告: 在目录 '{param_dir.name}' 及其子目录中未找到文件 '{expected_csv_filename}'。")
            continue # 继续检查下一个 param_dir

        # 通常预期只找到一个，如果找到多个则发出警告并处理第一个
        if len(found_csv_files) > 1:
            print(f"    警告: 在目录 '{param_dir.name}' 下递归找到多个 '{expected_csv_filename}' 文件。")
            print(f"    将只处理第一个: {found_csv_files[0].relative_to(results_base_dir)}")
            # 你可以在这里添加更复杂的逻辑来选择文件，例如基于修改时间或路径深度

        csv_file_path = found_csv_files[0] # 处理找到的第一个文件
        print(f"    找到文件: {csv_file_path.relative_to(results_base_dir)}")

        try:
            df_temp = pd.read_csv(csv_file_path)
            # 从 param_* 目录名解析参数名和值 (这仍然是正确的)
            param_dir_name = param_dir.name
            parts = param_dir_name[len("param_"):].split('=', 1)
            if len(parts) == 2:
                param_name_parsed = parts[0]
                # 将安全值转换回可能的形式，用于数据处理
                param_value_parsed = parts[1].replace('_', '.').replace('neg', '-')
                # 确保 DataFrame 包含这些列
                df_temp['ConfigParamName'] = param_name_parsed
                df_temp['ConfigParamValue'] = param_value_parsed
                all_data.append(df_temp)
                processed_files += 1
            else:
                print(f"警告: 无法从目录名 '{param_dir_name}' 解析参数，跳过文件: {csv_file_path.relative_to(results_base_dir)}")

        except pd.errors.EmptyDataError:
            print(f"警告: CSV 文件为空，跳过: {csv_file_path.relative_to(results_base_dir)}")
        except KeyError as e:
            print(f"警告: 读取 CSV 文件时缺少关键列 '{e}'，跳过: {csv_file_path.relative_to(results_base_dir)}")
        except Exception as e:
            print(f"错误: 读取或处理 CSV 文件时出错: {csv_file_path.relative_to(results_base_dir)} - {type(e).__name__}: {e}")

    if not all_data:
        print(f"错误: 未能在任何参数子目录中成功加载 '{expected_csv_filename}' 文件。")
        return None

    print(f"\n成功处理了 {processed_files} 个 '{expected_csv_filename}' 文件。")
    full_df = pd.concat(all_data, ignore_index=True)

    # --- 数据类型转换和聚合 (与 v4 相同) ---
    numeric_cols = ['TotalCost', 'TravelCost', 'TurnCost', 'WaitCost', 'CPUTime_s', 'Makespan']
    for col in numeric_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    try:
        # 更全面的布尔值映射
        bool_map = {'True': True, 'true': True, '1': True, 'False': False, 'false': False, '0': False, True: True, False: False}
        if 'Success' in full_df.columns:
             # 先转为字符串处理可能的混合类型或非标准表示
             full_df['Success'] = full_df['Success'].astype(str).map(bool_map).fillna(False).astype(bool)
        else:
            print("警告: 结果数据中缺少 'Success' 列。")
            full_df['Success'] = False
    except Exception as e:
        print(f"警告: 转换 'Success' 列为布尔值时出错: {e}。将假定为 False。")
        full_df['Success'] = False

    full_df['ConfigParamValueNumeric'] = pd.to_numeric(full_df['ConfigParamValue'], errors='coerce')
    full_df['ConfigParamValueStr'] = full_df['ConfigParamValue'].astype(str) # 保留字符串形式

    if 'ConfigParamName' not in full_df.columns or 'ConfigParamValueStr' not in full_df.columns:
         print("错误：合并后的 DataFrame 缺少 'ConfigParamName' 或 'ConfigParamValueStr' 列，无法进行分组。")
         return None

    # 按字符串分组，保留数值用于排序
    grouped = full_df.groupby(['ConfigParamName', 'ConfigParamValueStr'])

    summary = grouped.agg(
        ConfigParamValueNumeric=('ConfigParamValueNumeric', 'first'), # 保留数值用于排序
        Mean_TotalCost=('TotalCost', 'mean'),
        Std_TotalCost=('TotalCost', 'std'),
        Mean_CPUTime_s=('CPUTime_s', 'mean'),
        Std_CPUTime_s=('CPUTime_s', 'std'),
        Mean_Makespan=('Makespan', 'mean'),
        Std_Makespan=('Makespan', 'std'),
        SuccessRate=('Success', 'mean'),
        RunCount=('Algorithm', 'size') # 使用保证存在的列
    ).reset_index() # ConfigParamValueStr 变为普通列

    # 优先按数值排序，然后按字符串
    summary = summary.sort_values(by=['ConfigParamName', 'ConfigParamValueNumeric', 'ConfigParamValueStr'],
                                  na_position='last')

    summary_file = results_base_dir / "sensitivity_summary.csv"
    try:
        # 保存时格式化浮点数
        summary.to_csv(summary_file, index=False, float_format='%.4f')
        print(f"\n聚合结果已保存到: {summary_file}")
    except Exception as e:
        print(f"错误: 保存聚合结果 CSV 时出错: {e}")

    return summary

# --- 绘图函数 (plot_sensitivity_curves - 与 v4 相同) ---
def plot_sensitivity_curves(summary_df: pd.DataFrame, results_base_dir: FilePath):
    """根据聚合结果绘制敏感性曲线图。"""
    if not _visual_libs_available:
        print("依赖库 matplotlib 未加载，无法绘图。")
        return
    if summary_df is None or summary_df.empty:
        print("无聚合数据可用于绘图。")
        return

    param_names = summary_df['ConfigParamName'].unique()
    metrics_to_plot = {
        'TotalCost': ('Mean_TotalCost', 'Std_TotalCost'),
        'CPUTime_s': ('Mean_CPUTime_s', 'Std_CPUTime_s'),
        'Makespan': ('Mean_Makespan', 'Std_Makespan'),
        'SuccessRate': ('SuccessRate', None)
    }

    plot_dir = results_base_dir / "_plots"
    try:
        plot_dir.mkdir(exist_ok=True)
    except OSError as e:
        print(f"错误: 无法创建绘图目录 '{plot_dir}': {e}")
        return

    print("\n--- 生成敏感性图表 ---")
    for param_name in param_names:
        param_data = summary_df[summary_df['ConfigParamName'] == param_name].copy()
        if param_data.empty: continue

        num_metrics = len(metrics_to_plot)
        # 适应性调整子图布局
        ncols = 2 if num_metrics > 1 else 1
        nrows = (num_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, squeeze=False) # squeeze=False 确保 axes 总是二维数组
        axes_flat = axes.flatten() # 展平以便索引

        fig_title = f'Sensitivity Analysis for Parameter: {param_name}'
        fig.suptitle(fig_title, fontsize=16, y=1.02) # 调整 y 位置

        x_values_str = param_data['ConfigParamValueStr']
        x_values_num = param_data['ConfigParamValueNumeric']
        is_numeric_x = not x_values_num.isnull().all()

        if is_numeric_x:
            x_plot = x_values_num
            x_label = f'Parameter Value: {param_name}'
            # 如果是数值型X轴，确保按数值顺序绘图
            param_data = param_data.sort_values(by='ConfigParamValueNumeric')
            x_plot = param_data['ConfigParamValueNumeric']
        else:
            x_plot = param_data['ConfigParamValueStr']
            x_label = f'Parameter Value (Category): {param_name}'
            # 如果是类别型X轴，按字符串顺序绘图
            param_data = param_data.sort_values(by='ConfigParamValueStr')
            x_plot = param_data['ConfigParamValueStr']


        for i, (metric_label, (mean_col, std_col)) in enumerate(metrics_to_plot.items()):
            ax = axes_flat[i]
            if mean_col not in param_data.columns:
                 print(f"警告: 绘图时找不到列 '{mean_col}' for param '{param_name}'，跳过 {metric_label} 图。")
                 continue
            mean_values = param_data[mean_col]
            std_values = param_data[std_col] if std_col and std_col in param_data.columns else None
            plot_error_bars = (std_values is not None) and (not std_values.isnull().all())

            # 绘图
            if plot_error_bars:
                ax.errorbar(x_plot.astype(str) if not is_numeric_x else x_plot, # 确保x轴类型匹配
                            mean_values, yerr=std_values, fmt='-o', capsize=5, label=f'Mean {metric_label}', ecolor='lightgray', elinewidth=3)
            else:
                ax.plot(x_plot.astype(str) if not is_numeric_x else x_plot, # 确保x轴类型匹配
                        mean_values, '-o', label=f'Mean {metric_label}')

            ax.set_ylabel(metric_label)
            ax.set_title(f'{metric_label} vs {param_name}')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            run_counts = param_data.get('RunCount', pd.Series([0] * len(param_data)))
            # 添加运行次数标签
            for idx, count in enumerate(run_counts):
                 x_pos = x_plot.iloc[idx]
                 y_pos = mean_values.iloc[idx]
                 if pd.notna(y_pos) and pd.notna(x_pos):
                      ax_x_pos = str(x_pos) if not is_numeric_x else x_pos # 用于文本定位
                      ax.text(ax_x_pos, y_pos, f' n={count}', fontsize=8, verticalalignment='bottom', horizontalalignment='left', alpha=0.7)

            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f')) # 更高精度
            # X 轴标签旋转（如果需要）
            if not is_numeric_x or len(x_plot) > 5: # 类别轴或点过多时旋转
                 plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        # 隐藏多余的子图轴
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        # 在最后一个可见的子图上设置 x 轴标签
        last_visible_ax = axes_flat[i]
        last_visible_ax.set_xlabel(x_label)
        if not is_numeric_x or len(x_plot) > 5:
             plt.setp(last_visible_ax.get_xticklabels(), rotation=30, ha='right')


        fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # 调整 rect 避免标题重叠
        plot_filename = plot_dir / f"sensitivity_{param_name}.png"
        try:
            plt.savefig(plot_filename)
            print(f"图表已保存: {plot_filename}")
        except Exception as e:
            print(f"错误: 保存图表失败 '{plot_filename}': {e}")
        plt.close(fig)

# --- 主分析函数 (run_sensitivity_analysis - 与 v4 相同) ---
def run_sensitivity_analysis(config: Dict[str, Any]):
    """执行完整的参数敏感性分析流程。"""
    print("====== 开始参数敏感性分析 ======")
    analysis_start_time = time.time()

    # 1. 设置目录
    results_dir, temp_config_dir = setup_analysis_directories(config['results_base_dir'])
    if not results_dir or not temp_config_dir: return

    # 2. 加载基础配置
    base_config = load_base_config(config['base_config_path'])
    if not base_config: return

    # 3. 准备运行器脚本路径
    runner_script_path = FilePath(config['experiment_runner_script'])
    if not runner_script_path.is_file():
        print(f"错误: 实验运行器脚本 '{runner_script_path}' 未找到。")
        return
    runner_script_abs_path = str(runner_script_path.resolve())

    # 4. 迭代参数并运行实验
    parameters_to_test = config['parameter_ranges']
    total_settings = sum(len(v) for v in parameters_to_test.values())
    setting_count = 0
    failed_settings = []
    expected_csv_filename = config.get('fixed_csv_filename_in_er', 'results.csv')

    for param_name, param_values in parameters_to_test.items():
        print(f"\n===== 分析参数: {param_name} =====")
        for param_value in param_values:
            setting_count += 1
            print(f"\n--- 测试设置 {setting_count}/{total_settings}: {param_name} = {param_value} ---")

            # a. 创建临时配置
            current_config = copy.deepcopy(base_config)
            current_config[param_name] = param_value

            # b. 定义此设置的结果子目录
            safe_param_name = param_name.replace('/', '_')
            safe_param_value = str(param_value).replace('.', '_').replace('-', 'neg')
            setting_results_dir_name = f"param_{safe_param_name}={safe_param_value}"
            setting_results_path = results_dir / setting_results_dir_name
            # 传递给 ER 的重要配置
            current_config['results_base_dir'] = str(setting_results_path)
            current_config['_config_param_name'] = param_name
            current_config['_config_param_value'] = param_value
            current_config['_csv_filename'] = expected_csv_filename
            # 确保 ER 只运行需要的场景/算法
            current_config['run_scenario_1'] = config.get('run_scenario_1_in_er', False)
            current_config['run_scenario_2'] = config.get('run_scenario_2_in_er', True)
            current_config['run_alns'] = config.get('run_alns_only', True)
            current_config['run_benchmark'] = config.get('run_benchmark_in_er', False)
            current_config['num_random_instances_s2'] = config.get('num_instances_per_setting', 5)
            # 控制 ER 内部 ALNS 的行为
            current_config['alns_record_history'] = not config.get('disable_alns_history_in_er', True)
            current_config['alns_plot_convergence'] = not config.get('disable_alns_plotting_in_er', True)
            current_config['verbose_runner'] = False # 减少 ER 输出

            # c. 保存临时配置文件
            temp_config_path = generate_temp_config_path(temp_config_dir, param_name, param_value)
            try:
                temp_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_config_path, 'w', encoding='utf-8') as f:
                    json.dump(current_config, f, indent=4)
            except Exception as e:
                print(f"错误: 无法写入临时配置文件 '{temp_config_path}': {e}")
                failed_settings.append(f"{param_name}={param_value} (Config Save Error)")
                continue

            # d. 构建并运行命令
            command = [ sys.executable, runner_script_abs_path, "--config", str(temp_config_path.resolve()) ]
            run_success = run_command(command)

            if not run_success:
                print(f"警告: ExperimentRunner 执行失败，参数设置: {param_name}={param_value}。")
                failed_settings.append(f"{param_name}={param_value} (ExperimentRunner Failed)")

    print(f"\n====== 所有 {setting_count} 个参数设置运行完毕 ======")
    if failed_settings:
        print("\n--- 以下设置的 ExperimentRunner 执行失败 ---")
        for setting in failed_settings: print(f"  - {setting}")
        print("-----------------------------------------")

    # 5. 清理临时配置目录
    print(f"\n--- 清理临时配置目录: {temp_config_dir} ---")
    try:
        shutil.rmtree(temp_config_dir)
        print(f"临时配置目录已成功清理。")
    except OSError as e:
        print(f"警告: 清理临时配置目录时出错: {e}")

    # 6. 收集、聚合结果并绘图
    if _visual_libs_available:
        print("\n--- 开始收集和聚合结果 ---")
        summary_df = collect_and_aggregate_results(results_dir, expected_csv_filename)
        if summary_df is not None:
            plot_sensitivity_curves(summary_df, results_dir)
        else:
            print("未能生成聚合数据，跳过绘图。")
    else:
        print("\n跳过结果聚合和绘图 (缺少 pandas/matplotlib)。")

    analysis_end_time = time.time()
    print(f"\n====== 参数敏感性分析完成 ======")
    print(f"总耗时: {analysis_end_time - analysis_start_time:.2f} 秒")
    if failed_settings: print(f"注意: 有 {len(failed_settings)} 个参数设置的运行失败或遇到错误。")

# --- 入口点 (与 v4 相同) ---
if __name__ == "__main__":
    required_script = CONFIG_SENSITIVITY.get('experiment_runner_script', 'ExperimentRunner.py')
    if not FilePath(required_script).is_file():
         print(f"错误: 依赖的脚本 '{required_script}' 不存在。请确保它是 {required_script}-v24 或更高版本。")
         sys.exit(1)
    required_config = CONFIG_SENSITIVITY.get('base_config_path', 'base_experiment_config.json')
    if not FilePath(required_config).is_file():
         print(f"错误: 依赖的基础配置文件 '{required_config}' 不存在。请在运行前创建它。")
         sys.exit(1)

    run_sensitivity_analysis(CONFIG_SENSITIVITY)
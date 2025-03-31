# Analysis_Visualization.py-v9
"""
结果分析与可视化模块。
负责加载 ExperimentRunner 生成的实验结果 CSV 文件，进行数据预处理、
性能指标分析，并生成汇总表格和对比图表。

与论文的关联:
- 实验评估: 本脚本用于处理和可视化论文实验章节中描述的算法性能对比结果。
- 性能指标: 分析的关键指标 (TotalCost, TravelCost, TurnCost, WaitCost, CPUTime_s, Makespan)
           直接关联论文中用于评估算法效果的指标。其中成本指标源自论文的目标函数 (公式 1)。
- 算法对比: 生成的表格和图表用于对比论文核心算法 (ALNS) 与基准算法 (Benchmark) 的表现。
- 结果来源: 读取由 ExperimentRunner (v25+) 生成的 CSV 结果文件。

版本变更 (v8 -> v9):
- 添加了详细的模块和函数文档字符串，明确与论文实验评估框架的关联。
- 在关键分析和绘图部分添加了内联注释，解释指标和对比的意义。
- 更新 RESULTS_DIR 以匹配 ExperimentRunner v25。
- 清理了不再需要的注释。
- 确保代码风格一致性和完整性。
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path as FilePath
import re
from typing import Optional, List, Dict, Any, Tuple
import traceback
import csv

# --- 配置 ---
# !! 重要: 确保此目录指向最新 ExperimentRunner 的输出 !!
RESULTS_DIR = "experiment_results_v25"

# --- 文件查找函数 (保持 v8 逻辑) ---
def find_latest_csv(results_dir: str) -> Optional[FilePath]:
    """
    (内部辅助) 在指定目录及其子目录中查找最新的实验结果 CSV 文件。
    过滤掉可能存在的 summary_table.csv 文件。
    """
    dir_path = FilePath(results_dir)
    if not dir_path.is_dir():
        print(f"错误：结果目录 '{results_dir}' 不存在。")
        return None
    try:
        # 使用 rglob 递归查找所有 CSV 文件
        all_csv_files = list(dir_path.rglob('*.csv'))
        # 过滤掉名为 "summary_table.csv" 或类似摘要性质的文件
        experiment_csv_files = [
            f for f in all_csv_files
            if not f.name.lower().startswith('summary_table') and
               not f.name.lower().endswith('_cost_history.csv') and
               not f.name.lower().endswith('_operator_history.csv') and
               not f.name.lower().startswith('sensitivity_summary')
        ]

        if not experiment_csv_files:
            print(f"错误：在 '{results_dir}' 及其子目录中找不到任何符合条件的实验结果 CSV 文件。")
            print(f"      (已排除 summary_table*, *_history.csv, sensitivity_summary*)")
            return None
        # 按修改时间降序排序，获取最新的文件
        experiment_csv_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = experiment_csv_files[0]
        print(f"找到最新的实验结果文件: {latest_file}")
        return latest_file
    except Exception as e:
        print(f"错误: 搜索 CSV 文件时发生异常: {e}")
        traceback.print_exc()
        return None

# --- 数据加载与预处理 (保持 v8 逻辑，增强注释) ---
def load_and_preprocess_data(csv_filepath: FilePath) -> Optional[pd.DataFrame]:
    """
    加载指定的 CSV 文件，并进行数据预处理：
    - 替换无效值 (Inf, N/A, None, '') 为 NaN。
    - 转换数值列类型。
    - 转换布尔列类型。
    - 转换整数列类型。
    """
    if not csv_filepath.is_file():
        print(f"错误：文件 '{csv_filepath}' 不存在。")
        return None
    try:
        df = pd.read_csv(csv_filepath)
        print(f"成功加载数据，共 {len(df)} 行记录。")

        # 统一处理无效/缺失值
        df.replace(['Inf', 'N/A', 'None', '', float('inf')], np.nan, inplace=True)

        # 定义需要转换为数值类型的列 (包含成本、时间和部分参数)
        numeric_cols = [
            'TotalCost', 'TravelCost', 'TurnCost', 'WaitCost', 'CPUTime_s', 'Makespan',
            'CostWeightAlpha', 'CostWeightBeta', 'CostWeightGamma',
            'ALNS_MaxIter_Param', 'ALNS_NoImproveLimit_Param', 'ALNS_CoolingRate_Param', 'ALNS_InitialTemp_Param',
            'ALNS_RegretK_Param', 'ALNS_RemovalMax_Param', 'ALNS_RemovalMin_Param',
            'ALNS_RegretLimitAbs_Param', 'ALNS_TimeFactor_Param',
            'ObstacleRatio_Param', 'ExpansionRadius_Param'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # errors='coerce' 会将无法转换的值变为 NaN

        # 定义需要转换为布尔类型的列
        bool_cols = ['Success']
        for col in bool_cols:
            if col in df.columns:
                # 明确处理 'true'/'false' (不区分大小写) 及原生布尔值
                df[col] = df[col].apply(
                    lambda x: str(x).strip().lower() == 'true' if pd.notna(x) and isinstance(x, str) else (True if pd.notna(x) and x is True else False)
                )
                df[col] = df[col].astype('boolean') # 使用 Pandas 可空布尔类型
            else:
                print(f"警告: CSV 文件中缺少布尔列 '{col}'。")

        # 定义需要转换为整数类型的列 (使用可空整数类型 Int64)
        int_cols = [
            'Scenario', 'RunID', 'NumAGVs_Param', 'Makespan',
            'ALNS_MaxIter_Param', 'ALNS_NoImproveLimit_Param', 'ALNS_RegretK_Param',
            'ExpansionRadius_Param'
        ]
        for col in int_cols:
            if col in df.columns:
                # 先转为数值，再转为 Int64
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        print("数据预处理完成。")
        return df
    except FileNotFoundError:
        print(f"错误：文件 '{csv_filepath}' 未找到。")
        return None
    except Exception as e:
        print(f"错误：加载或预处理数据时出错: {e}")
        traceback.print_exc()
        return None

# --- 分析函数 (保持 v8 逻辑，增强注释) ---
def perform_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    对预处理后的数据进行分析。
    - 按场景 (Scenario 1, Scenario 2) 分组。
    - 过滤成功的运行记录 (Success=True)。
    - 计算场景 2 各算法性能指标的均值和标准差。
    """
    print("\n--- 开始数据分析 ---")
    analysis_results: Dict[str, Any] = {'Scenario1': {}, 'Scenario2': {}}

    # 检查 Success 列是否存在且有效
    if 'Success' not in df.columns or df['Success'].isnull().all() or not pd.api.types.is_bool_dtype(df['Success']):
        print("警告：缺少有效的 'Success' 列，无法按成功状态过滤。将尝试分析所有数据。")
        # 注意：如果不过滤，均值和标准差的意义可能会受失败案例 (Inf 成本) 影响，需要谨慎解释
        df_success = df.copy()
    else:
        # 只分析成功的运行记录
        df_success = df[df['Success'] == True].copy()

    if df_success.empty:
        print("警告：没有找到任何成功的运行记录，无法进行性能分析。")
        return analysis_results # 返回空结果

    # 定义要分析的核心性能指标 (关联论文评估)
    metrics_to_analyze = ['TotalCost', 'TravelCost', 'TurnCost', 'WaitCost', 'CPUTime_s', 'Makespan']

    # --- 分析场景 1 (通常只有一个运行实例) ---
    df_s1 = df_success[df_success['Scenario'] == 1]
    if not df_s1.empty:
        # 按算法分组，获取每个算法的指标值 (假设场景1每个算法只运行一次)
        # 使用 .first() 处理可能因重复运行产生的多行
        results_s1 = df_s1.groupby('Algorithm')[metrics_to_analyze].first().to_dict('index')
        analysis_results['Scenario1'] = results_s1
        print("场景 1 (固定场景) 结果:")
        print(pd.DataFrame(results_s1).T.round(2)) # 转置后打印更清晰
    else:
        print("未找到场景 1 的成功运行记录。")

    # --- 分析场景 2 (多个随机实例) ---
    df_s2 = df_success[df_success['Scenario'] == 2]
    if not df_s2.empty:
        grouped_s2 = df_s2.groupby('Algorithm') # 按算法分组
        # 定义聚合操作：计算均值、标准差，并统计成功运行次数
        agg_funcs = {metric: ['mean', 'std'] for metric in metrics_to_analyze}
        agg_funcs['RunID'] = ['count'] # 使用 RunID 或其他非空列计数
        try:
            results_s2_agg = grouped_s2.agg(agg_funcs)
            # 重命名列名以便访问 (例如 'TotalCost_mean')
            results_s2_agg.columns = ['_'.join(col).strip('_') for col in results_s2_agg.columns.values]
            results_s2_agg.rename(columns={'RunID_count': 'SuccessfulRuns'}, inplace=True)
            analysis_results['Scenario2'] = results_s2_agg.to_dict('index') # 转换为字典存储
            print("\n场景 2 (随机场景) 统计结果 (均值/标准差):")
            print(results_s2_agg.round(2))
        except Exception as agg_e:
            print(f"错误：在聚合场景 2 数据时发生错误: {agg_e}")
            traceback.print_exc()
    else:
        print("未找到场景 2 的成功运行记录。")

    print("--- 数据分析完成 ---")
    return analysis_results

# --- 表格生成函数 (保持 v7 逻辑，增强注释) ---
def generate_summary_table(analysis_data: Dict[str, Any], md_filename: str = "summary_table.md", csv_filename: str = "summary_table.csv"):
    """
    根据分析结果生成 Markdown 和 CSV 格式的汇总表格。
    表格清晰地展示了各算法在不同场景下的核心性能指标。
    """
    print("\n--- 生成汇总表格 ---")
    s1_results = analysis_data.get('Scenario1', {})
    s2_results_agg = analysis_data.get('Scenario2', {})

    if not s1_results and not s2_results_agg:
        print("错误：无分析数据可生成表格。")
        return

    # 获取所有参与比较的算法名称
    algorithms = sorted(list(set(list(s1_results.keys()) + list(s2_results_agg.keys()))))
    # 定义表格中要包含的指标列
    metrics = ['TotalCost', 'TravelCost', 'TurnCost', 'WaitCost', 'CPUTime_s', 'Makespan']
    table_data = [] # 用于存储表格行的列表

    # 遍历每个算法，为其生成场景 1 和场景 2 的数据行
    for alg in algorithms:
        # 提取场景 1 数据
        successful_runs_s1 = 1 if alg in s1_results and not pd.DataFrame(s1_results[alg], index=[0]).isnull().all().all() else 0
        row_s1 = {'Algorithm': alg, 'Scenario': 'Scenario 1', 'SuccessfulRuns': successful_runs_s1}
        for metric in metrics:
            val_s1 = s1_results.get(alg, {}).get(metric, np.nan)
            row_s1[metric] = f"{val_s1:.2f}" if pd.notna(val_s1) else "N/A"
        table_data.append(row_s1)

        # 提取场景 2 数据 (均值 ± 标准差)
        successful_runs_s2 = s2_results_agg.get(alg, {}).get('SuccessfulRuns', 0)
        row_s2 = {'Algorithm': alg, 'Scenario': 'Scenario 2 (Avg ± Std)', 'SuccessfulRuns': successful_runs_s2}
        for metric in metrics:
            mean_s2 = s2_results_agg.get(alg, {}).get(f'{metric}_mean', np.nan)
            std_s2 = s2_results_agg.get(alg, {}).get(f'{metric}_std', np.nan)
            # 格式化输出，仅在标准差有效时显示
            if pd.notna(mean_s2):
                 if pd.notna(std_s2) and std_s2 > 1e-6: # 标准差大于一个很小的值才显示
                      row_s2[metric] = f"{mean_s2:.2f} ± {std_s2:.2f}"
                 else:
                      row_s2[metric] = f"{mean_s2:.2f}" # 仅显示均值
            else:
                 row_s2[metric] = "N/A" # 无有效均值
        table_data.append(row_s2)

    # 创建 Pandas DataFrame 并设置列顺序
    df_table = pd.DataFrame(table_data)
    column_order = ['Algorithm', 'Scenario', 'SuccessfulRuns'] + metrics
    df_table = df_table[column_order]

    # --- 保存为 Markdown ---
    try:
        # 需要安装 tabulate: pip install tabulate
        md_table = df_table.to_markdown(index=False)
        results_path = FilePath(RESULTS_DIR)
        results_path.mkdir(parents=True, exist_ok=True)
        table_filepath_md = results_path / md_filename
        with open(table_filepath_md, 'w', encoding='utf-8') as f:
            f.write(md_table)
        print(f"Markdown 汇总表格已保存到: {table_filepath_md}")
        print("\n汇总结果 (Markdown):")
        print(md_table)
    except ImportError:
        print("警告: 未安装 'tabulate' 库，无法生成 Markdown 表格。请运行 'pip install tabulate'。")
    except Exception as e:
        print(f"错误: 保存 Markdown 表格失败: {e}")

    # --- 保存为 CSV ---
    try:
        results_path = FilePath(RESULTS_DIR)
        results_path.mkdir(parents=True, exist_ok=True)
        table_filepath_csv = results_path / csv_filename
        # 直接使用包含 '±' 符号的 DataFrame 保存
        df_table.to_csv(table_filepath_csv, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 提高 Excel 兼容性
        print(f"CSV 汇总表格已保存到: {table_filepath_csv}")
    except Exception as e:
        print(f"错误: 保存 CSV 表格失败: {e}")

# --- 绘图函数 (保持 v7 逻辑，增强注释) ---
def generate_plots(analysis_data: Dict[str, Any]):
    """
    根据分析结果生成条形图，对比不同算法在关键性能指标上的表现。
    包括主要指标（总成本、时间、Makespan）和成本构成指标。
    """
    print("\n--- 开始生成图表 ---")
    s1_results = analysis_data.get('Scenario1', {})
    s2_results_agg = analysis_data.get('Scenario2', {})

    if not s1_results and not s2_results_agg:
        print("错误：无分析数据用于绘图。")
        return

    # 获取算法和场景列表
    algorithms = sorted(list(set(list(s1_results.keys()) + list(s2_results_agg.keys()))))
    scenarios = ['Scenario 1', 'Scenario 2 (Avg)']

    # 定义要绘制的指标分组 (与论文评估重点关联)
    primary_metrics = ['TotalCost', 'CPUTime_s', 'Makespan'] # 主要性能指标
    cost_breakdown_metrics = ['TravelCost', 'TurnCost', 'WaitCost'] # 成本构成 (对应公式 1 分量)
    metrics_groups = {'Primary': primary_metrics, 'CostBreakdown': cost_breakdown_metrics}

    plot_dir = FilePath(RESULTS_DIR) / "_plots" # 将图表保存在子目录中
    try:
         plot_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         print(f"警告: 无法创建绘图目录 '{plot_dir}': {e}")
         plot_dir = FilePath(RESULTS_DIR) # 回退到主结果目录

    # 对每个指标组生成一张图
    for group_name, metrics in metrics_groups.items():
        # 准备绘图数据
        plot_data: Dict[str, pd.DataFrame] = {metric: pd.DataFrame(index=algorithms, columns=scenarios) for metric in metrics}
        plot_error: Dict[str, pd.DataFrame] = {metric: pd.DataFrame(index=algorithms, columns=scenarios) for metric in metrics}

        # 填充数据和误差 (标准差)
        for alg in algorithms:
            for metric in metrics:
                if alg in s1_results:
                    plot_data[metric].loc[alg, 'Scenario 1'] = s1_results[alg].get(metric, np.nan)
                    plot_error[metric].loc[alg, 'Scenario 1'] = 0 # 场景1 无误差棒
                if alg in s2_results_agg:
                    plot_data[metric].loc[alg, 'Scenario 2 (Avg)'] = s2_results_agg[alg].get(f'{metric}_mean', np.nan)
                    plot_error[metric].loc[alg, 'Scenario 2 (Avg)'] = s2_results_agg[alg].get(f'{metric}_std', 0) # 获取标准差作为误差

        # 过滤掉完全没有数据的指标
        valid_metrics = [m for m in metrics if not plot_data[m].isnull().all().all()]
        if not valid_metrics:
            print(f"警告：组 '{group_name}' 没有有效的指标数据用于绘图。")
            continue

        num_metrics = len(valid_metrics)
        # 创建子图布局 (一行多列)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5.5 * num_metrics + 1, 6), sharey=False)
        if num_metrics == 1: axes = [axes] # 处理只有一个子图的情况

        bar_width = 0.35 # 条形宽度
        x = np.arange(len(algorithms)) # 算法在 x 轴的位置

        # 绘制每个指标的条形图
        for i, metric in enumerate(valid_metrics):
            ax = axes[i]
            # 获取场景 1 和场景 2 的均值及场景 2 的误差
            means_s1 = plot_data[metric]['Scenario 1'].astype(float).fillna(0).values
            means_s2 = plot_data[metric]['Scenario 2 (Avg)'].astype(float).fillna(0).values
            errors_s2 = plot_error[metric]['Scenario 2 (Avg)'].astype(float).fillna(0).values
            errors_s2 = np.maximum(errors_s2, 0) # 确保误差非负

            # 绘制条形图 (并排显示场景 1 和 场景 2)
            rects1 = ax.bar(x - bar_width/2, means_s1, bar_width, label='Scenario 1', color='skyblue', zorder=3)
            rects2 = ax.bar(x + bar_width/2, means_s2, bar_width, label='Scenario 2 (Avg ± Std)', color='lightcoral', yerr=errors_s2, capsize=5, zorder=3, error_kw={'elinewidth': 1, 'capthick': 1})

            # 设置坐标轴标签、标题、刻度
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(algorithms, rotation=0) # 设置算法名称作为 x 轴标签
            ax.legend() # 显示图例
            ax.grid(True, axis='y', linestyle='--', alpha=0.7) # 添加水平网格线

            # 添加数值标签到条形图顶部
            ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=9)
            ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=9)

            # 动态调整 Y 轴范围以优化显示效果
            valid_values_s1 = plot_data[metric]['Scenario 1'].astype(float).dropna().values
            valid_values_s2 = plot_data[metric]['Scenario 2 (Avg)'].astype(float).dropna().values
            if len(valid_values_s1) > 0 or len(valid_values_s2) > 0:
                 all_valid_values = np.concatenate([v for v in [valid_values_s1, valid_values_s2] if len(v)>0])
                 min_val = all_valid_values.min() if len(all_valid_values) > 0 else 0
                 max_val = all_valid_values.max() if len(all_valid_values) > 0 else 0
                 max_val_with_error = max_val
                 # 考虑误差棒的上限
                 if len(valid_values_s2) > 0:
                       means_s2_valid = means_s2[~np.isnan(plot_data[metric]['Scenario 2 (Avg)'].astype(float))]
                       errors_s2_valid = errors_s2[~np.isnan(plot_data[metric]['Scenario 2 (Avg)'].astype(float))]
                       if len(means_s2_valid) > 0:
                           max_val_with_error = max(max_val, (means_s2_valid + errors_s2_valid).max())

                 y_margin = max(1.0, max_val_with_error * 0.15) # 顶部留白
                 y_min_lim = max(0, min_val - y_margin * 0.2) # 底部稍微留白，但不低于0
                 y_max_lim = max_val_with_error + y_margin

                 # 处理特殊情况 (所有值为0或非常接近)
                 if max_val_with_error < 1e-6: y_min_lim = -0.1; y_max_lim = 0.1
                 elif abs(max_val_with_error - min_val) < 1e-6: # 值非常接近
                      y_min_lim = max(0, min_val - 0.5); y_max_lim = max_val_with_error + 0.5

                 ax.set_ylim(y_min_lim, y_max_lim)
            else:
                 ax.set_ylim(0, 1) # 如果没有有效数据，设置默认范围

        # 设置整张图的标题和布局
        fig.suptitle(f'{group_name} Metrics Comparison', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，防止标题重叠

        # 保存图表
        plot_filename = plot_dir / f"{group_name}_comparison_plot.png"
        try:
            plt.savefig(plot_filename, dpi=150)
            print(f"图表 '{group_name}' 已保存到: {plot_filename}")
        except Exception as e:
            print(f"错误: 保存图表 '{plot_filename}' 失败: {e}")
        plt.close(fig) # 关闭图形，释放内存

    print("--- 图表生成完成 ---")

# --- 主执行块 (保持 v8 逻辑) ---
if __name__ == "__main__":
    print("--- 开始结果分析与可视化 (v9 - Enhanced Docs & Thesis Links) ---")
    # 1. 查找最新的实验结果 CSV 文件
    latest_csv_file = find_latest_csv(RESULTS_DIR)

    if latest_csv_file:
        # 2. 加载并预处理数据
        df_results = load_and_preprocess_data(latest_csv_file)
        if df_results is not None:
            # 3. 执行数据分析
            analysis_results = perform_analysis(df_results)
            if analysis_results:
                # 4. 生成汇总表格
                generate_summary_table(analysis_results)
                # 5. 生成对比图表
                generate_plots(analysis_results)
            else:
                print("分析未产生有效结果，无法生成表格和图表。")
        else:
            print("数据加载或预处理失败，无法进行分析。")
    else:
        print("未找到有效的实验结果文件，无法进行分析。")

    print("\n--- 分析与可视化流程结束 ---")
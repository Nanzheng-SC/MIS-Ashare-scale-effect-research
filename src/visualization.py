import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 配置日志，使用绝对路径
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'visualization.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger('visualization')

# 数据目录和结果目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_group_data_from_files(data_dir=None):
    """
    从data文件夹加载分组数据文件
    
    Args:
        data_dir: 数据文件夹路径
    
    Returns:
        DataFrame: 合并后的分组数据
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    logger.info(f"开始从文件夹加载分组数据: {data_dir}")
    
    all_group_data = []
    group_file_pattern = r'group_\d+_data\.csv'
    
    # 查找分组数据文件
    import re
    csv_files = [f for f in os.listdir(data_dir) if re.match(group_file_pattern, f)]
    
    if not csv_files:
        logger.error(f"未找到分组数据文件，请先运行data_fetch.py生成分组数据")
        return None
    
    logger.info(f"找到 {len(csv_files)} 个分组数据文件")
    
    for file in csv_files:
        try:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            
            # 检查必要的列是否存在
            required_columns = ['ts_code', 'trade_date', 'close', 'market_cap', 'monthly_return', 'market_cap_group']
            if all(col in df.columns for col in required_columns):
                # 使用已有的monthly_return作为return列
                if 'return' not in df.columns:
                    df['return'] = df['monthly_return']
                all_group_data.append(df)
                logger.info(f"成功加载分组数据文件: {file}, 记录数: {len(df)}")
            else:
                logger.warning(f"分组文件 {file} 缺少必要的列，跳过")
                
        except Exception as e:
            logger.error(f"加载分组文件 {file} 时出错: {str(e)}")
    
    if not all_group_data:
        logger.error("没有成功加载任何分组数据文件")
        return None
    
    # 合并所有分组数据
    merged_data = pd.concat(all_group_data, ignore_index=True)
    logger.info(f"分组数据加载完成，总记录数: {len(merged_data)}, 股票数量: {merged_data['ts_code'].nunique()}")
    
    # 转换日期格式
    if 'trade_date' in merged_data.columns and not pd.api.types.is_datetime64_any_dtype(merged_data['trade_date']):
        try:
            merged_data['trade_date'] = pd.to_datetime(merged_data['trade_date'], format='%Y%m%d')
        except:
            merged_data['trade_date'] = pd.to_datetime(merged_data['trade_date'])
    
    # 添加分组名称映射
    group_name_map = {1: '小市值组', 2: '次小市值组', 3: '中等市值组', 4: '次大市值组', 5: '大市值组'}
    merged_data['group_name'] = merged_data['market_cap_group'].map(group_name_map)
    
    return merged_data

def analyze_scale_effect(group_data):
    """
    分析A股市场规模效应（基于分组数据）
    
    Args:
        group_data: 分组股票数据
    
    Returns:
        dict: 分析结果
    """
    logger.info("开始基于分组数据的规模效应分析")
    
    # 计算各分组的统计指标
    results = {}
    months_per_year = 12  # 按月度数据计算
    risk_free_rate = 0.02  # 假设无风险利率为2%
    
    # 计算各组的年化收益率、波动率、夏普比率
    group_stats = []
    group_order = ['小市值组', '次小市值组', '中等市值组', '次大市值组', '大市值组']
    
    for group in group_order:
        group_data_subset = group_data[group_data['group_name'] == group]
        
        if len(group_data_subset) > 0:
            # 计算平均月度收益率并年化
            avg_monthly_return = group_data_subset['monthly_return'].mean()
            annual_return = ((1 + avg_monthly_return) ** months_per_year) - 1
            
            # 计算月度波动率并年化
            monthly_volatility = group_data_subset['monthly_return'].std()
            annual_volatility = monthly_volatility * np.sqrt(months_per_year)
            
            # 计算夏普比率
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            
            # 计算股票数量
            stock_count = group_data_subset['ts_code'].nunique()
            
            # 计算平均市值（亿元）
            avg_market_cap = group_data_subset['market_cap'].mean()
            
            group_stats.append({
                'group_name': group,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'stock_count': stock_count,
                'avg_market_cap': avg_market_cap
            })
    
    results['group_stats'] = pd.DataFrame(group_stats)
    
    # 计算分组的每日平均收益率（用于累计收益图）
    daily_returns = group_data.pivot_table(
        index='trade_date', 
        columns='group_name', 
        values='monthly_return', 
        aggfunc='mean'
    )
    results['daily_returns'] = daily_returns
    
    logger.info("规模效应分析完成")
    return results

def create_size_distribution_chart(group_stats):
    """
    创建市值分布图表
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='group_name', y='stock_count', data=group_stats, palette='viridis')
        plt.title('不同市值组的股票数量分布', fontsize=14)
        plt.xlabel('市值组', fontsize=12)
        plt.ylabel('股票数量', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(group_stats['stock_count']):
            plt.text(i, v + 0.5, str(v), ha='center', fontsize=10)
        
        output_path = os.path.join(RESULTS_DIR, '市值分布.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"市值分布图已保存: {output_path}")
        return True
    except Exception as e:
        logger.error(f"创建市值分布图失败: {str(e)}")
        return False

def create_return_comparison_chart(group_stats):
    """
    创建收益率对比图表
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # 设置颜色：正收益为绿色，负收益为红色
        colors = ['green' if x > 0 else 'red' for x in group_stats['annual_return']]
        
        sns.barplot(x='group_name', y='annual_return', data=group_stats, palette=colors)
        plt.title('不同市值组的年化收益率对比', fontsize=14)
        plt.xlabel('市值组', fontsize=12)
        plt.ylabel('年化收益率', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(group_stats['annual_return']):
            plt.text(i, v + (0.01 if v > 0 else -0.03), f'{v:.2%}', ha='center', fontsize=10)
        
        # 添加零线
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        output_path = os.path.join(RESULTS_DIR, '年化收益率对比.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"收益率对比图已保存: {output_path}")
        return True
    except Exception as e:
        logger.error(f"创建收益率对比图失败: {str(e)}")
        return False

def create_volatility_comparison_chart(group_stats):
    """
    创建波动率对比图表
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='group_name', y='annual_volatility', data=group_stats, palette='Blues')
        plt.title('不同市值组的年化波动率对比', fontsize=14)
        plt.xlabel('市值组', fontsize=12)
        plt.ylabel('年化波动率', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(group_stats['annual_volatility']):
            plt.text(i, v + 0.01, f'{v:.2%}', ha='center', fontsize=10)
        
        output_path = os.path.join(RESULTS_DIR, '波动率对比.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"波动率对比图已保存: {output_path}")
        return True
    except Exception as e:
        logger.error(f"创建波动率对比图失败: {str(e)}")
        return False

def create_sharpe_ratio_chart(group_stats):
    """
    创建夏普比率图表
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # 设置颜色：夏普比率>0为绿色，否则为红色
        colors = ['green' if x > 0 else 'red' for x in group_stats['sharpe_ratio']]
        
        sns.barplot(x='group_name', y='sharpe_ratio', data=group_stats, palette=colors)
        plt.title('不同市值组的夏普比率对比', fontsize=14)
        plt.xlabel('市值组', fontsize=12)
        plt.ylabel('夏普比率', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(group_stats['sharpe_ratio']):
            plt.text(i, v + (0.1 if v > 0 else -0.2), f'{v:.2f}', ha='center', fontsize=10)
        
        # 添加零线
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        output_path = os.path.join(RESULTS_DIR, '夏普比率对比.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"夏普比率图已保存: {output_path}")
        return True
    except Exception as e:
        logger.error(f"创建夏普比率图失败: {str(e)}")
        return False

def create_cumulative_return_chart(daily_returns):
    """
    创建累计收益率图表
    """
    try:
        # 计算累计收益率
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        plt.figure(figsize=(12, 6))
        
        for column in cumulative_returns.columns:
            plt.plot(cumulative_returns.index, cumulative_returns[column], label=column, linewidth=2)
        
        plt.title('不同市值组的累计收益率曲线', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('累计收益率', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # 格式化x轴日期标签
        plt.xticks(rotation=45)
        
        output_path = os.path.join(RESULTS_DIR, '累计收益率曲线.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"累计收益率曲线图已保存: {output_path}")
        return True
    except Exception as e:
        logger.error(f"创建累计收益率曲线图失败: {str(e)}")
        return False

def create_time_series_chart(data, value_column, title, ylabel, output_filename, is_percentage=False):
    """
    创建时间序列折线图的通用函数
    
    Args:
        data: 时间序列数据
        value_column: 要绘制的列名
        title: 图表标题
        ylabel: Y轴标签
        output_filename: 输出文件名
        is_percentage: 是否按百分比显示
    
    Returns:
        bool: 是否成功创建
    """
    try:
        logger.info(f"创建时间序列图表: {title}")
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 为每个组别绘制折线
        for group in data.columns:
            plt.plot(data.index, data[group], marker='o', markersize=3, linewidth=2, label=group)
        
        # 设置图表属性
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='市值组别', fontsize=10)
        plt.xticks(rotation=45)
        
        # 如果是百分比格式，设置Y轴格式
        if is_percentage:
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(RESULTS_DIR, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"时间序列图表已保存到: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"创建时间序列图表 {title} 时出错: {str(e)}")
        return False

def create_monthly_return_time_series(group_data):
    """
    创建各分组月度收益率随时间变化的折线图
    
    Args:
        group_data: 分组数据
    
    Returns:
        bool: 是否成功创建
    """
    try:
        logger.info("创建月度收益率时间序列图表")
        
        # 准备月度收益率数据
        monthly_returns = group_data.pivot_table(
            index='trade_date', 
            columns='group_name', 
            values='monthly_return', 
            aggfunc='mean'
        )
        
        # 按时间排序
        monthly_returns = monthly_returns.sort_index()
        
        # 调用通用函数创建图表
        return create_time_series_chart(
            monthly_returns, 
            None,  # 所有列都会被绘制
            '各分组月度收益率时间序列', 
            '月度收益率', 
            '月度收益率时间序列.png',
            is_percentage=True
        )
        
    except Exception as e:
        logger.error(f"创建月度收益率时间序列图表时出错: {str(e)}")
        return False

def create_annualized_return_time_series(group_data, rolling_window=12):
    """
    创建各分组滚动年化收益率随时间变化的折线图
    
    Args:
        group_data: 分组数据
        rolling_window: 滚动窗口大小（月）
    
    Returns:
        bool: 是否成功创建
    """
    try:
        logger.info("创建滚动年化收益率时间序列图表")
        
        # 准备月度收益率数据
        monthly_returns = group_data.pivot_table(
            index='trade_date', 
            columns='group_name', 
            values='monthly_return', 
            aggfunc='mean'
        )
        
        # 按时间排序
        monthly_returns = monthly_returns.sort_index()
        
        # 计算滚动年化收益率
        rolling_annual_return = ((1 + monthly_returns).rolling(window=rolling_window).apply(lambda x: np.prod(1+x)) - 1)
        
        # 调用通用函数创建图表
        return create_time_series_chart(
            rolling_annual_return, 
            None,  # 所有列都会被绘制
            f'各分组滚动{rolling_window}个月年化收益率时间序列', 
            '年化收益率', 
            '滚动年化收益率时间序列.png',
            is_percentage=True
        )
        
    except Exception as e:
        logger.error(f"创建滚动年化收益率时间序列图表时出错: {str(e)}")
        return False

def create_volatility_time_series(group_data, rolling_window=12):
    """
    创建各分组滚动波动率随时间变化的折线图
    
    Args:
        group_data: 分组数据
        rolling_window: 滚动窗口大小（月）
    
    Returns:
        bool: 是否成功创建
    """
    try:
        logger.info("创建滚动波动率时间序列图表")
        
        # 准备月度收益率数据
        monthly_returns = group_data.pivot_table(
            index='trade_date', 
            columns='group_name', 
            values='monthly_return', 
            aggfunc='mean'
        )
        
        # 按时间排序
        monthly_returns = monthly_returns.sort_index()
        
        # 计算滚动波动率（年化）
        rolling_monthly_volatility = monthly_returns.rolling(window=rolling_window).std()
        rolling_annual_volatility = rolling_monthly_volatility * np.sqrt(12)  # 年化
        
        # 调用通用函数创建图表
        return create_time_series_chart(
            rolling_annual_volatility, 
            None,  # 所有列都会被绘制
            f'各分组滚动{rolling_window}个月波动率时间序列', 
            '年化波动率', 
            '滚动波动率时间序列.png',
            is_percentage=True
        )
        
    except Exception as e:
        logger.error(f"创建滚动波动率时间序列图表时出错: {str(e)}")
        return False

def create_sharpe_ratio_time_series(group_data, rolling_window=12, risk_free_rate=0.02):
    """
    创建各分组滚动夏普比率随时间变化的折线图
    
    Args:
        group_data: 分组数据
        rolling_window: 滚动窗口大小（月）
        risk_free_rate: 无风险利率
    
    Returns:
        bool: 是否成功创建
    """
    try:
        logger.info("创建滚动夏普比率时间序列图表")
        
        # 准备月度收益率数据
        monthly_returns = group_data.pivot_table(
            index='trade_date', 
            columns='group_name', 
            values='monthly_return', 
            aggfunc='mean'
        )
        
        # 按时间排序
        monthly_returns = monthly_returns.sort_index()
        
        # 计算滚动年化收益率
        rolling_annual_return = ((1 + monthly_returns).rolling(window=rolling_window).apply(lambda x: np.prod(1+x)) - 1)
        
        # 计算滚动波动率（年化）
        rolling_monthly_volatility = monthly_returns.rolling(window=rolling_window).std()
        rolling_annual_volatility = rolling_monthly_volatility * np.sqrt(12)  # 年化
        
        # 计算滚动夏普比率
        rolling_sharpe = (rolling_annual_return - risk_free_rate) / rolling_annual_volatility
        
        # 调用通用函数创建图表
        return create_time_series_chart(
            rolling_sharpe, 
            None,  # 所有列都会被绘制
            f'各分组滚动{rolling_window}个月夏普比率时间序列', 
            '夏普比率', 
            '滚动夏普比率时间序列.png',
            is_percentage=False
        )
        
    except Exception as e:
        logger.error(f"创建滚动夏普比率时间序列图表时出错: {str(e)}")
        return False

def create_comprehensive_analysis_chart(analysis_results):
    """
    创建规模效应综合分析图表
    
    Args:
        analysis_results: 分析结果
    
    Returns:
        bool: 是否成功创建
    """
    try:
        logger.info("创建规模效应综合分析图表")
        
        group_stats = analysis_results['group_stats']
        
        # 创建图表
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 年化收益率对比
        ax1 = plt.subplot(2, 2, 1)
        bars1 = ax1.bar(group_stats['group_name'], group_stats['annual_return'], color='skyblue')
        ax1.set_title('各分组年化收益率对比')
        ax1.set_ylabel('年化收益率')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{height:.2%}', 
                    ha='center', va='bottom')
        
        # 2. 年化波动率对比
        ax2 = plt.subplot(2, 2, 2)
        bars2 = ax2.bar(group_stats['group_name'], group_stats['annual_volatility'], color='lightcoral')
        ax2.set_title('各分组年化波动率对比')
        ax2.set_ylabel('年化波动率')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 在柱状图上添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{height:.2%}', 
                    ha='center', va='bottom')
        
        # 3. 夏普比率对比
        ax3 = plt.subplot(2, 2, 3)
        bars3 = ax3.bar(group_stats['group_name'], group_stats['sharpe_ratio'], color='lightgreen')
        ax3.set_title('各分组夏普比率对比')
        ax3.set_ylabel('夏普比率')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 在柱状图上添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{height:.2f}', 
                    ha='center', va='bottom')
        
        # 4. 股票数量对比
        ax4 = plt.subplot(2, 2, 4)
        bars4 = ax4.bar(group_stats['group_name'], group_stats['stock_count'], color='plum')
        ax4.set_title('各分组股票数量对比')
        ax4.set_ylabel('股票数量')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        # 在柱状图上添加数值标签
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{int(height)}', 
                    ha='center', va='bottom')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(RESULTS_DIR, '规模效应综合分析.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"规模效应综合分析图表已保存到: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"创建规模效应综合分析图表时出错: {str(e)}")
        return False

def run_visualization_pipeline():
    """
    运行完整的可视化分析流程（基于分组数据）
    
    Returns:
        bool: 是否成功完成
    """
    try:
        # 1. 加载分组数据
        group_data = load_group_data_from_files()
        if group_data is None:
            return False, None
        
        # 2. 分析规模效应
        analysis_results = analyze_scale_effect(group_data)
        
        # 3. 准备月度收益率数据用于累计收益图
        monthly_returns = group_data.pivot_table(
            index='trade_date', 
            columns='group_name', 
            values='monthly_return', 
            aggfunc='mean'
        )
        analysis_results['monthly_returns'] = monthly_returns
        
        # 4. 创建各个图表
        charts = [
            create_size_distribution_chart(analysis_results['group_stats']),
            create_return_comparison_chart(analysis_results['group_stats']),
            create_volatility_comparison_chart(analysis_results['group_stats']),
            create_sharpe_ratio_chart(analysis_results['group_stats']),
            create_cumulative_return_chart(analysis_results['monthly_returns']),
            create_comprehensive_analysis_chart(analysis_results),
            create_monthly_return_time_series(group_data),
            create_annualized_return_time_series(group_data, rolling_window=rolling_window),
    create_volatility_time_series(group_data, rolling_window=rolling_window),
    create_sharpe_ratio_time_series(group_data, rolling_window=rolling_window, risk_free_rate=0.02)
        ]
        
        # 检查是否所有图表都成功创建
        success_count = sum(charts)
        total_count = len(charts)
        
        logger.info(f"分组数据可视化完成，成功创建 {success_count}/{total_count} 个图表")
        
        # 返回规模效应分析结果
        return True, analysis_results['group_stats']
    except Exception as e:
        logger.error(f"运行可视化流程时出错: {str(e)}", exc_info=True)
        return False, None

def main():
    """
    分组数据可视化分析主函数
    """
    print("="*80)
    print("         A股市场规模效应分组可视化工具         ")
    print("="*80)
    print("本工具用于分析A股市场规模效应并生成可视化图表")
    print("请确保已经运行data_fetch.py生成分组数据")
    print("仅对分组数据进行分析")
    print("="*80)
    
    try:
        # 检查分组数据文件是否存在
        import re
        group_file_pattern = r'group_\d+_data\.csv'
        group_files = [f for f in os.listdir(DATA_DIR) if re.match(group_file_pattern, f)]
        
        if not group_files:
            print("\n✗ 错误：未找到分组数据文件")
            print("  请先运行: python src/data_fetch.py 生成分组数据")
            return False
        
        print(f"\n1. 分组数据文件检查完成")
        print(f"✓ 找到数据文件夹: {DATA_DIR}")
        print(f"✓ 找到 {len(group_files)} 个分组数据文件")
        
        # 运行可视化流程
        print(f"\n2. 开始运行规模效应可视化分析...")
        success, group_stats = run_visualization_pipeline()
        
        if success and group_stats is not None:
            print(f"\n✓ 可视化分析完成！")
            print(f"✓ 所有图表已保存到: {RESULTS_DIR}")
            
            # 显示分析结果摘要
            print(f"\n{'-'*80}")
            print("规模效应分析结果摘要:")
            print("{:<10} {:<10} {:<12} {:<12} {:<10}".format(
                "市值组", "股票数量", "年化收益率", "年化波动率", "夏普比率"
            ))
            print("-"*80)
            
            for _, row in group_stats.iterrows():
                print("{:<12} {:<10} {:<12.2%} {:<12.2%} {:<10.2f} {:<12.2f}".format(
                    row['group_name'],
                    int(row['stock_count']),
                    row['annual_return'],
                    row['annual_volatility'],
                    row['sharpe_ratio'],
                    row['avg_market_cap']
                ))
            print("-"*80)
            print("{:<12} {:<10} {:<12} {:<12} {:<10} {:<12}".format(
                "分组", "股票数量", "年化收益率", "年化波动率", "夏普比率", "平均市值(亿元)"
            ))
            print("-"*80)
            
            # 计算规模效应溢价
            small_cap_return = group_stats[group_stats['group_name'] == '小市值组']['annual_return'].iloc[0]
            large_cap_return = group_stats[group_stats['group_name'] == '大市值组']['annual_return'].iloc[0]
            scale_effect_premium = small_cap_return - large_cap_return
            
            print(f"\n规模效应溢价（小市值组 - 大市值组）: {scale_effect_premium:.2%}")
            
            if scale_effect_premium > 0:
                print("结论: 存在正规模效应（小市值股票表现优于大市值股票）")
            else:
                print("结论: 不存在正规模效应（小市值股票表现不如大市值股票）")
                
        else:
            print(f"\n✗ 可视化分析失败，请检查日志文件")
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        return False
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        print(f"\n✗ 程序运行出错: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    print("\n" + "="*80)
    if success:
        print("分组数据可视化分析完成！请检查results文件夹中的图表")
        print("图表列表:")
        print("1. 市值分布.png")
        print("2. 年化收益率对比.png")
        print("3. 波动率对比.png")
        print("4. 夏普比率对比.png")
        print("5. 累计收益率曲线.png")
        print("6. 规模效应综合分析.png")
        print("7. 月度收益率时间序列.png")
        print("8. 滚动年化收益率时间序列.png")
        print("9. 滚动波动率时间序列.png")
        print("10. 滚动夏普比率时间序列.png")
        print("\n注意：分析仅基于已分组的数据，确保先运行data_fetch.py生成分组数据")
    else:
        print("可视化分析失败，请检查错误信息")
    print("="*80)

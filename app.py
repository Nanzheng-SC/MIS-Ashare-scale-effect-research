import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import sys
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    logger.warning(f"设置中文字体失败: {str(e)}")

# 确保应用在Streamlit Cloud上正常运行的路径设置
# 获取当前文件目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
# 确保数据目录存在
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logger.warning(f"创建数据目录: {DATA_DIR}")

# 检查环境（开发环境或生产环境）
IS_LOCAL = os.getenv('STREAMLIT_LOCAL', 'true').lower() == 'true'
logger.info(f"应用运行环境: {'本地开发环境' if IS_LOCAL else 'Streamlit Cloud生产环境'}")

# Streamlit页面配置
st.set_page_config(
    page_title="A股市场规模效应研究",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 函数定义区域
def load_group_data():
    """
    加载所有分组数据
    
    Returns:
        tuple: (分组数据, 分组信息)
    """
    try:
        logger.info("开始加载分组数据")
        
        # 分组映射信息
        group_info = {
            1: {"name": "小市值组", "avg_cap": 20.00},
            2: {"name": "次小市值组", "avg_cap": 57.50},
            3: {"name": "中等市值组", "avg_cap": 180.00},
            4: {"name": "次大市值组", "avg_cap": 380.00},
            5: {"name": "大市值组", "avg_cap": 850.00}
        }
        
        all_data = []
        # 设置最大允许日期为2025-12-31
        max_allowed_date = pd.Timestamp('2025-12-31')
        
        # 加载每个分组的数据
        for group_id in range(1, 6):
            # 构建文件路径 - 使用更健壮的路径管理
            group_file = os.path.join(DATA_DIR, f'group_{group_id}_data.csv')
            
            logger.info(f"尝试加载分组 {group_id} 数据: {group_file}")
            
            # 检查文件是否存在
            if not os.path.exists(group_file):
                logger.error(f"分组 {group_id} 数据文件不存在: {group_file}")
                # 创建空的DataFrame结构以避免后续处理错误
                empty_df = pd.DataFrame({
                    'trade_date': pd.Series([], dtype='datetime64[ns]'),
                    'monthly_return': pd.Series([], dtype='float64'),
                    'group_id': [group_id],
                    'group_name': [group_info[group_id]['name']],
                    'avg_market_cap': [group_info[group_id]['avg_cap']]
                })
                all_data.append(empty_df)
                continue
            
            try:
                # 使用低内存模式加载数据
                df = pd.read_csv(group_file, low_memory=False)
                logger.info(f"成功读取分组 {group_id} 数据文件，原始数据形状: {df.shape}")
                
                # 添加分组信息
                df['group_id'] = group_id
                df['group_name'] = group_info[group_id]['name']
                df['avg_market_cap'] = group_info[group_id]['avg_cap']
                
                # 确保必要的列存在
                required_columns = ['monthly_return']
                for col in required_columns:
                    if col not in df.columns:
                        logger.warning(f"分组 {group_id} 数据中缺少必要列: {col}")
                        df[col] = 0.0
                
                # 确保日期列格式正确并验证日期范围
                if 'trade_date' in df.columns:
                    # 增强日期转换逻辑
                    if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                        # 首先检查是否为整数格式（YYYYMMDD）
                        if pd.api.types.is_integer_dtype(df['trade_date']):
                            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
                        else:
                            # 尝试其他格式，允许转换失败的值为NaT
                            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
                            
                        # 检查并记录转换失败的情况
                        if df['trade_date'].isna().any():
                            invalid_count = df['trade_date'].isna().sum()
                            logger.warning(f"分组 {group_id} 数据中有 {invalid_count} 条记录日期转换失败")
                            # 删除转换失败的记录
                            df = df[df['trade_date'].notna()]
                            logger.info(f"删除无效日期后，分组 {group_id} 剩余数据: {len(df)} 条")
                    
                    # 过滤超出最大允许日期的数据
                    original_len = len(df)
                    df = df[df['trade_date'] <= max_allowed_date]
                    if len(df) < original_len:
                        logger.warning(f"分组 {group_id} 数据中存在 {original_len - len(df)} 条超出2025-12-31的记录，已过滤")
                else:
                    logger.warning(f"分组 {group_id} 数据中不存在 'trade_date' 列")
                
                # 只保留非空数据
                if not df.empty:
                    logger.info(f"分组 {group_id} 数据加载完成，共 {len(df)} 条有效记录")
                    all_data.append(df)
                else:
                    logger.warning(f"分组 {group_id} 数据为空，跳过")
                    
            except Exception as inner_e:
                logger.error(f"处理分组 {group_id} 数据时出错: {str(inner_e)}")
                # 创建空的DataFrame结构以避免后续处理错误
                empty_df = pd.DataFrame({
                    'trade_date': pd.Series([], dtype='datetime64[ns]'),
                    'monthly_return': pd.Series([], dtype='float64'),
                    'group_id': [group_id],
                    'group_name': [group_info[group_id]['name']],
                    'avg_market_cap': [group_info[group_id]['avg_cap']]
                })
                all_data.append(empty_df)
        
        # 过滤掉空DataFrame
        valid_data = [df for df in all_data if not df.empty]
        
        if not valid_data:
            logger.error("没有找到任何有效分组数据")
            return None, group_info
        
        # 合并所有分组数据
        combined_data = pd.concat(valid_data, ignore_index=True)
        
        # 记录最终数据的日期范围
        if not combined_data.empty and 'trade_date' in combined_data.columns:
            min_date = combined_data['trade_date'].min()
            max_date = combined_data['trade_date'].max()
            logger.info(f"成功合并 {len(combined_data)} 条数据记录")
            logger.info(f"数据日期范围: {min_date} 至 {max_date}")
        
        return combined_data, group_info
    
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}", exc_info=True)
        st.error(f"数据加载失败: {str(e)}")
        return None, None

def filter_data_by_time(data, start_date, end_date):
    """
    根据时间范围过滤数据
    
    Args:
        data: 原始数据
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        DataFrame: 过滤后的数据
    """
    try:
        # 检查输入数据
        if data is None or data.empty:
            logger.warning("输入数据为空")
            return data
            
        # 如果没有指定日期范围，返回全部数据
        if start_date is None or end_date is None:
            logger.info("未指定日期范围，返回全部数据")
            return data
        
        logger.info(f"开始按日期范围过滤数据: {start_date} 至 {end_date}")
        
        # 确保trade_date列存在
        if 'trade_date' not in data.columns:
            logger.warning("数据中不存在'trade_date'列")
            return data
        
        try:
            # 确保trade_date列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(data['trade_date']):
                logger.info("转换'trade_date'列为datetime类型")
                if pd.api.types.is_integer_dtype(data['trade_date']):
                    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d', errors='coerce')
                else:
                    data['trade_date'] = pd.to_datetime(data['trade_date'], errors='coerce')
                
                # 移除无效日期
                invalid_count = data['trade_date'].isna().sum()
                if invalid_count > 0:
                    logger.warning(f"发现 {invalid_count} 条无效日期记录，已移除")
                    data = data.dropna(subset=['trade_date'])
        except Exception as date_conv_error:
            logger.error(f"日期类型转换失败: {str(date_conv_error)}")
            return data
        
        # 转换开始和结束日期为日期对象（去除时间）
        try:
            start_datetime = pd.to_datetime(start_date).normalize()  # 转换为当天00:00:00
            end_datetime = pd.to_datetime(end_date).normalize()  # 转换为当天00:00:00
            
            # 过滤数据，使用日期部分进行比较
            filtered_data = data[(data['trade_date'].dt.date >= start_datetime.date()) & 
                                 (data['trade_date'].dt.date <= end_datetime.date())].copy()
            
            logger.info(f"过滤后数据量: {len(filtered_data)}")
            
            # 如果过滤后没有数据，记录警告并返回原始数据
            if len(filtered_data) == 0:
                logger.warning(f"指定日期范围内({start_date}至{end_date})没有数据")
                # 记录实际数据日期范围，帮助调试
                if not data.empty and 'trade_date' in data.columns:
                    actual_min_date = data['trade_date'].min()
                    actual_max_date = data['trade_date'].max()
                    logger.info(f"实际数据日期范围: {actual_min_date.strftime('%Y-%m-%d')} 至 {actual_max_date.strftime('%Y-%m-%d')}")
                return data  # 返回原始数据而不是空DataFrame
                
            return filtered_data
        except Exception as filter_error:
            logger.error(f"日期过滤过程中出错: {str(filter_error)}")
            return data
    except Exception as e:
        logger.error(f"按日期过滤数据时出错: {str(e)}", exc_info=True)
        return data  # 出错时返回原始数据而不是空DataFrame

def calculate_monthly_returns(data, groups):
    """
    计算月度收益率
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        
    Returns:
        DataFrame: 月度收益率数据
    """
    try:
        # 过滤选择的分组
        filtered_data = data[data['group_name'].isin(groups)]
        
        # 按日期和分组计算平均月度收益率
        monthly_returns = filtered_data.pivot_table(
            index='trade_date', 
            columns='group_name', 
            values='monthly_return', 
            aggfunc='mean'
        )
        
        # 按时间排序
        monthly_returns = monthly_returns.sort_index()
        
        return monthly_returns
    except Exception as e:
        logger.error(f"计算月度收益率失败: {str(e)}")
        return None

def calculate_rolling_annual_return(data, groups, window=12):
    """
    计算滚动年化收益率，处理早期数据
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        window: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动年化收益率数据
    """
    try:
        # 获取月度收益率
        monthly_returns = calculate_monthly_returns(data, groups)
        if monthly_returns is None:
            return None
        
        # 计算滚动年化收益率，使用min_periods=1确保早期数据也能显示
        # 对于不足窗口大小的数据，仍然计算但使用可用的历史数据
        rolling_annual = ((1 + monthly_returns).rolling(window=window, min_periods=1).apply(
            lambda x: np.prod(1+x)) - 1)
        
        # 标记数据有效性
        for i in range(min(window-1, len(rolling_annual))):
            # 在每个不足窗口大小的行添加标记
            if i < window-1:
                for col in rolling_annual.columns:
                    # 我们保留这些值但在显示时需要注意
                    pass
        
        logger.info(f"计算滚动年化收益率完成，数据行数: {len(rolling_annual)}")
        return rolling_annual
    except Exception as e:
        logger.error(f"计算滚动年化收益率失败: {str(e)}")
        return None

def calculate_rolling_volatility(data, groups, window=12):
    """
    计算滚动波动率，处理早期数据
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        window: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动波动率数据
    """
    try:
        # 获取月度收益率
        monthly_returns = calculate_monthly_returns(data, groups)
        if monthly_returns is None:
            return None
        
        # 计算滚动波动率（年化），使用min_periods=2确保至少有2个数据点计算标准差
        rolling_vol = monthly_returns.rolling(window=window, min_periods=2).std() * np.sqrt(12)
        
        logger.info(f"计算滚动波动率完成，数据行数: {len(rolling_vol)}")
        return rolling_vol
    except Exception as e:
        logger.error(f"计算滚动波动率失败: {str(e)}")
        return None

def calculate_rolling_sharpe(data, groups, window=12, risk_free_rate=0.02):
    """
    计算滚动夏普比率，处理早期数据和除以零的情况
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        window: 滚动窗口大小
        risk_free_rate: 无风险利率
        
    Returns:
        DataFrame: 滚动夏普比率数据
    """
    try:
        # 获取滚动年化收益率
        rolling_annual = calculate_rolling_annual_return(data, groups, window)
        # 获取滚动波动率
        rolling_vol = calculate_rolling_volatility(data, groups, window)
        
        if rolling_annual is None or rolling_vol is None:
            return None
        
        # 计算滚动夏普比率，处理除以零的情况
        # 使用np.where避免除以零
        rolling_sharpe = np.where(
            rolling_vol == 0,
            np.nan,  # 当波动率为0时设为NaN
            (rolling_annual - risk_free_rate) / rolling_vol
        )
        
        # 转换回DataFrame格式
        rolling_sharpe_df = pd.DataFrame(
            rolling_sharpe, 
            index=rolling_annual.index, 
            columns=rolling_annual.columns
        )
        
        logger.info(f"计算滚动夏普比率完成，数据行数: {len(rolling_sharpe_df)}")
        return rolling_sharpe_df
    except Exception as e:
        logger.error(f"计算滚动夏普比率失败: {str(e)}")
        return None

def calculate_time_series_metrics(data, groups, metric=None, rolling_window=12):
    """
    计算时间序列指标
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        metric: 指标类型 ('monthly_return', 'annual_return', 'volatility', 'sharpe')，如果为None则计算所有指标用于总评分
        rolling_window: 滚动窗口大小
        
    Returns:
        DataFrame: 单个指标时返回DataFrame，计算所有指标时返回包含各指标结果的字典
    """
    if metric is not None:
        # 单个指标计算模式
        if metric == 'monthly_return':
            return calculate_monthly_returns(data, groups)
        elif metric == 'annual_return':
            return calculate_rolling_annual_return(data, groups, rolling_window)
        elif metric == 'volatility':
            return calculate_rolling_volatility(data, groups, rolling_window)
        elif metric == 'sharpe':
            return calculate_rolling_sharpe(data, groups, rolling_window)
        else:
            logger.error(f"未知的指标类型: {metric}")
            return None
    else:
        # 计算所有指标用于总评分
        try:
            # 计算各个指标
            monthly_returns = calculate_monthly_returns(data, groups)
            annual_returns = calculate_rolling_annual_return(data, groups, rolling_window)
            volatility = calculate_rolling_volatility(data, groups, rolling_window)
            sharpe = calculate_rolling_sharpe(data, groups, rolling_window)
            
            # 计算总评分
            total_scores = None
            if monthly_returns is not None and annual_returns is not None and volatility is not None and sharpe is not None:
                # 使用加权平均计算总评分（收益率30%、波动率20%、夏普比率50%）
                # 首先进行标准化处理，将各指标映射到0-100分
                
                # 确保所有指标有相同的索引和列
                idx = monthly_returns.index
                cols = monthly_returns.columns
                
                # 标准化年化收益率（越高越好）
                annual_min, annual_max = annual_returns.min().min(), annual_returns.max().max()
                annual_score = 0
                if annual_max > annual_min:
                    annual_score = 100 * (annual_returns - annual_min) / (annual_max - annual_min)
                
                # 标准化波动率（越低越好）
                vol_min, vol_max = volatility.min().min(), volatility.max().max()
                vol_score = 100
                if vol_max > vol_min:
                    vol_score = 100 * (vol_max - volatility) / (vol_max - vol_min)
                
                # 标准化夏普比率（越高越好）
                sharpe_min, sharpe_max = sharpe.min().min(), sharpe.max().max()
                sharpe_score = 0
                if sharpe_max > sharpe_min:
                    sharpe_score = 100 * (sharpe - sharpe_min) / (sharpe_max - sharpe_min)
                
                # 计算加权总评分
                total_scores = 0.3 * annual_score + 0.2 * vol_score + 0.5 * sharpe_score
            
            return {
                'monthly_returns': monthly_returns,
                'annual_returns': annual_returns,
                'volatility': volatility,
                'sharpe': sharpe,
                'total_scores': total_scores
            }
        except Exception as e:
            logger.error(f"计算总评分时出错: {str(e)}")
            return {
                'monthly_returns': None,
                'annual_returns': None,
                'volatility': None,
                'sharpe': None,
                'total_scores': None
            }

# 已将总评分计算整合到现有的calculate_time_series_metrics函数中

def calculate_total_scores(metrics, groups):
    """
    计算总评分，基于年化收益率、波动率和夏普比率
    
    Args:
        metrics: 包含各种指标的字典
        groups: 要分析的分组
        
    Returns:
        DataFrame: 各分组的总评分
    """
    try:
        # 确保所有必要的指标都存在
        if not all(key in metrics and metrics[key] is not None for key in ['rolling_annual', 'rolling_vol', 'rolling_sharpe']):
            logger.warning("缺少计算总评分所需的指标")
            return None
        
        # 复制数据，避免修改原始数据
        annual_returns = metrics['rolling_annual'].copy()
        volatility = metrics['rolling_vol'].copy()
        sharpe = metrics['rolling_sharpe'].copy()
        
        # 初始化总评分DataFrame
        scores = pd.DataFrame(index=annual_returns.index, columns=groups)
        
        # 为每个时间点计算评分
        for date in annual_returns.index:
            # 获取当前日期的指标值
            annual_values = annual_returns.loc[date].dropna()
            vol_values = volatility.loc[date].dropna()
            sharpe_values = sharpe.loc[date].dropna()
            
            # 计算综合评分（权重：收益率30%，波动率20%，夏普比率50%）
            # 对每个分组分别计算
            for group in groups:
                if group in annual_values.index and group in vol_values.index and group in sharpe_values.index:
                    # 归一化处理
                    # 收益率得分：将收益率转换为0-100的得分
                    # 波动率得分：低波动率得高分，范围0-100
                    # 夏普比率得分：转换为0-100的得分
                    
                    # 计算收益率得分（基于历史最大值和最小值）
                    hist_annual_min = annual_returns[group].min()
                    hist_annual_max = annual_returns[group].max()
                    annual_score = 0
                    if hist_annual_max > hist_annual_min:
                        annual_score = 100 * (annual_values[group] - hist_annual_min) / (hist_annual_max - hist_annual_min)
                    
                    # 计算波动率得分（低波动率得高分）
                    hist_vol_min = volatility[group].min()
                    hist_vol_max = volatility[group].max()
                    vol_score = 0
                    if hist_vol_max > hist_vol_min:
                        # 波动率越低得分越高
                        vol_score = 100 * (1 - (vol_values[group] - hist_vol_min) / (hist_vol_max - hist_vol_min))
                    
                    # 计算夏普比率得分
                    hist_sharpe_min = sharpe[group].min()
                    hist_sharpe_max = sharpe[group].max()
                    sharpe_score = 0
                    if hist_sharpe_max > hist_sharpe_min:
                        sharpe_score = 100 * (sharpe_values[group] - hist_sharpe_min) / (hist_sharpe_max - hist_sharpe_min)
                    
                    # 计算总评分（加权平均）
                    scores.loc[date, group] = 0.3 * annual_score + 0.2 * vol_score + 0.5 * sharpe_score
        
        # 处理可能的NaN值
        scores = scores.fillna(0)
        
        logger.info(f"计算总评分完成，数据行数: {len(scores)}")
        return scores
    except Exception as e:
        logger.error(f"计算总评分失败: {str(e)}")
        return None

def plot_time_series(ax, data, title, ylabel, is_percentage=False):
    """
    绘制时间序列图表
    
    Args:
        ax: matplotlib轴对象
        data: 要绘制的数据
        title: 图表标题
        ylabel: Y轴标签
        is_percentage: 是否按百分比显示
    """
    try:
        # 为每个组别绘制折线
        for column in data.columns:
            ax.plot(data.index, data[column], marker='o', markersize=3, linewidth=2, label=column)
        
        # 设置图表属性
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='市值组别', fontsize=10)
        
        # 设置日期格式
        fig = ax.get_figure()
        fig.autofmt_xdate()
        
        # 如果是百分比格式，设置Y轴格式
        if is_percentage:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            
    except Exception as e:
        logger.error(f"绘制图表失败: {str(e)}")
        raise

# 设置页面配置
st.set_page_config(
    page_title="A股市场规模效应研究",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面标题
st.title("A股市场规模效应研究")
st.markdown("""
    本工具用于可视化分析A股市场中不同市值组别股票的表现.
    您可以选择不同的市值组别、时间范围和指标进行交互式分析.
""")

# 侧边栏 - 用户输入区域
with st.sidebar:
    st.header("参数设置")
    
    # 加载分组信息
    _, group_info = load_group_data()
    
    # 分组选择功能
    st.subheader("选择分组")
    if group_info:
        # 提取所有分组名称
        all_groups = [group_info[group_id]['name'] for group_id in sorted(group_info.keys())]
        
        # 分组多选控件，改为勾选形式
        st.markdown("选择要分析的市值组别：")
        selected_groups = []
        cols = st.columns(2)
        for i, group_name in enumerate(all_groups):
            col_idx = i % 2
            if cols[col_idx].checkbox(group_name, value=True, help=f"显示{group_name}的表现数据及分析"):
                selected_groups.append(group_name)
        
        # 显示已选分组的平均市值信息
        if selected_groups:
            st.write("\n已选分组信息：")
            for group_name in selected_groups:
                # 找到对应的分组ID
                group_id = next(gid for gid, info in group_info.items() if info['name'] == group_name)
                avg_cap = group_info[group_id]['avg_cap']
                st.write(f"- {group_name}: 平均市值约 {avg_cap} 亿元")
    else:
        st.warning("无法加载分组信息")
        selected_groups = []
    
    # 时间范围选择
    st.subheader("选择时间范围")
    
    # 预设时间范围选项
    time_period_options = {
        "全部数据": None,
        "近1年": 365,
        "近3年": 1095,
        "近5年": 1825
    }
    
    # 时间范围选择器
    selected_time_period = st.selectbox(
        "快速选择时间范围：",
        options=list(time_period_options.keys()),
        index=0,
        help="快速选择分析的时间区间，影响数据覆盖范围"
    )
    
    # 自定义日期选择器
    use_custom_date = st.checkbox("使用自定义日期范围", value=False, help="启用自定义日期选择，可精确控制分析的起始和结束时间")
    
    # 初始化日期变量
    start_date = None
    end_date = None
    
    # 如果有数据，获取数据的日期范围
    data, _ = load_group_data()
    min_date = None
    max_date = None
    
    if data is not None and not data.empty:
        min_date = data['trade_date'].min().date()
        max_date = data['trade_date'].max().date()
    else:
        # 默认日期范围
        max_date = pd.Timestamp.now().date()
        min_date = max_date - pd.Timedelta(days=365*5)
    
    # 自定义日期范围选择
    if use_custom_date:
        start_date = st.date_input(
            "开始日期：",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        end_date = st.date_input(
            "结束日期：",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    else:
        # 根据预设选项计算日期范围
        if time_period_options[selected_time_period] is not None:
            days = time_period_options[selected_time_period]
            end_date = max_date
            start_date = end_date - pd.Timedelta(days=days)
            # 确保不超过数据的最小日期
            if start_date < min_date:
                start_date = min_date
        else:
            # 全部数据
            start_date = min_date
            end_date = max_date
    
    # 指标选择
    st.subheader("选择分析指标")
    metrics_options = {
        "月度收益率": "monthly_return",
        "滚动年化收益率": "annual_return",
        "滚动波动率": "volatility",
        "滚动夏普比率": "sharpe"
    }
    
    # 指标解释字典
    metric_explanations = {
        "月度收益率": "反映每月投资回报百分比，直接展示短期收益表现",
        "滚动年化收益率": "基于指定窗口的月度收益率计算的年化回报率，衡量长期收益能力",
        "滚动波动率": "反映价格变动的剧烈程度，衡量投资风险水平",
        "滚动夏普比率": "衡量风险调整后的投资回报，综合考虑收益和风险"
    }
    
    # 指标选择，改为勾选形式
    st.markdown("选择要分析的指标：")
    selected_metrics = []
    for metric_display in list(metrics_options.keys()):
        default_selected = metric_display in ["月度收益率", "滚动年化收益率"]
        if st.checkbox(metric_display, value=default_selected, help=f"{metric_display}：{metric_explanations[metric_display]}"):
            selected_metrics.append(metric_display)
    
    # 滚动窗口大小选择
    st.subheader("滚动窗口设置")
    window_size = st.slider(
        "滚动窗口大小（月）：",
        min_value=3,
        max_value=36,
        value=12,
        step=1,
        help="计算滚动指标的时间窗口，窗口越大结果越稳定但对变化反应越慢，窗口越小结果越敏感但波动越大")
    
    # 图表设置说明
    st.subheader("图表使用提示")
    st.info("💡 图表支持使用鼠标滚轮放大缩小，拖动平移查看细节")
    
    st.divider()
    st.info("选择参数后，点击分析按钮查看结果")

# 主内容区
main_content = st.container()

# 函数定义区域

def calculate_monthly_returns(data, groups):
    """
    计算月度收益率
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        
    Returns:
        DataFrame: 月度收益率数据
    """
    try:
        # 过滤选择的分组
        filtered_data = data[data['group_name'].isin(groups)]
        
        # 按日期和分组计算平均月度收益率
        monthly_returns = filtered_data.pivot_table(
            index='trade_date', 
            columns='group_name', 
            values='monthly_return', 
            aggfunc='mean'
        )
        
        # 按时间排序
        monthly_returns = monthly_returns.sort_index()
        
        return monthly_returns
    except Exception as e:
        logger.error(f"计算月度收益率失败: {str(e)}")
        return None

def calculate_rolling_annual_return(data, groups, window=12):
    """
    计算滚动年化收益率
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        window: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动年化收益率数据
    """
    try:
        # 获取月度收益率
        monthly_returns = calculate_monthly_returns(data, groups)
        if monthly_returns is None:
            return None
        
        # 计算滚动年化收益率
        rolling_annual = ((1 + monthly_returns).rolling(window=window).apply(
            lambda x: np.prod(1+x)) - 1)
        
        return rolling_annual
    except Exception as e:
        logger.error(f"计算滚动年化收益率失败: {str(e)}")
        return None

def calculate_rolling_volatility(data, groups, window=12):
    """
    计算滚动波动率
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        window: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动波动率数据
    """
    try:
        # 获取月度收益率
        monthly_returns = calculate_monthly_returns(data, groups)
        if monthly_returns is None:
            return None
        
        # 计算滚动波动率（年化）
        rolling_vol = monthly_returns.rolling(window=window).std() * np.sqrt(12)
        
        return rolling_vol
    except Exception as e:
        logger.error(f"计算滚动波动率失败: {str(e)}")
        return None

def calculate_rolling_sharpe(data, groups, window=12, risk_free_rate=0.02):
    """
    计算滚动夏普比率
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        window: 滚动窗口大小
        risk_free_rate: 无风险利率
        
    Returns:
        DataFrame: 滚动夏普比率数据
    """
    try:
        # 获取滚动年化收益率
        rolling_annual = calculate_rolling_annual_return(data, groups, window)
        # 获取滚动波动率
        rolling_vol = calculate_rolling_volatility(data, groups, window)
        
        if rolling_annual is None or rolling_vol is None:
            return None
        
        # 计算滚动夏普比率
        rolling_sharpe = (rolling_annual - risk_free_rate) / rolling_vol
        
        return rolling_sharpe
    except Exception as e:
        logger.error(f"计算滚动夏普比率失败: {str(e)}")
        return None

def calculate_time_series_metrics(data, groups, metric=None, rolling_window=12):
    """
    计算时间序列指标
    
    Args:
        data: 分组数据
        groups: 要分析的分组
        metric: 指标类型 ('monthly_return', 'annual_return', 'volatility', 'sharpe')，为None时返回所有指标
        rolling_window: 滚动窗口大小
        
    Returns:
        DataFrame或dict: 单个指标返回DataFrame，多个指标返回包含所有指标的字典
    """
    # 如果指定了具体指标，只返回该指标
    if metric:
        if metric == 'monthly_return':
            return calculate_monthly_returns(data, groups)
        elif metric == 'annual_return':
            return calculate_rolling_annual_return(data, groups, rolling_window)
        elif metric == 'volatility':
            return calculate_rolling_volatility(data, groups, rolling_window)
        elif metric == 'sharpe':
            return calculate_rolling_sharpe(data, groups, rolling_window)
        else:
            return None
    
    # 如果metric为None，计算所有指标并返回字典
    try:
        # 计算所有基础指标
        monthly_returns = calculate_monthly_returns(data, groups)
        rolling_annual = calculate_rolling_annual_return(data, groups, rolling_window)
        rolling_vol = calculate_rolling_volatility(data, groups, rolling_window)
        rolling_sharpe = calculate_rolling_sharpe(data, groups, rolling_window)
        
        # 初始化结果字典
        result = {
            'monthly_returns': monthly_returns,
            'rolling_annual': rolling_annual,
            'rolling_vol': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'total_scores': None  # 初始化总评分
        }
        
        # 只有当所有必要的指标都非空时才计算总评分
        if (monthly_returns is not None and rolling_annual is not None and 
            rolling_vol is not None and rolling_sharpe is not None):
            
            # 初始化总评分DataFrame
            scores = pd.DataFrame(index=rolling_annual.index, columns=groups)
            
            # 为每个时间点计算评分
            for date in rolling_annual.index:
                # 获取当前日期的指标值
                annual_values = rolling_annual.loc[date].dropna()
                vol_values = rolling_vol.loc[date].dropna()
                sharpe_values = rolling_sharpe.loc[date].dropna()
                
                # 计算综合评分（权重：收益率30%，波动率20%，夏普比率50%）
                # 对每个分组分别计算
                for group in groups:
                    if group in annual_values.index and group in vol_values.index and group in sharpe_values.index:
                        try:
                            # 计算收益率得分（基于历史最大值和最小值）
                                hist_annual_min = rolling_annual[group].min()
                                hist_annual_max = rolling_annual[group].max()
                                annual_score = 0
                                if hist_annual_max > hist_annual_min:
                                    # 收益率越高越好，转换为0-100分
                                    annual_score = 100 * (annual_values[group] - hist_annual_min) / (hist_annual_max - hist_annual_min)
                                
                                # 计算波动率得分（基于历史最大值和最小值）
                                hist_vol_min = rolling_vol[group].min()
                                hist_vol_max = rolling_vol[group].max()
                                vol_score = 0
                                if hist_vol_max > hist_vol_min:
                                    # 波动率越低越好，所以进行反向评分
                                    vol_score = 100 * (1 - (vol_values[group] - hist_vol_min) / (hist_vol_max - hist_vol_min))
                                    # 额外处理：对于极高的波动率给予更低的分数
                                    if vol_values[group] > hist_vol_min + 0.75 * (hist_vol_max - hist_vol_min):
                                        vol_score = vol_score * 0.7  # 对高波动进行惩罚
                                
                                # 计算夏普比率得分
                                hist_sharpe_min = rolling_sharpe[group].min()
                                hist_sharpe_max = rolling_sharpe[group].max()
                                sharpe_score = 0
                                if hist_sharpe_max > hist_sharpe_min:
                                    # 夏普比率越高越好，转换为0-100分
                                    sharpe_score = 100 * (sharpe_values[group] - hist_sharpe_min) / (hist_sharpe_max - hist_sharpe_min)
                                    # 额外处理：对于负的夏普比率给予更低的分数
                                    if sharpe_values[group] < 0:
                                        sharpe_score = sharpe_score * 0.5  # 对负夏普比率进行惩罚
                                
                                # 计算总评分（加权平均）
                                # 权重分配：收益率(30%)、波动率(20%)、夏普比率(50%)
                                # 基于投资价值评估原则：风险调整回报(夏普比率)最重要，其次是绝对收益，最后是风险控制
                                scores.loc[date, group] = 0.3 * annual_score + 0.2 * vol_score + 0.5 * sharpe_score
                        except Exception as e:
                            logger.error(f"计算{group}的评分时出错: {str(e)}")
                            scores.loc[date, group] = 0
            
            # 处理可能的NaN值
            scores = scores.fillna(0)
            result['total_scores'] = scores
            logger.info(f"计算总评分完成，数据行数: {len(scores)}")
        
        return result
    except Exception as e:
        logger.error(f"计算指标集时出错: {str(e)}")
        return None
        return calculate_rolling_sharpe(data, groups, rolling_window)
    else:
        logger.error(f"未知的指标类型: {metric}")
        return None

def plot_time_series(ax, data, title, ylabel, is_percentage=False):
    """
    绘制时间序列图表
    
    Args:
        ax: matplotlib轴对象
        data: 要绘制的数据
        title: 图表标题
        ylabel: Y轴标签
        is_percentage: 是否按百分比显示
    """
    try:
        # 为每个组别绘制折线
        for column in data.columns:
            ax.plot(data.index, data[column], marker='o', markersize=3, linewidth=2, label=column)
        
        # 设置图表属性
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='市值组别', fontsize=10)
        
        # 设置日期格式
        fig = ax.get_figure()
        fig.autofmt_xdate()
        
        # 如果是百分比格式，设置Y轴格式
        if is_percentage:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            
    except Exception as e:
        logger.error(f"绘制图表失败: {str(e)}")
        raise

# 应用已经包含优化后的calculate_metric函数定义

# 主内容区域
with main_content:
    st.header("A股市场规模效应研究")
    st.write("""
    本应用展示了A股市场不同市值组别(小市值到大市值)的投资表现指标.
    通过选择不同的组别,时间范围和指标类型,可以进行多角度的规模效应分析.
    """)
    
    # 添加应用说明和提示
    st.info("💡 提示: 本应用已针对Streamlit Cloud部署进行了优化，包含全面的错误处理机制，确保稳定运行。")
    
    # 显示数据状态信息
    if data is not None and not data.empty:
        st.success(f"✅ 数据已加载: 共 {len(data)} 条记录")
        if 'trade_date' in data.columns:
            try:
                min_date = data['trade_date'].min()
                max_date = data['trade_date'].max()
                st.info(f"📅 数据日期范围: {min_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')}")
            except Exception as date_error:
                logger.warning(f"日期显示错误: {str(date_error)}")
    
    # 显示分析按钮
    if st.button("开始分析", use_container_width=True):
        # 检查是否选择了分组
        if not selected_groups:
            st.error("请至少选择一个市值组别")
        else:
            try:
                # 加载数据
                with st.spinner("正在加载数据..."):
                    data, _ = load_group_data()
                    
                if data is not None and not data.empty:
                    # 根据选择的时间范围过滤数据
                    with st.spinner("正在筛选数据..."):
                        filtered_data = filter_data_by_time(data, start_date, end_date)
                    
                    # 检查过滤后的数据
                    if filtered_data is None or filtered_data.empty:
                        st.warning("⚠️ 过滤后的数据为空，使用全部数据进行分析")
                        filtered_data = data
                    
                    st.success(f"✅ 成功加载数据，已选择 {len(selected_groups)} 个分组进行分析")
                    
                    # 显示数据统计信息
                    st.subheader("📊 数据概览")
                    stats_container = st.container()
                    with stats_container:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("总数据量", len(filtered_data))
                        col2.metric("滚动窗口大小", f"{window_size} 个月")
                        
                        # 安全地显示时间范围
                        try:
                            if 'trade_date' in filtered_data.columns:
                                min_date_val = filtered_data['trade_date'].min()
                                max_date_val = filtered_data['trade_date'].max()
                                col3.metric("时间跨度", f"{min_date_val.strftime('%Y-%m')} 至 {max_date_val.strftime('%Y-%m')}")
                            else:
                                col3.metric("时间跨度", "不可用")
                        except Exception as date_error:
                            logger.warning(f"日期统计错误: {str(date_error)}")
                            col3.metric("时间跨度", "计算错误")
                    
                    # 显示前5行数据作为样例
                    st.subheader("📋 数据样例")
                    st.dataframe(filtered_data.head())
                    
                    # 指标计算和可视化
                    for metric_display in selected_metrics:
                        metric_key = metrics_options[metric_display]
                        
                        st.subheader(f"📈 {metric_display} 分析")
                        
                        # 计算指标
                        with st.spinner(f"正在计算{metric_display}..."):
                            try:
                                result_data = calculate_time_series_metrics(
                                    filtered_data, selected_groups, metric_key, window_size
                                )
                                
                                if result_data is not None and not result_data.empty:
                                    st.success(f"✅ {metric_display} 计算完成")
                                    
                                    # 显示数据统计
                                    st.write(f"计算了 {len(result_data)} 个时间点的{metric_display}")
                                    
                                    # 计算每个分组的平均值
                                    st.subheader("🏆 指标平均值")
                                    try:
                                        avg_values = result_data.mean()
                                        avg_df = pd.DataFrame({
                                            "平均值": avg_values,
                                            "排名": avg_values.rank(ascending=metric_key != "volatility")
                                        })
                                        
                                        # 格式化显示
                                        if metric_key in ["monthly_return", "annual_return"]:
                                            avg_df["平均值"] = avg_df["平均值"].apply(lambda x: f"{x:.2%}")
                                        elif metric_key == "volatility":
                                            avg_df["平均值"] = avg_df["平均值"].apply(lambda x: f"{x:.2%}")
                                        elif metric_key == "sharpe":
                                            avg_df["平均值"] = avg_df["平均值"].apply(lambda x: f"{x:.2f}")
                                        
                                        st.dataframe(avg_df)
                                    except Exception as calc_error:
                                        logger.error(f"计算平均值时出错: {str(calc_error)}")
                                        st.warning("⚠️ 无法计算指标平均值，但将继续显示图表")
                                    
                                    # 绘制图表
                                    st.subheader(f"📊 {metric_display} 时间序列图")
                                    
                                    try:
                                        # 使用plotly绘制交互式图表
                                        import plotly.graph_objects as go
                                        
                                        fig = go.Figure()
                                        
                                        # 为每个分组添加折线
                                        for group in result_data.columns:
                                            fig.add_trace(go.Scatter(
                                                x=result_data.index,
                                                y=result_data[group],
                                                mode='lines+markers',
                                                name=group,
                                                marker=dict(size=5),
                                                line=dict(width=2)
                                            ))
                                        
                                        # 设置图表属性
                                        fig.update_layout(
                                            title={
                                                'text': f'{metric_display} 时间序列比较',
                                                'font': {'size': 18, 'weight': 'bold'}
                                            },
                                            xaxis_title='日期',
                                            yaxis_title=metric_display,
                                            legend_title='市值组别',
                                            hovermode='x unified',
                                            template='plotly_white',
                                            # 默认高度已优化为700像素，支持鼠标滚轮缩放
                                            margin=dict(l=60, r=60, t=60, b=60)
                                        )
                                        
                                        # 根据指标类型设置Y轴格式
                                        if metric_key in ["monthly_return", "annual_return", "volatility"]:
                                            fig.update_layout(
                                                yaxis=dict(
                                                    tickformat='.1%'
                                                )
                                            )
                                        
                                        # 显示图表，增加滚动信息说明
                                        st.markdown("**注意:** 滚动指标（年化收益率、波动率、夏普比率）在数据开始阶段可能使用部分窗口计算，随着时间推移才会使用完整窗口大小。")
                                        st.plotly_chart(fig, use_container_width=True, config={
                                            'scrollZoom': True,  # 确保启用滚轮缩放
                                            'displayModeBar': True,
                                            'toImageButtonOptions': {
                                                'format': 'png',
                                                'filename': 'scale_effect_chart',
                                                'height': 700,
                                                'width': 1200,
                                                'scale': 2
                                            }
                                        })
                                        
                                    except Exception as plot_error:
                                        logger.error(f"绘制图表失败: {str(plot_error)}")
                                        st.warning(f"⚠️ 图表绘制失败: {str(plot_error)}，但将继续处理其他指标")
                                else:
                                    st.warning(f"⚠️ 无法计算{metric_display}数据")
                            except Exception as metric_error:
                                logger.error(f"计算{metric_display}时出错: {str(metric_error)}")
                                st.warning(f"⚠️ 计算{metric_display}时发生错误，但将继续处理其他指标")
                        
                        st.divider()
                    
                    # 计算所有指标用于总评分
                    metrics = calculate_time_series_metrics(filtered_data, selected_groups, metric=None, rolling_window=window_size)
                    
                    # 显示总评分折线图
                    st.subheader("📊 总评分折线图")
                    
                    # 安全检查metrics和total_scores
                    if metrics is not None and 'total_scores' in metrics and metrics['total_scores'] is not None and not metrics['total_scores'].empty:
                        # 创建总评分图表
                        fig = go.Figure()
                        
                        # 为每个分组添加评分线
                        for group in selected_groups:
                            if group in metrics['total_scores'].columns:
                                # 计算该分组的评分数据
                                scores = metrics['total_scores'][group]
                                
                                # 添加折线
                                fig.add_trace(go.Scatter(
                                    x=scores.index, 
                                    y=scores, 
                                    mode='lines+markers',
                                    name=group,
                                    line=dict(width=2),
                                    marker=dict(size=5, opacity=0.7)
                                ))
                        
                        # 添加评分均值线（作为参考）
                        avg_scores = metrics['total_scores'].mean(axis=1)
                        fig.add_trace(go.Scatter(
                            x=avg_scores.index, 
                            y=avg_scores, 
                            mode='lines',
                            name='平均评分',
                            line=dict(width=2, dash='dash', color='black'),
                            hoverinfo='skip',
                            legendrank=10  # 确保均值线在图例底部
                        ))
                        
                        # 设置布局
                        fig.update_layout(
                            title='各分组总评分变化趋势',
                            xaxis_title='日期',
                            yaxis_title='评分 (0-100)',
                            legend_title='分组',
                            hovermode='x unified',
                            template='plotly_white',
                            height=600,
                            margin=dict(l=60, r=40, t=60, b=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        # 设置Y轴范围
                        fig.update_yaxes(range=[0, 100])
                        
                        # 添加参考线
                        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
                        
                        # 显示图表
                        st.plotly_chart(fig, use_container_width=True, config={
                            'scrollZoom': True,
                            'displayModeBar': True,
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'scale_effect_scores',
                                'height': 600,
                                'width': 1200,
                                'scale': 2
                            }
                        })
                        
                        # 添加评分说明
                        st.markdown("### 评分说明")
                        st.markdown("- 总评分基于三个核心投资指标加权计算，体现全面的投资价值评估：")
                        st.markdown("  - **收益率(30%)**：衡量投资回报水平，收益率越高评分越高")
                        st.markdown("  - **波动率(20%)**：衡量风险水平，波动率越低评分越高，高波动率会受到额外惩罚")
                        st.markdown("  - **夏普比率(50%)**：衡量风险调整后回报，是最重要的综合指标，负夏普比率会受到额外惩罚")
                        st.markdown("- 评分范围：0-100，**分值越高表示综合表现越好**，反映投资组合的质量")
                        st.markdown("- 黑色虚线表示所有分组的平均评分，可作为市场基准参考")
                        
                        # 显示最新评分统计
                        st.markdown("### 最新评分统计")
                        latest_scores = metrics['total_scores'].iloc[-1]
                        best_score_group = latest_scores.idxmax()
                        best_score_value = latest_scores.max()
                        
                        st.markdown(f"- **最佳评分:** {best_score_group}，评分值: {best_score_value:.1f}")
                        
                        # 计算评分排名
                        sorted_scores = latest_scores.sort_values(ascending=False)
                        ranking_df = pd.DataFrame({
                            '分组': sorted_scores.index,
                            '评分': sorted_scores.values
                        }).round(1)
                        
                        # 显示排名表格
                        st.dataframe(ranking_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("无法计算总评分数据")
                    

                    
                else:
                    st.error("❌ 无法加载数据或数据为空")
                    # 如果本地开发环境，显示更多调试信息
                    if IS_LOCAL:
                        st.info("开发环境调试信息:")
                        st.info(f"数据目录: {DATA_DIR}")
                        st.info(f"目录内容: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else '目录不存在'}")
                    
            except Exception as e:
                logger.error(f"分析过程中出错: {str(e)}", exc_info=True)
                st.error(f"❌ 分析过程中发生错误: {str(e)}")
                # 如果是本地开发环境，显示详细错误信息
                if IS_LOCAL:
                    st.exception(e)
                else:
                    st.info("如果问题持续，请联系管理员或查看应用日志")
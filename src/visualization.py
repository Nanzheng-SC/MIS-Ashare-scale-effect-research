import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def create_visualizations(grouped_data):
    """
    为五个市值分组创建可视化图表，展示A股市场规模效应
    
    Args:
        grouped_data: 包含五个分组数据的字典
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建结果目录
    os.makedirs('../results', exist_ok=True)
    
    # 1. 各分组收益率分布箱线图
    plt.figure(figsize=(12, 6))
    returns_data = []
    group_labels = []
    colors = sns.color_palette('RdYlGn_r', len(grouped_data))  # 红色到绿色的渐变色
    
    for group_num, data in sorted(grouped_data.items()):
        # 计算每个分组的周收益率
        weekly_returns = data['weekly_return'].dropna()
        returns_data.append(weekly_returns)
        group_labels.append(f'第{group_num}组')
    
    box_plot = plt.boxplot(returns_data, labels=group_labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('各市值分组周收益率分布', fontsize=14)
    plt.xlabel('市值分组（1=最小，5=最大）', fontsize=12)
    plt.ylabel('周收益率', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/returns_distribution.png', dpi=300)
    plt.close()
    
    # 2. 各分组累计收益率折线图（规模效应核心图表）
    plt.figure(figsize=(14, 8))
    colors = sns.color_palette('viridis', len(grouped_data))
    
    for group_num, data in sorted(grouped_data.items()):
        if not data.empty:
            # 按月份计算平均收益
            monthly_returns = data.groupby('year_month')['weekly_return'].mean().fillna(0)
            
            # 计算累计收益
            cumulative_returns = (1 + monthly_returns).cumprod()
            
            # 转换索引为日期格式以便绘图
            cumulative_returns.index = cumulative_returns.index.astype(str)
            
            # 绘制折线图
            plt.plot(cumulative_returns.index, cumulative_returns, 
                    label=f'第{group_num}组（{'小市值' if group_num == 1 else '大市值' if group_num == 5 else '中等市值'}）',
                    linewidth=2)
    
    # 添加小市值减大市值的超额收益线（规模效应的直接体现）
    if 1 in grouped_data and 5 in grouped_data:
        # 计算小市值和大市值的月度平均收益
        small_cap_monthly = grouped_data[1].groupby('year_month')['weekly_return'].mean().fillna(0)
        large_cap_monthly = grouped_data[5].groupby('year_month')['weekly_return'].mean().fillna(0)
        
        # 计算超额收益
        size_premium = small_cap_monthly - large_cap_monthly
        cumulative_premium = (1 + size_premium).cumprod()
        cumulative_premium.index = cumulative_premium.index.astype(str)
        
        # 绘制超额收益线
        plt.plot(cumulative_premium.index, cumulative_premium, 
                label='小市值-大市值超额收益', 
                linestyle='--', linewidth=3, color='red')
    
    plt.title('A股市场规模效应：各市值分组累计收益率对比（2019-2025）', fontsize=14)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('累计收益率（初始=1）', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加网格线和标注
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../results/cumulative_returns.png', dpi=300)
    plt.close()
    
    # 3. 各分组平均收益率柱状图（直观展示规模效应）
    plt.figure(figsize=(10, 6))
    avg_returns = []
    group_nums = []
    
    for group_num, data in sorted(grouped_data.items()):
        if not data.empty:
            avg_return = data['weekly_return'].mean() * 100  # 转换为百分比
            avg_returns.append(avg_return)
            group_nums.append(group_num)
    
    # 使用渐变色，突出小市值和大市值的对比
    colors = sns.color_palette('RdYlGn_r', len(avg_returns))
    bars = plt.bar([f'第{i}组' for i in group_nums], avg_returns, color=colors)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                f'{height:.3f}%', 
                ha='center', va='bottom')
    
    plt.title('A股市场规模效应：各市值分组平均周收益率（2019-2025）', fontsize=14)
    plt.xlabel('市值分组（1=最小，5=最大）', fontsize=12)
    plt.ylabel('平均周收益率（%）', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加一条水平线表示零收益
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/average_returns.png', dpi=300)
    plt.close()
    
    # 4. 各分组收益率热力图（按年份）
    # 准备数据
    heatmap_data = []
    years = sorted(set([str(date)[:4] for data in grouped_data.values() 
                      for date in data['trade_date'].astype(str) if pd.notna(date)]))
    
    for group_num, data in sorted(grouped_data.items()):
        group_returns = []
        for year in years:
            # 筛选该年份的数据
            year_mask = data['trade_date'].astype(str).str.startswith(year)
            yearly_data = data[year_mask]
            if not yearly_data.empty:
                yearly_return = yearly_data['weekly_return'].mean() * 100
            else:
                yearly_return = 0
            group_returns.append(yearly_return)
        heatmap_data.append(group_returns)
    
    # 创建DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, 
                            index=[f'第{i}组' for i in sorted(grouped_data.keys())],
                            columns=years)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn', center=0, fmt='.2f',
               cbar_kws={'label': '平均收益率（%）'})
    plt.title('A股市场规模效应：各年份各市值分组平均收益率热力图（%）', fontsize=14)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('市值分组（1=最小，5=最大）', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/returns_heatmap.png', dpi=300)
    plt.close()
    
    # 5. 新增：规模效应时间序列图（小市值-大市值的月度超额收益）
    if 1 in grouped_data and 5 in grouped_data:
        plt.figure(figsize=(14, 6))
        
        # 计算月度超额收益
        small_cap_monthly = grouped_data[1].groupby('year_month')['weekly_return'].mean().fillna(0)
        large_cap_monthly = grouped_data[5].groupby('year_month')['weekly_return'].mean().fillna(0)
        size_premium = small_cap_monthly - large_cap_monthly
        size_premium = size_premium * 100  # 转换为百分比
        
        # 转换索引为字符串格式
        size_premium.index = size_premium.index.astype(str)
        
        # 绘制柱状图
        colors = ['green' if x > 0 else 'red' for x in size_premium]
        plt.bar(size_premium.index, size_premium, color=colors, alpha=0.7)
        
        # 添加移动平均线
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title('A股市场规模效应：小市值-大市值月度超额收益率（2019-2025）', fontsize=14)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('超额收益率（%）', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('../results/size_premium_ts.png', dpi=300)
        plt.close()
    
    print("所有可视化图表已保存到results文件夹")
    print("\n规模效应研究图表说明：")
    print("1. returns_distribution.png: 各市值分组的周收益率分布箱线图")
    print("2. cumulative_returns.png: 各分组累计收益率对比，包含小市值-大市值超额收益")
    print("3. average_returns.png: 各分组平均周收益率柱状图，直观展示规模效应")
    print("4. returns_heatmap.png: 按年份展示的各分组收益率热力图")
    if 1 in grouped_data and 5 in grouped_data:
        print("5. size_premium_ts.png: 小市值-大市值月度超额收益率时间序列图")
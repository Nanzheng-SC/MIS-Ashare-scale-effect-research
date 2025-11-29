import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_and_group_data(stock_data):
    """
    处理股票数据并按市值进行五分位分组
    
    Args:
        stock_data: 原始股票数据
    
    Returns:
        dict: 包含五个分组数据的字典
    """
    print("开始处理数据...")
    
    # 确保日期列格式正确
    if isinstance(stock_data['trade_date'].iloc[0], str):
        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'], format='%Y%m%d')
    
    # 计算周收益率
    stock_data['weekly_return'] = stock_data.groupby('ts_code')['close'].pct_change()
    
    # 检查是否已经有市值数据，如果没有则创建
    if 'estimated_market_cap' not in stock_data.columns:
        # 为了简化，我们使用收盘价乘以一个估算的股本数
        stock_data['estimated_market_cap'] = stock_data['close'] * 1000000  # 简化计算
    
    # 去除NaN值
    stock_data = stock_data.dropna(subset=['estimated_market_cap', 'weekly_return'])
    
    # 按时间周期分组进行市值分组
    # 这里我们使用月度重分组，这是研究规模效应的常见做法
    result_dict = {}
    
    # 1. 先按月对数据进行分组
    stock_data['year_month'] = stock_data['trade_date'].dt.to_period('M')
    grouped_by_month = stock_data.groupby('year_month')
    
    print(f"共有{len(grouped_by_month)}个月的数据需要处理")
    
    # 存储每个分组的数据
    groups_data = {i: [] for i in range(1, 6)}
    
    for year_month, monthly_data in grouped_by_month:
        print(f"处理{year_month}的数据...")
        
        # 确保每个月份有足够的数据进行分组
        if len(monthly_data) >= 5:
            try:
                # 按市值进行五分位分组
                # 使用rank方法确保即使有重复值也能正确分组
                monthly_data['market_cap_rank'] = monthly_data['estimated_market_cap'].rank(method='first')
                total_stocks = len(monthly_data)
                
                # 手动计算分组界限，避免使用qcut可能遇到的重复值问题
                monthly_data['market_cap_group'] = pd.cut(
                    monthly_data['market_cap_rank'],
                    bins=[0, total_stocks*0.2, total_stocks*0.4, total_stocks*0.6, total_stocks*0.8, total_stocks+1],
                    labels=[1, 2, 3, 4, 5]
                )
                
                # 保存每个分组的数据
                for group_num in range(1, 6):
                    group_data = monthly_data[monthly_data['market_cap_group'] == group_num]
                    if not group_data.empty:
                        groups_data[group_num].append(group_data)
            except Exception as e:
                print(f"处理{year_month}数据时出错: {e}")
                continue
    
    # 合并每个分组的数据
    for group_num in range(1, 6):
        if groups_data[group_num]:
            result_dict[group_num] = pd.concat(groups_data[group_num], ignore_index=True)
            # 计算该组的平均市值和平均收益率
            avg_market_cap = result_dict[group_num]['estimated_market_cap'].mean()
            avg_return = result_dict[group_num]['weekly_return'].mean()
            print(f"第{group_num}组（{'小市值' if group_num == 1 else '大市值' if group_num == 5 else '中等市值'}）共有{len(result_dict[group_num])}条记录")
            print(f"  - 平均市值: {avg_market_cap:.2f}")
            print(f"  - 平均周收益率: {avg_return*100:.4f}%")
    
    # 保存分组后的数据
    os.makedirs('../data', exist_ok=True)
    for group_num, data in result_dict.items():
        save_path = f"../data/group_{group_num}_data.csv"
        data.to_csv(save_path, index=False)
        print(f"第{group_num}组数据已保存到{save_path}")
    
    return result_dict
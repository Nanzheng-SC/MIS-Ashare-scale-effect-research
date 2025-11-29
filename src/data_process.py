import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_and_group_data(stock_data):
    """
    处理股票数据并按市值进行五分位分组，研究A股市场规模效应
    
    Args:
        stock_data: 原始股票数据
    
    Returns:
        dict: 包含五个分组数据的字典
    """
    print("开始处理数据...")
    
    # 确保日期列格式正确
    if 'trade_date' in stock_data.columns:
        if isinstance(stock_data['trade_date'].iloc[0], str):
            stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'], format='%Y%m%d')
    else:
        raise KeyError("数据中缺少'trade_date'列")
    
    # 使用正确的市值字段
    market_cap_field = 'market_cap' if 'market_cap' in stock_data.columns else \
                      'estimated_market_cap' if 'estimated_market_cap' in stock_data.columns else None
    
    if not market_cap_field:
        # 如果没有市值数据，则创建估算值
        print("警告：未找到市值数据，使用估算值")
        if 'close' in stock_data.columns:
            stock_data['market_cap'] = stock_data['close'] * 1000000  # 单位：万元
            market_cap_field = 'market_cap'
        else:
            raise KeyError("数据中缺少计算市值所需的'close'列")
    
    print(f"使用{market_cap_field}字段进行市值分组")
    
    # 计算周收益率和月收益率
    if 'ts_code' in stock_data.columns:
        # 确保数据按股票代码和日期排序
        stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
        
        # 计算周收益率
        stock_data['weekly_return'] = stock_data.groupby('ts_code')['close'].pct_change(periods=5)  # 近似周收益率
        
        # 计算月收益率
        stock_data['monthly_return'] = stock_data.groupby('ts_code')['close'].pct_change(periods=20)  # 近似月收益率
    else:
        raise KeyError("数据中缺少'ts_code'列")
    
    # 去除NaN值
    stock_data = stock_data.dropna(subset=[market_cap_field, 'weekly_return'])
    
    print(f"处理后的数据量: {len(stock_data)}条记录")
    
    # 按时间周期分组进行市值分组
    # 使用月度重分组，这是研究规模效应的常见做法
    result_dict = {}
    
    # 先按月对数据进行分组
    stock_data['year_month'] = stock_data['trade_date'].dt.to_period('M')
    grouped_by_month = stock_data.groupby('year_month')
    
    print(f"共有{len(grouped_by_month)}个月的数据需要处理")
    
    # 存储每个分组的数据
    groups_data = {i: [] for i in range(1, 6)}
    
    # 记录每个月的分组信息
    monthly_group_stats = []
    
    for year_month, monthly_data in grouped_by_month:
        print(f"处理{year_month}的数据...")
        
        # 确保每个月份有足够的数据进行分组
        if len(monthly_data) >= 5:
            try:
                # 按市值进行五分位分组
                # 首先获取每个股票在该月的最新市值
                monthly_stocks = monthly_data.groupby('ts_code').last().reset_index()
                
                # 确保有足够的股票进行分组
                if len(monthly_stocks) >= 5:
                    # 使用qcut进行五分位分组，处理重复值问题
                    try:
                        # 尝试使用qcut
                        monthly_stocks['market_cap_group'] = pd.qcut(
                            monthly_stocks[market_cap_field], 
                            5, 
                            labels=[1, 2, 3, 4, 5],
                            duplicates='drop'  # 处理重复值
                        )
                    except ValueError:
                        # 如果qcut失败（通常是因为重复值太多），则使用rank方法
                        print(f"{year_month}使用rank方法进行分组（qcut失败）")
                        monthly_stocks['market_cap_rank'] = monthly_stocks[market_cap_field].rank(method='first')
                        total_stocks = len(monthly_stocks)
                        monthly_stocks['market_cap_group'] = pd.cut(
                            monthly_stocks['market_cap_rank'],
                            bins=[0, total_stocks*0.2, total_stocks*0.4, total_stocks*0.6, total_stocks*0.8, total_stocks+1],
                            labels=[1, 2, 3, 4, 5]
                        )
                    
                    # 记录当月分组统计信息
                    month_stats = {
                        'year_month': str(year_month),
                        'total_stocks': len(monthly_stocks)
                    }
                    
                    # 将分组信息合并回原始数据
                    monthly_data_with_groups = monthly_data.merge(
                        monthly_stocks[['ts_code', 'market_cap_group']],
                        on='ts_code',
                        how='left'
                    )
                    
                    # 保存每个分组的数据
                    for group_num in range(1, 6):
                        group_data = monthly_data_with_groups[monthly_data_with_groups['market_cap_group'] == group_num]
                        if not group_data.empty:
                            groups_data[group_num].append(group_data)
                            # 记录分组统计信息
                            avg_cap = group_data[market_cap_field].mean()
                            avg_return = group_data['weekly_return'].mean()
                            month_stats[f'group_{group_num}_avg_cap'] = avg_cap
                            month_stats[f'group_{group_num}_avg_return'] = avg_return
                    
                    monthly_group_stats.append(month_stats)
                else:
                    print(f"{year_month}的数据不足，仅{len(monthly_stocks)}只股票")
            except Exception as e:
                print(f"处理{year_month}数据时出错: {e}")
                continue
        else:
            print(f"{year_month}的数据不足，仅{len(monthly_data)}条记录")
    
    # 合并每个分组的数据
    for group_num in range(1, 6):
        if groups_data[group_num]:
            result_dict[group_num] = pd.concat(groups_data[group_num], ignore_index=True)
            # 计算该组的统计信息
            avg_market_cap = result_dict[group_num][market_cap_field].mean()
            avg_weekly_return = result_dict[group_num]['weekly_return'].mean()
            avg_monthly_return = result_dict[group_num]['monthly_return'].mean() if 'monthly_return' in result_dict[group_num].columns else np.nan
            
            group_type = "小市值" if group_num == 1 else "大市值" if group_num == 5 else "中等市值"
            print(f"\n第{group_num}组（{group_type}）统计信息:")
            print(f"  - 记录数: {len(result_dict[group_num])}条")
            print(f"  - 平均市值: {avg_market_cap:.2f}万元")
            print(f"  - 平均周收益率: {avg_weekly_return*100:.4f}%")
            if not np.isnan(avg_monthly_return):
                print(f"  - 平均月收益率: {avg_monthly_return*100:.4f}%")
    
    # 保存分组后的数据
    os.makedirs('../data', exist_ok=True)
    for group_num, data in result_dict.items():
        save_path = f"../data/group_{group_num}_data.csv"
        data.to_csv(save_path, index=False)
        print(f"\n第{group_num}组数据已保存到{save_path}")
    
    # 保存月度分组统计信息
    if monthly_group_stats:
        stats_df = pd.DataFrame(monthly_group_stats)
        stats_save_path = "../data/monthly_group_stats.csv"
        stats_df.to_csv(stats_save_path, index=False)
        print(f"月度分组统计信息已保存到{stats_save_path}")
    
    # 计算整体规模效应指标
    if 1 in result_dict and 5 in result_dict:
        small_cap_return = result_dict[1]['weekly_return'].mean()
        large_cap_return = result_dict[5]['weekly_return'].mean()
        size_premium = small_cap_return - large_cap_return
        
        print(f"\n规模效应分析结果:")
        print(f"小市值组（第1组）平均周收益率: {small_cap_return*100:.4f}%")
        print(f"大市值组（第5组）平均周收益率: {large_cap_return*100:.4f}%")
        print(f"小市值-大市值超额收益: {size_premium*100:.4f}%")
        print(f"年化超额收益: {size_premium*52*100:.4f}%")
    
    return result_dict
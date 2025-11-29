import os
import tushare as ts
import pandas as pd
import time
from datetime import datetime

def fetch_stock_data(start_date, end_date):
    """
    从tushare获取A股市场所有股票的历史数据
    
    Args:
        start_date: 开始日期，格式为'YYYYMMDD'
        end_date: 结束日期，格式为'YYYYMMDD'
    
    Returns:
        pd.DataFrame: 包含股票数据的DataFrame
    """
    # 检查数据是否已经存在
    save_path = f"../data/all_stock_data_{start_date}_{end_date}.csv"
    if os.path.exists(save_path):
        print(f"数据文件已存在，直接读取: {save_path}")
        return pd.read_csv(save_path)
    
    # 初始化tushare
    token = os.getenv('toshare_token')
    if not token:
        raise ValueError("未找到tushare token，请检查.env文件")
    
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 获取A股所有股票的基本信息
    print("正在获取A股股票列表...")
    stock_basic = pro.stock_basic(
        exchange='', 
        list_status='L',  # 上市
        fields='ts_code,symbol,name,area,industry,market,list_date'
    )
    
    # 为了演示，我们先获取一部分股票数据（前200只）
    # 实际使用时可以注释掉这行获取所有股票
    stock_basic = stock_basic.head(200)
    
    all_data = []
    total_stocks = len(stock_basic)
    print(f"共有{total_stocks}只股票需要获取数据")
    
    # 由于API限制，分批获取数据
    batch_size = 50
    for batch_start in range(0, total_stocks, batch_size):
        batch_end = min(batch_start + batch_size, total_stocks)
        print(f"正在处理批次 {batch_start//batch_size + 1}/{(total_stocks + batch_size - 1)//batch_size}")
        
        for i in range(batch_start, batch_end):
            stock = stock_basic.iloc[i]
            ts_code = stock['ts_code']
            
            if i % 50 == 0:
                print(f"已处理{i}/{total_stocks}只股票")
            
            try:
                # 获取日线数据
                df = pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not df.empty:
                    # 合并基本信息
                    stock_info = stock_basic[stock_basic['ts_code'] == ts_code].iloc[0].to_dict()
                    for key, value in stock_info.items():
                        df[key] = value
                    
                    # 尝试获取市值数据（实际市值数据可能需要从其他API获取）
                    # 这里我们使用一个代理变量
                    df['estimated_market_cap'] = df['close'] * 1000000  # 简化计算
                    
                    all_data.append(df)
                
                # 避免请求过快
                time.sleep(0.1)
            except Exception as e:
                print(f"获取股票{ts_code}数据时出错: {e}")
                continue
        
        # 每批次结束后休息一下
        if batch_end < total_stocks:
            print("休息10秒，避免触发API限制...")
            time.sleep(10)
    
    if not all_data:
        raise ValueError("未能获取到任何股票数据")
    
    # 合并所有数据
    result = pd.concat(all_data, ignore_index=True)
    
    # 保存原始数据
    os.makedirs('../data', exist_ok=True)
    result.to_csv(save_path, index=False)
    print(f"数据已保存到{save_path}")
    
    return result
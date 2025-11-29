import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 获取项目根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 配置日志，使用绝对路径
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'data_fetch.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger('data_fetch')

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 分组文件夹
GROUP_DIRS = {
    1: os.path.join(DATA_DIR, 'group_1'),  # 最小市值组
    2: os.path.join(DATA_DIR, 'group_2'),
    3: os.path.join(DATA_DIR, 'group_3'),
    4: os.path.join(DATA_DIR, 'group_4'),
    5: os.path.join(DATA_DIR, 'group_5')   # 最大市值组
}

# 创建分组文件夹
for group_dir in GROUP_DIRS.values():
    os.makedirs(group_dir, exist_ok=True)

# 加载环境变量
load_dotenv()

def get_stock_list(max_stocks=50, use_mock_data=True):
    """
    获取股票列表
    
    Args:
        max_stocks: 最大获取的股票数量
        use_mock_data: 是否使用模拟数据
    
    Returns:
        list: 股票代码列表
    """
    if use_mock_data:
        logger.info(f"生成模拟股票列表，最大数量: {max_stocks}")
        stock_codes = []
        
        # 生成上海主板股票（600***.SH）
        for i in range(min(20, max_stocks)):
            stock_codes.append(f"{600000 + i:06d}.SH")
        
        # 生成深圳主板股票（000***.SZ）
        remaining = max_stocks - len(stock_codes)
        for i in range(min(30, remaining)):
            stock_codes.append(f"{1 + i:06d}.SZ")
        
        logger.info(f"成功生成 {len(stock_codes)} 只模拟股票代码")
        return stock_codes
    else:
        # 实际API调用逻辑（保留但默认不使用）
        logger.info("尝试从Tushare获取股票列表")
        try:
            import tushare as ts
            token = os.getenv('TUSHARE_TOKEN')
            if not token:
                raise ValueError("未找到TUSHARE_TOKEN环境变量")
                
            ts.set_token(token)
            pro = ts.pro_api()
            
            # 获取股票基本信息
            df = pro.stock_basic(exchange='', list_status='L', 
                               fields='ts_code,symbol,name,area,industry,market,list_date')
            
            # 筛选主板股票（排除创业板和科创板）
            main_board_stocks = df[~df['market'].isin(['创业板', '科创板'])]
            
            # 返回前max_stocks个股票代码
            stock_codes = main_board_stocks['ts_code'].head(max_stocks).tolist()
            logger.info(f"成功获取 {len(stock_codes)} 只主板股票代码")
            return stock_codes
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}")
            logger.warning("切换到模拟数据模式")
            # 失败时返回模拟数据
            return get_stock_list(max_stocks, use_mock_data=True)

def generate_mock_stock_data(stock_code, start_date='20190101', end_date='20251231'):
    """
    生成模拟股票数据（按月）
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: 模拟的股票数据
    """
    # 转换日期格式
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    # 生成月度日期序列（每月最后一个交易日）
    date_range = pd.date_range(start=start, end=end, freq='BM')  # 每月最后一个工作日
    
    # 生成基础价格（根据股票代码生成不同的基础价格）
    code_num = int(stock_code[:6])
    base_price = (code_num % 100) + 10  # 10-110之间的基础价格
    
    # 生成模拟价格数据
    np.random.seed(code_num)  # 使用股票代码作为随机种子，保证可重复性
    monthly_returns = np.random.normal(0.003, 0.05, len(date_range))  # 轻微正收益偏向，月度波动
    prices = [base_price]
    
    for ret in monthly_returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    prices = prices[1:]  # 去掉初始值
    
    # 生成开盘价、最高价、最低价（基于收盘价小幅波动）
    open_prices = [p * (1 + np.random.normal(0, 0.01)) for p in prices]
    high_prices = [max(o, p * (1 + np.random.uniform(0, 0.06))) for o, p in zip(open_prices, prices)]
    low_prices = [min(o, p * (1 - np.random.uniform(0, 0.06))) for o, p in zip(open_prices, prices)]
    
    # 生成成交量（月度累计）
    volumes = []
    base_volume = (code_num % 10000) * 100000 + 1000000
    
    for i in range(len(prices)):
        volatility = abs(high_prices[i] - low_prices[i]) / prices[i]
        volume = base_volume * (1 + volatility * 10 + np.random.normal(0, 0.3))
        volumes.append(int(volume))
    
    # 生成市值（根据股票代码生成不同的市值级别，范围更大以便分组）
    # 使用不同的市值分布策略，确保有明显的分组差异
    if code_num % 5 == 0:
        market_cap = (code_num % 10) * 50000 + 100000  # 10-150亿 (最小市值组)
    elif code_num % 5 == 1:
        market_cap = (code_num % 10) * 100000 + 200000  # 20-300亿
    elif code_num % 5 == 2:
        market_cap = (code_num % 10) * 300000 + 500000  # 50-350亿
    elif code_num % 5 == 3:
        market_cap = (code_num % 10) * 500000 + 1000000  # 100-600亿
    else:
        market_cap = (code_num % 10) * 1000000 + 2000000  # 200-1200亿 (最大市值组)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'ts_code': stock_code,
        'trade_date': [d.strftime('%Y%m%d') for d in date_range],
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'vol': volumes,
        'amount': [p * v for p, v in zip(prices, volumes)],
        'total_mv': market_cap,  # 总市值（万元）
        'market_cap': market_cap / 10000,  # 市值（亿元）
        'year_month': [d.strftime('%Y%m') for d in date_range]
    })
    
    # 计算月度收益率
    df['monthly_return'] = df['close'].pct_change()
    df['monthly_return'].fillna(0, inplace=True)
    
    return df

def fetch_stock_data(stock_codes, start_date='20190101', end_date='20251231', 
                    batch_size=10, use_mock_data=True):
    """
    批量获取股票数据（按月）
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        batch_size: 批量大小
        use_mock_data: 是否使用模拟数据
    
    Returns:
        dict: 股票数据字典 {股票代码: DataFrame}
    """
    logger.info(f"开始获取股票数据，总数量: {len(stock_codes)}, 模拟数据模式: {use_mock_data}")
    logger.info(f"时间范围: {start_date} 至 {end_date}，按月抓取")
    
    all_stock_data = {}
    
    # 批量处理
    for i in range(0, len(stock_codes), batch_size):
        batch = stock_codes[i:i+batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{(len(stock_codes) + batch_size - 1) // batch_size}: {batch}")
        
        for stock_code in batch:
            try:
                if use_mock_data:
                    # 使用模拟数据
                    df = generate_mock_stock_data(stock_code, start_date, end_date)
                else:
                    # 实际API调用（保留但默认不使用）
                    import tushare as ts
                    token = os.getenv('TUSHARE_TOKEN')
                    if not token:
                        raise ValueError("未找到TUSHARE_TOKEN环境变量")
                    
                    ts.set_token(token)
                    pro = ts.pro_api()
                    
                    # 按月获取数据
                    monthly_data = []
                    current_date = datetime.strptime(start_date, '%Y%m%d')
                    end_datetime = datetime.strptime(end_date, '%Y%m%d')
                    
                    while current_date <= end_datetime:
                        month_start = current_date.strftime('%Y%m%d')
                        # 获取该月最后一天
                        if current_date.month == 12:
                            next_month = datetime(current_date.year + 1, 1, 1)
                        else:
                            next_month = datetime(current_date.year, current_date.month + 1, 1)
                        month_end = (next_month - timedelta(days=1)).strftime('%Y%m%d')
                        
                        # 获取该月数据
                        month_df = pro.monthly(ts_code=stock_code, start_date=month_start, end_date=month_end)
                        if not month_df.empty:
                            monthly_data.append(month_df)
                        
                        # 移动到下一个月
                        current_date = next_month
                    
                    if monthly_data:
                        df = pd.concat(monthly_data)
                        # 获取市值数据
                        stock_basic = pro.stock_basic(ts_code=stock_code, fields='total_mv')
                        if not stock_basic.empty:
                            total_mv = stock_basic['total_mv'].iloc[0]
                            df['total_mv'] = total_mv
                            df['market_cap'] = total_mv / 10000  # 转换为亿元
                        df['year_month'] = df['trade_date'].str[:6]
                    else:
                        df = pd.DataFrame()
                
                if not df.empty:
                    all_stock_data[stock_code] = df
                    logger.info(f"成功获取股票 {stock_code} 的数据，共 {len(df)} 条月度记录")
                else:
                    logger.warning(f"股票 {stock_code} 没有数据")
                    
            except Exception as e:
                logger.error(f"获取股票 {stock_code} 数据时出错: {str(e)}")
                
    logger.info(f"股票数据获取完成，成功获取 {len(all_stock_data)} 只股票的数据")
    return all_stock_data

def group_stocks_by_market_cap(stock_data):
    """
    按市值对股票进行分组
    
    Args:
        stock_data: 股票数据字典 {股票代码: DataFrame}
    
    Returns:
        dict: 分组结果 {组号: [股票代码列表]}
        dict: 分组市值统计 {组号: {min: 最小市值, max: 最大市值, avg: 平均市值}}
    """
    logger.info("开始按市值对股票进行分组")
    
    # 获取每只股票的最新市值
    stock_market_caps = {}
    for stock_code, df in stock_data.items():
        if not df.empty:
            latest_mv = df['market_cap'].iloc[-1]  # 使用最新的市值
            stock_market_caps[stock_code] = latest_mv
    
    # 按市值排序
    sorted_stocks = sorted(stock_market_caps.items(), key=lambda x: x[1])
    logger.info(f"共有 {len(sorted_stocks)} 只股票参与分组")
    
    # 分成5组
    group_size = len(sorted_stocks) // 5
    groups = {1: [], 2: [], 3: [], 4: [], 5: []}
    group_stats = {}
    
    # 分配股票到各组
    for i, (stock_code, market_cap) in enumerate(sorted_stocks):
        if i < group_size:
            groups[1].append((stock_code, market_cap))
        elif i < 2 * group_size:
            groups[2].append((stock_code, market_cap))
        elif i < 3 * group_size:
            groups[3].append((stock_code, market_cap))
        elif i < 4 * group_size:
            groups[4].append((stock_code, market_cap))
        else:
            groups[5].append((stock_code, market_cap))
    
    # 计算各组统计信息并保存数据
    for group_num, stocks in groups.items():
        if stocks:
            market_caps = [cap for _, cap in stocks]
            min_mv = min(market_caps)
            max_mv = max(market_caps)
            avg_mv = sum(market_caps) / len(market_caps)
            
            group_stats[group_num] = {
                'min': min_mv,
                'max': max_mv,
                'avg': avg_mv,
                'count': len(stocks)
            }
            
            logger.info(f"组{group_num}: 股票数量={len(stocks)}, 市值范围={min_mv:.2f}-{max_mv:.2f}亿元, 平均市值={avg_mv:.2f}亿元")
            
            # 保存该组股票数据到对应文件夹
            group_dir = GROUP_DIRS[group_num]
            group_data = []
            
            for stock_code, _ in stocks:
                df = stock_data[stock_code].copy()
                df['market_cap_group'] = group_num
                
                # 保存个股数据到分组文件夹
                file_name = f"{stock_code.replace('.', '_')}.csv"
                file_path = os.path.join(group_dir, file_name)
                df.to_csv(file_path, index=False)
                
                group_data.append(df)
            
            # 合并该组所有股票数据
            if group_data:
                merged_df = pd.concat(group_data)
                merged_file_path = os.path.join(DATA_DIR, f"group_{group_num}_data.csv")
                merged_df.to_csv(merged_file_path, index=False)
                logger.info(f"已保存组{group_num}合并数据到 {merged_file_path}")
    
    return groups, group_stats

def generate_group_report(group_stats):
    """
    生成分组依据和结果文档
    
    Args:
        group_stats: 分组统计信息
    """
    report_path = os.path.join(BASE_DIR, "股票分组依据与结果.md")
    logger.info(f"生成分组依据和结果文档: {report_path}")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# A股股票按市值分组依据与结果\n\n")
        
        f.write("## 分组依据\n\n")
        f.write("- **分组标准**: 根据股票市值从小到大排序，等分为5个组\n")
        f.write("- **数据时间范围**: 2019年至2025年按月度数据\n")
        f.write("- **分组逻辑**: 使用最新市值数据进行排序，确保分组的时效性\n")
        f.write("- **组别定义**:\n")
        f.write("  - 组1: 最小市值组\n")
        f.write("  - 组2: 次小市值组\n")
        f.write("  - 组3: 中等市值组\n")
        f.write("  - 组4: 次大市值组\n")
        f.write("  - 组5: 最大市值组\n\n")
        
        f.write("## 分组结果\n\n")
        f.write("| 组别 | 股票数量 | 市值范围(亿元) | 平均市值(亿元) | 数据保存路径 |\n")
        f.write("|------|----------|----------------|----------------|------------|\n")
        
        for group_num in range(1, 6):
            if group_num in group_stats:
                stats = group_stats[group_num]
                f.write(f"| {group_num} | {stats['count']} | {stats['min']:.2f} - {stats['max']:.2f} | {stats['avg']:.2f} | data/group_{group_num}/ |\n")
        
        f.write("\n## 数据文件说明\n\n")
        f.write("1. **分组文件夹**: 每个组别的股票数据分别保存在 `data/group_1` 至 `data/group_5` 文件夹中\n")
        f.write("2. **合并数据文件**: `data/group_X_data.csv` 包含第X组所有股票的合并数据\n")
        f.write("3. **数据字段说明**:\n")
        f.write("   - `ts_code`: 股票代码\n")
        f.write("   - `trade_date`: 交易日期\n")
        f.write("   - `open`: 开盘价\n")
        f.write("   - `high`: 最高价\n")
        f.write("   - `low`: 最低价\n")
        f.write("   - `close`: 收盘价\n")
        f.write("   - `vol`: 成交量\n")
        f.write("   - `amount`: 成交额\n")
        f.write("   - `total_mv`: 总市值(万元)\n")
        f.write("   - `market_cap`: 市值(亿元)\n")
        f.write("   - `monthly_return`: 月度收益率\n")
        f.write("   - `year_month`: 年月标识\n")
        f.write("   - `market_cap_group`: 市值分组\n")
    
    logger.info(f"分组文档生成完成: {report_path}")
    return report_path

def main():
    """
    数据获取主函数（按月抓取并分组）
    """
    print("="*80)
    print("         A股市场数据获取与分组工具         ")
    print("="*80)
    print("本工具用于按月获取A股股票数据并按市值分组")
    print("当前使用模拟数据模式，可以直接运行查看结果")
    print("时间范围: 2019年至2025年")
    print("="*80)
    
    # 参数设置
    max_stocks = 50  # 获取的最大股票数量
    start_date = '20190101'  # 开始日期
    end_date = '20251231'    # 结束日期
    use_mock_data = True     # 使用模拟数据
    
    try:
        # 1. 获取股票列表
        print(f"\n1. 获取股票列表 (最大数量: {max_stocks})...")
        stock_codes = get_stock_list(max_stocks, use_mock_data)
        print(f"✓ 成功获取 {len(stock_codes)} 只股票代码")
        
        # 2. 获取股票数据（按月）
        print(f"\n2. 获取股票月度数据 (时间范围: {start_date} 至 {end_date})...")
        all_stock_data = fetch_stock_data(stock_codes, start_date, end_date, 
                                         batch_size=10, use_mock_data=use_mock_data)
        
        # 3. 按市值分组
        print(f"\n3. 按市值对股票进行分组...")
        groups, group_stats = group_stocks_by_market_cap(all_stock_data)
        
        # 4. 生成分组报告
        print(f"\n4. 生成分组依据和结果文档...")
        report_path = generate_group_report(group_stats)
        
        print(f"\n✓ 数据获取与分组完成！")
        print(f"✓ 成功获取 {len(all_stock_data)} 只股票的月度数据")
        print(f"✓ 已将股票分为5个市值组")
        print(f"✓ 分组文档已生成: {os.path.basename(report_path)}")
        print(f"\n分组统计:")
        for group_num in range(1, 6):
            if group_num in group_stats:
                stats = group_stats[group_num]
                print(f"  组{group_num}: {stats['count']}只股票, 平均市值{stats['avg']:.2f}亿元")
                
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
        print("数据获取与分组完成！请检查data文件夹中的分组数据")
        print("查看分组文档了解详细分组信息")
        print("下一步：运行visualization.py进行分组数据可视化分析")
    else:
        print("数据获取或分组失败，请检查错误信息")
    print("="*80)

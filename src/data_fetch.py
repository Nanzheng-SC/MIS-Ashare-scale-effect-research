import tushare as ts
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(start_date="20190101", end_date="20250831", max_stocks=300, batch_size=50, retry_times=3):
    """
    从tushare获取A股市场股票数据
    
    Args:
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
        max_stocks: 最大获取的股票数量
        batch_size: 批处理大小
        retry_times: API调用失败重试次数
        
    Returns:
        DataFrame: 包含股票数据的DataFrame
    """
    try:
        # 获取tushare token
        token = os.getenv('toshare_token')
        if not token:
            logger.error("未找到tushare token，请在.env文件中设置")
            raise ValueError("未找到tushare token")
        
        # 初始化pro接口
        pro = ts.pro_api(token)
        logger.info(f"已初始化tushare API，准备获取{start_date}至{end_date}的股票数据")
        
        # 创建数据目录
        os.makedirs('../data', exist_ok=True)
        
        # 1. 获取股票列表
        logger.info("正在获取股票列表...")
        stock_basic = None
        for retry in range(retry_times):
            try:
                stock_basic = pro.stock_basic(
                    exchange='',
                    list_status='L',  # L: 上市
                    fields='ts_code,symbol,name,area,industry,market,list_date'
                )
                break
            except Exception as e:
                logger.warning(f"获取股票列表失败，第{retry+1}次重试，错误: {str(e)}")
                time.sleep(2)
        
        if stock_basic is None or stock_basic.empty:
            logger.error("获取股票列表失败")
            return pd.DataFrame()
        
        # 筛选出A股股票（上海和深圳市场）
        a_share_stocks = stock_basic[stock_basic['market'].isin(['主板', '创业板', '科创板', '中小板'])]
        logger.info(f"共获取到{len(a_share_stocks)}只A股股票")
        
        # 限制最大股票数量
        if len(a_share_stocks) > max_stocks:
            a_share_stocks = a_share_stocks.head(max_stocks)
            logger.info(f"已限制为获取前{max_stocks}只股票")
        
        # 2. 分批次获取股票数据
        all_stock_data = []
        total_stocks = len(a_share_stocks)
        success_count = 0
        fail_count = 0
        failed_stocks = []
        
        # 记录已处理的股票
        processed_stocks = set()
        
        # 分批次处理
        for i in range(0, total_stocks, batch_size):
            batch = a_share_stocks.iloc[i:i+batch_size]
            batch_size_actual = len(batch)
            logger.info(f"正在处理批次 {i//batch_size + 1}/{(total_stocks + batch_size - 1) // batch_size}，股票数量: {batch_size_actual}")
            
            for idx, stock in batch.iterrows():
                ts_code = stock['ts_code']
                stock_name = stock['name']
                
                # 跳过已处理的股票
                if ts_code in processed_stocks:
                    continue
                
                logger.info(f"[{idx+1}/{total_stocks}] 正在获取股票: {ts_code} {stock_name}")
                
                # 获取股票日线数据
                stock_daily_data = None
                for retry in range(retry_times):
                    try:
                        stock_daily_data = pro.daily(
                            ts_code=ts_code,
                            start_date=start_date,
                            end_date=end_date,
                            fields='ts_code,trade_date,open,high,low,close,vol,amount'
                        )
                        break
                    except Exception as e:
                        logger.warning(f"获取{ts_code}日线数据失败，第{retry+1}次重试，错误: {str(e)}")
                        time.sleep(2)
                
                if stock_daily_data is None or stock_daily_data.empty:
                    logger.error(f"获取{ts_code}日线数据失败")
                    fail_count += 1
                    failed_stocks.append(ts_code)
                    continue
                
                # 获取市值数据（如果有）
                mkt_cap_data = None
                for retry in range(retry_times):
                    try:
                        # 尝试获取市值数据
                        mkt_cap_data = pro.daily_basic(
                            ts_code=ts_code,
                            start_date=start_date,
                            end_date=end_date,
                            fields='ts_code,trade_date,total_mv,circ_mv'  # 总市值和流通市值
                        )
                        break
                    except Exception as e:
                        logger.warning(f"获取{ts_code}市值数据失败，使用收盘价估算，错误: {str(e)}")
                        # 如果获取市值失败，就不重试了
                        break
                
                # 合并数据
                if mkt_cap_data is not None and not mkt_cap_data.empty:
                    # 合并日线数据和市值数据
                    merged_data = pd.merge(stock_daily_data, mkt_cap_data, on=['ts_code', 'trade_date'], how='left')
                else:
                    # 如果没有市值数据，复制close列作为估算的总市值和流通市值
                    # 注意：这只是一个估算，实际市值计算需要考虑股本
                    merged_data = stock_daily_data.copy()
                    merged_data['total_mv'] = merged_data['close'] * 1000000  # 简单估算
                    merged_data['circ_mv'] = merged_data['total_mv'] * 0.8  # 假设80%流通
                    logger.warning(f"{ts_code} 未获取到市值数据，使用收盘价进行估算")
                
                # 添加股票基本信息
                merged_data['name'] = stock_name
                merged_data['industry'] = stock['industry']
                merged_data['area'] = stock['area']
                
                # 添加到结果
                all_stock_data.append(merged_data)
                processed_stocks.add(ts_code)
                success_count += 1
                
                # 避免请求过于频繁
                time.sleep(0.1)
            
            # 批次之间暂停一下
            if i + batch_size < total_stocks:
                logger.info("批次处理完成，暂停5秒...")
                time.sleep(5)
        
        # 3. 合并所有数据
        if all_stock_data:
            final_data = pd.concat(all_stock_data, ignore_index=True)
            
            # 转换日期格式
            final_data['trade_date'] = pd.to_datetime(final_data['trade_date'], format='%Y%m%d')
            
            # 保存数据到CSV
            csv_file = f"../data/a_share_data_{start_date}_{end_date}.csv"
            final_data.to_csv(csv_file, index=False, encoding='utf-8-sig')
            logger.info(f"已保存股票数据到 {csv_file}")
            logger.info(f"获取成功: {success_count} 只股票, 获取失败: {fail_count} 只股票")
            
            if failed_stocks:
                logger.warning(f"失败的股票列表: {failed_stocks}")
            
            return final_data
        else:
            logger.error("未获取到任何股票数据")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"获取股票数据时发生错误: {str(e)}")
        raise

def get_stock_list():
    """
    获取A股股票列表
    
    Returns:
        DataFrame: 包含股票基本信息的DataFrame
    """
    try:
        token = os.getenv('toshare_token')
        if not token:
            raise ValueError("未找到tushare token")
        
        pro = ts.pro_api(token)
        stock_basic = pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,market,list_date'
        )
        
        return stock_basic[stock_basic['market'].isin(['主板', '创业板', '科创板', '中小板'])]
        
    except Exception as e:
        logger.error(f"获取股票列表时发生错误: {str(e)}")
        raise
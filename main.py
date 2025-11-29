import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetch import fetch_stock_data
from src.data_process import process_and_group_data
from src.visualization import create_visualizations

if __name__ == "__main__":
    print("开始A股市场规模效应研究...")
    
    # 设置时间范围
    start_date = "20190101"
    end_date = "20250831"
    
    # 1. 获取数据
    print("正在获取A股市场数据...")
    stock_data = fetch_stock_data(start_date, end_date)
    
    # 2. 处理数据并按市值分组
    print("正在处理数据并按市值分组...")
    grouped_data = process_and_group_data(stock_data)
    
    # 3. 创建可视化
    print("正在创建可视化图表...")
    create_visualizations(grouped_data)
    
    print("研究完成！结果已保存到results文件夹。")
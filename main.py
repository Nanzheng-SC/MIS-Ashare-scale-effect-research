import os
import sys
import time
import datetime
from dotenv import load_dotenv

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入自定义模块
from data_fetch import fetch_stock_data
from data_process import process_and_group_data
from visualization import visualize_scale_effect


def main():
    """
    主函数：执行A股市场规模效应研究的完整流程
    """
    print("=" * 80)
    print("A股市场规模效应研究系统")
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # 1. 加载环境变量
        print("\n[1/4] 加载环境配置...")
        load_dotenv()
        print("✓ 环境变量加载完成")
        
        # 2. 配置参数
        print("\n[2/4] 设置研究参数...")
        # 时间范围
        start_date = "20190101"  # 起始日期
        end_date = "20250831"    # 结束日期
        
        # 数据处理参数
        max_stocks = 300         # 处理的最大股票数量
        batch_size = 50          # 批处理大小
        retry_times = 3          # API调用重试次数
        
        print(f"✓ 研究参数设置完成:")
        print(f"  • 时间范围: {start_date} 至 {end_date}")
        print(f"  • 最大股票数量: {max_stocks}")
        print(f"  • 批处理大小: {batch_size}")
        print(f"  • 重试次数: {retry_times}")
        
        # 3. 获取数据
        print("\n[3/4] 开始获取股票数据...")
        start_time = time.time()
        stock_data = fetch_stock_data(
            start_date=start_date,
            end_date=end_date,
            max_stocks=max_stocks,
            batch_size=batch_size,
            retry_times=retry_times
        )
        fetch_duration = time.time() - start_time
        
        if stock_data is None or stock_data.empty:
            print("✗ 数据获取失败或没有数据")
            return
            
        print(f"✓ 数据获取完成，耗时: {fetch_duration:.2f}秒")
        print(f"  • 获取的股票数量: {stock_data['ts_code'].nunique()}")
        print(f"  • 数据记录条数: {len(stock_data)}")
        
        # 4. 数据处理和分组
        print("\n[4/4] 开始数据处理和市值分组...")
        start_time = time.time()
        grouped_data = process_and_group_data(stock_data)
        process_duration = time.time() - start_time
        
        if not grouped_data or all(df.empty for df in grouped_data.values()):
            print("✗ 数据处理失败或分组后没有数据")
            return
            
        print(f"✓ 数据处理完成，耗时: {process_duration:.2f}秒")
        print(f"  • 成功分组的数量: {sum(1 for df in grouped_data.values() if not df.empty)}")
        for group_num, data in sorted(grouped_data.items()):
            if not data.empty:
                print(f"  • 第{group_num}组（{'小' if group_num == 1 else '大' if group_num == 5 else '中'}市值）: {len(data)}条记录")
        
        # 5. 创建可视化
        print("\n[5/5] 开始创建可视化图表...")
        start_time = time.time()
        visualize_scale_effect(grouped_data)
        viz_duration = time.time() - start_time
        print(f"✓ 可视化完成，耗时: {viz_duration:.2f}秒")
        
        # 6. 总结
        print("\n" + "=" * 80)
        print(f"研究流程执行完成!")
        print(f"总耗时: {fetch_duration + process_duration + viz_duration:.2f}秒")
        print(f"结果保存在: {os.path.join(os.path.dirname(__file__), 'results')}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n✗ 程序执行出错: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
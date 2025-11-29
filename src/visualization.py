import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ScaleEffectVisualizer:
    """A股市场规模效应可视化类"""
    
    def __init__(self):
        # 创建结果目录
        self.results_dir = '../results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置图表风格
        sns.set(style="whitegrid")
        sns.set_palette("viridis", 5)  # 使用色彩鲜明的调色板
    
    def create_all_visualizations(self, grouped_data):
        """创建所有可视化图表"""
        print("开始创建可视化图表...")
        
        # 确保数据有效
        if not grouped_data or any(not group_data.empty for group_num, group_data in grouped_data.items()):
            # 创建收益率分布箱线图
            self.create_return_distribution_boxplot(grouped_data)
            
            # 创建累计收益率折线图
            self.create_cumulative_returns_plot(grouped_data)
            
            # 创建平均收益率柱状图
            self.create_avg_returns_barplot(grouped_data)
            
            # 创建收益率热力图
            self.create_returns_heatmap(grouped_data)
            
            # 创建规模效应时间序列图
            self.create_scale_effect_time_series(grouped_data)
            
            # 创建新增的图表类型
            self.create_annualized_returns_comparison(grouped_data)
            self.create_risk_adjusted_returns(grouped_data)
            
            print("所有可视化图表已创建完成！")
        else:
            print("警告：数据为空，无法创建可视化图表")
    
    def create_return_distribution_boxplot(self, grouped_data):
        """创建五组收益率分布的箱线图"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            boxplot_data = []
            labels = []
            
            for i in range(1, 6):
                if i in grouped_data and not grouped_data[i].empty and 'weekly_return' in grouped_data[i].columns:
                    # 过滤异常值
                    data = grouped_data[i]['weekly_return']
                    # 使用IQR方法过滤异常值
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    data_filtered = data[(data >= Q1 - 1.5*IQR) & (data <= Q3 + 1.5*IQR)]
                    boxplot_data.append(data_filtered)
                    labels.append(f'{i}组({"小" if i==1 else "大" if i==5 else "中"}市值)')
            
            if boxplot_data:
                # 创建箱线图
                box = plt.boxplot(boxplot_data, labels=labels, patch_artist=True, 
                                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                                 whiskerprops=dict(color='black'),
                                 capprops=dict(color='black'),
                                 medianprops=dict(color='red', linewidth=2))
                
                # 为每个箱子设置不同的颜色
                colors = sns.color_palette("viridis", len(boxplot_data))
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                
                plt.title('各市值分组周收益率分布箱线图', fontsize=16, fontweight='bold')
                plt.ylabel('周收益率', fontsize=12)
                plt.xlabel('市值分组', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 添加均值标记
                for i, data in enumerate(boxplot_data):
                    plt.scatter(i+1, data.mean(), color='black', marker='*', s=100, zorder=3)
                    plt.text(i+1.1, data.mean(), f'均值: {data.mean()*100:.2f}%', 
                             verticalalignment='center', fontsize=9)
                
                plt.tight_layout()
                save_path = os.path.join(self.results_dir, '收益率分布箱线图.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"收益率分布箱线图已保存到: {save_path}")
            else:
                print("警告：没有足够的数据创建收益率分布箱线图")
        except Exception as e:
            print(f"创建收益率分布箱线图时出错: {e}")
    
    def create_cumulative_returns_plot(self, grouped_data):
        """创建累计收益率折线图"""
        try:
            plt.figure(figsize=(14, 8))
            
            # 确保数据按日期排序并计算累计收益
            for i in range(1, 6):
                if i in grouped_data and not grouped_data[i].empty:
                    data = grouped_data[i].copy()
                    # 按日期分组计算平均周收益
                    if 'trade_date' in data.columns and 'weekly_return' in data.columns:
                        # 按日期分组并计算平均收益率
                        date_grouped = data.groupby('trade_date')['weekly_return'].mean().reset_index()
                        date_grouped = date_grouped.sort_values('trade_date')
                        
                        # 计算累计收益率
                        date_grouped['cumulative_return'] = (1 + date_grouped['weekly_return']).cumprod() - 1
                        
                        # 绘制累计收益率曲线
                        group_label = f'{i}组({"小" if i==1 else "大" if i==5 else "中"}市值)'
                        plt.plot(date_grouped['trade_date'], date_grouped['cumulative_return']*100, 
                                linewidth=2, label=group_label, alpha=0.8)
            
            plt.title('各市值分组累计收益率对比图', fontsize=16, fontweight='bold')
            plt.ylabel('累计收益率 (%)', fontsize=12)
            plt.xlabel('日期', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best', fontsize=10)
            
            # 添加辅助线
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, '累计收益率折线图.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"累计收益率折线图已保存到: {save_path}")
        except Exception as e:
            print(f"创建累计收益率折线图时出错: {e}")
    
    def create_avg_returns_barplot(self, grouped_data):
        """创建平均收益率柱状图"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            avg_returns = []
            labels = []
            
            for i in range(1, 6):
                if i in grouped_data and not grouped_data[i].empty and 'weekly_return' in grouped_data[i].columns:
                    avg_return = grouped_data[i]['weekly_return'].mean()
                    avg_returns.append(avg_return * 100)  # 转换为百分比
                    labels.append(f'{i}组({"小" if i==1 else "大" if i==5 else "中"}市值)')
            
            if avg_returns:
                # 创建柱状图
                colors = sns.color_palette("viridis", len(avg_returns))
                bars = plt.bar(labels, avg_returns, color=colors, alpha=0.8)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height, 
                             f'{height:.4f}%', ha='center', va='bottom')
                
                # 计算小市值-大市值差异
                if len(avg_returns) >= 5:
                    small_minus_large = avg_returns[0] - avg_returns[-1]
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.text(len(avg_returns)/2, max(avg_returns)*0.9, 
                             f'小市值-大市值差异: {small_minus_large:.4f}%', 
                             ha='center', fontsize=12, fontweight='bold')
                
                plt.title('各市值分组平均周收益率对比图', fontsize=16, fontweight='bold')
                plt.ylabel('平均周收益率 (%)', fontsize=12)
                plt.xlabel('市值分组', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=0)
                
                plt.tight_layout()
                save_path = os.path.join(self.results_dir, '平均收益率柱状图.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"平均收益率柱状图已保存到: {save_path}")
            else:
                print("警告：没有足够的数据创建平均收益率柱状图")
        except Exception as e:
            print(f"创建平均收益率柱状图时出错: {e}")
    
    def create_returns_heatmap(self, grouped_data):
        """创建收益率热力图，展示不同时间段各分组的表现"""
        try:
            # 准备热力图数据
            heatmap_data = []
            periods = []
            groups = range(1, 6)
            
            # 按月计算各分组的平均收益率
            for i in range(1, 6):
                if i in grouped_data and not grouped_data[i].empty:
                    data = grouped_data[i].copy()
                    if 'trade_date' in data.columns and 'weekly_return' in data.columns:
                        # 按月分组
                        data['year_month'] = data['trade_date'].dt.to_period('M')
                        monthly_returns = data.groupby('year_month')['weekly_return'].mean()
                        heatmap_data.append(monthly_returns)
                        
                        # 更新periods列表
                        if not periods:
                            periods = [str(p) for p in monthly_returns.index]
            
            # 创建DataFrame
            if heatmap_data and len(heatmap_data) == 5:
                df_heatmap = pd.DataFrame(heatmap_data, index=[f'组{i}' for i in groups], columns=periods).T
                
                # 只显示最近12个月的数据（如果有）
                if len(df_heatmap) > 12:
                    df_heatmap = df_heatmap.tail(12)
                
                plt.figure(figsize=(14, 8))
                # 转换为百分比
                df_heatmap_percent = df_heatmap * 100
                
                # 创建热力图
                sns.heatmap(df_heatmap_percent, annot=True, fmt='.2f', cmap='RdYlGn', 
                           center=0, linewidths=0.5, cbar_kws={'label': '周收益率 (%)'})
                
                plt.title('各市值分组收益率热力图 (最近12个月)', fontsize=16, fontweight='bold')
                plt.ylabel('月份', fontsize=12)
                plt.xlabel('市值分组', fontsize=12)
                plt.tight_layout()
                save_path = os.path.join(self.results_dir, '收益率热力图.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"收益率热力图已保存到: {save_path}")
            else:
                print("警告：没有足够的数据创建收益率热力图")
        except Exception as e:
            print(f"创建收益率热力图时出错: {e}")
    
    def create_scale_effect_time_series(self, grouped_data):
        """创建规模效应时间序列图，展示小市值vs大市值的差异"""
        try:
            if 1 in grouped_data and 5 in grouped_data and not grouped_data[1].empty and not grouped_data[5].empty:
                # 获取小市值组和大市值组数据
                small_cap_data = grouped_data[1].copy()
                large_cap_data = grouped_data[5].copy()
                
                if 'trade_date' in small_cap_data.columns and 'weekly_return' in small_cap_data.columns:
                    # 按月计算平均收益率
                    small_cap_data['year_month'] = small_cap_data['trade_date'].dt.to_period('M')
                    large_cap_data['year_month'] = large_cap_data['trade_date'].dt.to_period('M')
                    
                    small_monthly = small_cap_data.groupby('year_month')['weekly_return'].mean()
                    large_monthly = large_cap_data.groupby('year_month')['weekly_return'].mean()
                    
                    # 合并数据
                    combined = pd.DataFrame({
                        '小市值组': small_monthly,
                        '大市值组': large_monthly
                    })
                    
                    # 计算差异
                    combined['规模效应(SML)'] = combined['小市值组'] - combined['大市值组']
                    
                    # 创建图表
                    plt.figure(figsize=(14, 9))
                    
                    # 绘制条形图显示月度差异
                    plt.subplot(2, 1, 1)
                    plt.bar(combined.index.astype(str), combined['规模效应(SML)'] * 100, 
                            color=['green' if x > 0 else 'red' for x in combined['规模效应(SML)']])
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.title('月度规模效应 (小市值 - 大市值)', fontsize=14, fontweight='bold')
                    plt.ylabel('收益率差异 (%)', fontsize=10)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # 绘制累计规模效应
                    plt.subplot(2, 1, 2)
                    combined['累计规模效应'] = (1 + combined['规模效应(SML)']).cumprod() - 1
                    plt.plot(combined.index.astype(str), combined['累计规模效应'] * 100, 
                             linewidth=2, color='purple')
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.title('累计规模效应', fontsize=14, fontweight='bold')
                    plt.ylabel('累计收益率 (%)', fontsize=10)
                    plt.xlabel('月份', fontsize=10)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    save_path = os.path.join(self.results_dir, '规模效应时间序列图.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"规模效应时间序列图已保存到: {save_path}")
            else:
                print("警告：缺少小市值组或大市值组数据，无法创建规模效应时间序列图")
        except Exception as e:
            print(f"创建规模效应时间序列图时出错: {e}")
    
    def create_annualized_returns_comparison(self, grouped_data):
        """创建年化收益率对比图"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            annualized_returns = []
            labels = []
            
            for i in range(1, 6):
                if i in grouped_data and not grouped_data[i].empty and 'weekly_return' in grouped_data[i].columns:
                    # 计算年化收益率
                    weekly_return = grouped_data[i]['weekly_return'].mean()
                    annualized_return = (1 + weekly_return) ** 52 - 1  # 假设一年52周
                    annualized_returns.append(annualized_return * 100)  # 转换为百分比
                    labels.append(f'{i}组({"小" if i==1 else "大" if i==5 else "中"}市值)')
            
            if annualized_returns:
                # 创建柱状图
                colors = sns.color_palette("viridis", len(annualized_returns))
                bars = plt.bar(labels, annualized_returns, color=colors, alpha=0.8)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height, 
                             f'{height:.2f}%', ha='center', va='bottom')
                
                plt.title('各市值分组年化收益率对比图', fontsize=16, fontweight='bold')
                plt.ylabel('年化收益率 (%)', fontsize=12)
                plt.xlabel('市值分组', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                plt.tight_layout()
                save_path = os.path.join(self.results_dir, '年化收益率对比图.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"年化收益率对比图已保存到: {save_path}")
            else:
                print("警告：没有足够的数据创建年化收益率对比图")
        except Exception as e:
            print(f"创建年化收益率对比图时出错: {e}")
    
    def create_risk_adjusted_returns(self, grouped_data):
        """创建风险调整后收益率对比图（夏普比率）"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 假设无风险利率为年化2%
            risk_free_rate_annual = 0.02
            risk_free_rate_weekly = (1 + risk_free_rate_annual) ** (1/52) - 1
            
            # 准备数据
            sharpe_ratios = []
            labels = []
            annualized_returns = []
            annualized_volatilities = []
            
            for i in range(1, 6):
                if i in grouped_data and not grouped_data[i].empty and 'weekly_return' in grouped_data[i].columns:
                    # 计算周收益率的均值和标准差
                    weekly_return = grouped_data[i]['weekly_return'].mean()
                    weekly_volatility = grouped_data[i]['weekly_return'].std()
                    
                    # 计算夏普比率
                    if weekly_volatility > 0:
                        sharpe_ratio = (weekly_return - risk_free_rate_weekly) / weekly_volatility
                        sharpe_ratios.append(sharpe_ratio)
                        
                        # 计算年化收益率和波动率
                        annualized_return = (1 + weekly_return) ** 52 - 1
                        annualized_volatility = weekly_volatility * np.sqrt(52)
                        
                        annualized_returns.append(annualized_return * 100)
                        annualized_volatilities.append(annualized_volatility * 100)
                        
                        labels.append(f'{i}组({"小" if i==1 else "大" if i==5 else "中"}市值)')
            
            if sharpe_ratios:
                # 创建夏普比率柱状图
                colors = sns.color_palette("viridis", len(sharpe_ratios))
                bars = plt.bar(labels, sharpe_ratios, color=colors, alpha=0.8)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height, 
                             f'{height:.3f}', ha='center', va='bottom')
                
                plt.title('各市值分组风险调整后收益率 (夏普比率)', fontsize=16, fontweight='bold')
                plt.ylabel('夏普比率', fontsize=12)
                plt.xlabel('市值分组', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # 添加注释说明计算方法
                plt.figtext(0.5, 0.01, f'注：夏普比率 = (周收益率 - 无风险利率)/周波动率，假设无风险利率为年化{risk_free_rate_annual*100}%', 
                            ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_path = os.path.join(self.results_dir, '风险调整后收益率.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"风险调整后收益率图已保存到: {save_path}")
                
                # 创建风险收益散点图
                self.create_risk_return_scatter(annualized_returns, annualized_volatilities, labels)
            else:
                print("警告：没有足够的数据创建风险调整后收益率图")
        except Exception as e:
            print(f"创建风险调整后收益率图时出错: {e}")
    
    def create_risk_return_scatter(self, annualized_returns, annualized_volatilities, labels):
        """创建风险-收益散点图"""
        try:
            plt.figure(figsize=(10, 8))
            
            # 创建散点图
            scatter = plt.scatter(annualized_volatilities, annualized_returns, 
                                 s=100, alpha=0.7, c=range(len(labels)), cmap='viridis')
            
            # 添加标签
            for i, label in enumerate(labels):
                plt.annotate(label, 
                            (annualized_volatilities[i], annualized_returns[i]),
                            xytext=(5, 5), textcoords='offset points')
            
            plt.title('风险-收益散点图', fontsize=16, fontweight='bold')
            plt.xlabel('年化波动率 (%)', fontsize=12)
            plt.ylabel('年化收益率 (%)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('市值分组 (1=小市值, 5=大市值)')
            
            # 添加无风险利率参考线（假设年化2%）
            plt.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='无风险利率 (2%)')
            plt.legend()
            
            plt.tight_layout()
            save_path = os.path.join(self.results_dir, '风险收益散点图.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"风险收益散点图已保存到: {save_path}")
        except Exception as e:
            print(f"创建风险收益散点图时出错: {e}")

def visualize_scale_effect(grouped_data):
    """
    可视化A股市场规模效应的主函数
    
    Args:
        grouped_data: 按市值分组的股票数据
    """
    visualizer = ScaleEffectVisualizer()
    visualizer.create_all_visualizations(grouped_data)
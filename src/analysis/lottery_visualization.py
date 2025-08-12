"""
AI彩票预测系统 - 数据可视化模块
提供各种图表和可视化功能，用于展示彩票数据分析结果
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class LotteryVisualization:
    """彩票数据可视化类"""
    
    def __init__(self):
        """初始化可视化模块"""
        self.color_schemes = {
            'red_blue': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            'cool': ['#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#6C5CE7'],
            'warm': ['#E17055', '#FDCB6E', '#E84393', '#00B894', '#0984E3'],
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
        self.current_scheme = 'red_blue'
    
    def set_color_scheme(self, scheme_name: str):
        """设置颜色方案"""
        if scheme_name in self.color_schemes:
            self.current_scheme = scheme_name
    
    def create_number_frequency_chart(self, data: Dict, lottery_type: str, 
                                    chart_type: str = 'bar') -> plt.Figure:
        """
        创建号码频率图表
        
        Args:
            data: 号码频率数据
            lottery_type: 彩票类型
            chart_type: 图表类型 ('bar', 'heatmap', 'scatter')
            
        Returns:
            matplotlib图表对象
        """
        if lottery_type == "双色球":
            return self._create_ssq_frequency_chart(data, chart_type)
        else:
            return self._create_dlt_frequency_chart(data, chart_type)
    
    def _create_ssq_frequency_chart(self, data: Dict, chart_type: str) -> plt.Figure:
        """创建双色球频率图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        colors = self.color_schemes[self.current_scheme]
        
        # 解析数据
        red_balls = data.get('red_balls', {})
        blue_balls = data.get('blue_balls', {})
        
        if chart_type == 'bar':
            # 红球频率条形图
            if red_balls:
                red_nums = list(range(1, 34))
                red_freqs = [red_balls.get(str(num), 0) for num in red_nums]
                
                bars1 = ax1.bar(red_nums, red_freqs, color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.set_title('双色球红球号码频率分布', fontsize=16, fontweight='bold', pad=20)
                ax1.set_xlabel('红球号码', fontsize=12)
                ax1.set_ylabel('出现次数', fontsize=12)
                ax1.set_xticks(red_nums)
                ax1.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}', ha='center', va='bottom', fontsize=8)
            
            # 蓝球频率条形图
            if blue_balls:
                blue_nums = list(range(1, 17))
                blue_freqs = [blue_balls.get(str(num), 0) for num in blue_nums]
                
                bars2 = ax2.bar(blue_nums, blue_freqs, color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5)
                ax2.set_title('双色球蓝球号码频率分布', fontsize=16, fontweight='bold', pad=20)
                ax2.set_xlabel('蓝球号码', fontsize=12)
                ax2.set_ylabel('出现次数', fontsize=12)
                ax2.set_xticks(blue_nums)
                ax2.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        elif chart_type == 'heatmap':
            # 红球热力图 (6行x6列显示1-33号球，剩余3个位置空白)
            red_matrix = np.zeros((6, 6))
            for i in range(33):
                row, col = i // 6, i % 6
                num = i + 1
                freq = red_balls.get(str(num), 0)
                red_matrix[row, col] = freq
            
            im1 = ax1.imshow(red_matrix, cmap='Reds', aspect='auto')
            ax1.set_title('双色球红球频率热力图', fontsize=16, fontweight='bold', pad=20)
            
            # 添加号码标签
            for i in range(6):
                for j in range(6):
                    num = i * 6 + j + 1
                    if num <= 33:
                        freq = red_matrix[i, j]
                        ax1.text(j, i, f'{num}\n({int(freq)})', ha='center', va='center',
                                fontsize=10, color='white' if freq > red_matrix.max()/2 else 'black')
            
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # 蓝球热力图 (2行x8列显示1-16号球)
            blue_matrix = np.zeros((2, 8))
            for i in range(16):
                row, col = i // 8, i % 8
                num = i + 1
                freq = blue_balls.get(str(num), 0)
                blue_matrix[row, col] = freq
            
            im2 = ax2.imshow(blue_matrix, cmap='Blues', aspect='auto')
            ax2.set_title('双色球蓝球频率热力图', fontsize=16, fontweight='bold', pad=20)
            
            # 添加号码标签
            for i in range(2):
                for j in range(8):
                    num = i * 8 + j + 1
                    freq = blue_matrix[i, j]
                    ax2.text(j, i, f'{num}\n({int(freq)})', ha='center', va='center',
                            fontsize=10, color='white' if freq > blue_matrix.max()/2 else 'black')
            
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def _create_dlt_frequency_chart(self, data: Dict, chart_type: str) -> plt.Figure:
        """创建大乐透频率图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        colors = self.color_schemes[self.current_scheme]
        
        # 解析数据
        front_balls = data.get('front_balls', {})
        back_balls = data.get('back_balls', {})
        
        if chart_type == 'bar':
            # 前区频率条形图
            if front_balls:
                front_nums = list(range(1, 36))
                front_freqs = [front_balls.get(str(num), 0) for num in front_nums]
                
                bars1 = ax1.bar(front_nums, front_freqs, color=colors[2], alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.set_title('大乐透前区号码频率分布', fontsize=16, fontweight='bold', pad=20)
                ax1.set_xlabel('前区号码', fontsize=12)
                ax1.set_ylabel('出现次数', fontsize=12)
                ax1.set_xticks(front_nums[::2])  # 显示每隔一个号码
                ax1.grid(True, alpha=0.3)
                
                # 只在高频号码上添加标签（避免过于密集）
                max_freq = max(front_freqs) if front_freqs else 0
                for i, bar in enumerate(bars1):
                    height = bar.get_height()
                    if height > max_freq * 0.7:  # 只标注高频号码
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}', ha='center', va='bottom', fontsize=8)
            
            # 后区频率条形图
            if back_balls:
                back_nums = list(range(1, 13))
                back_freqs = [back_balls.get(str(num), 0) for num in back_nums]
                
                bars2 = ax2.bar(back_nums, back_freqs, color=colors[3], alpha=0.8, edgecolor='black', linewidth=0.5)
                ax2.set_title('大乐透后区号码频率分布', fontsize=16, fontweight='bold', pad=20)
                ax2.set_xlabel('后区号码', fontsize=12)
                ax2.set_ylabel('出现次数', fontsize=12)
                ax2.set_xticks(back_nums)
                ax2.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        elif chart_type == 'heatmap':
            # 前区热力图 (6行x6列显示1-35号球，最后一个位置空白)
            front_matrix = np.zeros((6, 6))
            for i in range(35):
                row, col = i // 6, i % 6
                num = i + 1
                freq = front_balls.get(str(num), 0)
                front_matrix[row, col] = freq
            
            im1 = ax1.imshow(front_matrix, cmap='Greens', aspect='auto')
            ax1.set_title('大乐透前区频率热力图', fontsize=16, fontweight='bold', pad=20)
            
            # 添加号码标签
            for i in range(6):
                for j in range(6):
                    num = i * 6 + j + 1
                    if num <= 35:
                        freq = front_matrix[i, j]
                        ax1.text(j, i, f'{num:02d}\n({int(freq)})', ha='center', va='center',
                                fontsize=9, color='white' if freq > front_matrix.max()/2 else 'black')
            
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # 后区热力图 (2行x6列显示1-12号球)
            back_matrix = np.zeros((2, 6))
            for i in range(12):
                row, col = i // 6, i % 6
                num = i + 1
                freq = back_balls.get(str(num), 0)
                back_matrix[row, col] = freq
            
            im2 = ax2.imshow(back_matrix, cmap='Oranges', aspect='auto')
            ax2.set_title('大乐透后区频率热力图', fontsize=16, fontweight='bold', pad=20)
            
            # 添加号码标签
            for i in range(2):
                for j in range(6):
                    num = i * 6 + j + 1
                    if num <= 12:
                        freq = back_matrix[i, j]
                        ax2.text(j, i, f'{num:02d}\n({int(freq)})', ha='center', va='center',
                                fontsize=10, color='white' if freq > back_matrix.max()/2 else 'black')
            
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def create_trend_chart(self, history_data: List[Dict], lottery_type: str, 
                          trend_type: str = 'frequency') -> plt.Figure:
        """
        创建走势图
        
        Args:
            history_data: 历史数据列表
            lottery_type: 彩票类型
            trend_type: 走势类型 ('frequency', 'sum', 'pattern')
            
        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        colors = self.color_schemes[self.current_scheme]
        
        if not history_data:
            ax.text(0.5, 0.5, '暂无历史数据', ha='center', va='center', fontsize=20)
            ax.set_title(f'{lottery_type}走势图', fontsize=16, fontweight='bold')
            return fig
        
        if trend_type == 'frequency':
            self._create_frequency_trend(ax, history_data, lottery_type, colors)
        elif trend_type == 'sum':
            self._create_sum_trend(ax, history_data, lottery_type, colors)
        elif trend_type == 'pattern':
            self._create_pattern_trend(ax, history_data, lottery_type, colors)
        
        plt.tight_layout()
        return fig
    
    def _create_frequency_trend(self, ax, history_data: List[Dict], lottery_type: str, colors: List[str]):
        """创建频率走势图"""
        # 准备数据
        periods = []
        if lottery_type == "双色球":
            red_trends = {i: [] for i in range(1, 34)}  # 红球1-33
            blue_trends = {i: [] for i in range(1, 17)}  # 蓝球1-16
            
            for record in history_data:
                periods.append(record.get('period', ''))
                red_nums = record.get('numbers', {}).get('red', '').split(',')
                blue_num = record.get('numbers', {}).get('blue', '')
                
                # 统计红球
                for red_num in range(1, 34):
                    red_trends[red_num].append(1 if str(red_num) in red_nums else 0)
                
                # 统计蓝球
                for blue_num in range(1, 17):
                    blue_trends[blue_num].append(1 if str(blue_num) == blue_num else 0)
        
        else:  # 大乐透
            front_trends = {i: [] for i in range(1, 36)}  # 前区1-35
            back_trends = {i: [] for i in range(1, 13)}   # 后区1-12
            
            for record in history_data:
                periods.append(record.get('period', ''))
                front_nums = record.get('numbers', {}).get('front', '').split(',')
                back_nums = record.get('numbers', {}).get('back', '').split(',')
                
                # 统计前区
                for front_num in range(1, 36):
                    front_trends[front_num].append(1 if str(front_num) in front_nums else 0)
                
                # 统计后区
                for back_num in range(1, 13):
                    back_trends[back_num].append(1 if str(back_num) in back_nums else 0)
        
        # 绘制热门号码的走势
        x_positions = range(len(periods))
        
        if lottery_type == "双色球":
            # 选择出现频率最高的5个红球绘制走势
            red_totals = {num: sum(trend) for num, trend in red_trends.items()}
            hot_reds = sorted(red_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (num, _) in enumerate(hot_reds):
                # 计算移动平均
                trend_data = red_trends[num]
                moving_avg = pd.Series(trend_data).rolling(window=min(10, len(trend_data))).mean()
                ax.plot(x_positions, moving_avg, label=f'红球{num}', 
                       color=colors[i % len(colors)], linewidth=2, marker='o', markersize=3)
        
        else:  # 大乐透
            # 选择出现频率最高的5个前区号码绘制走势
            front_totals = {num: sum(trend) for num, trend in front_trends.items()}
            hot_fronts = sorted(front_totals.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (num, _) in enumerate(hot_fronts):
                trend_data = front_trends[num]
                moving_avg = pd.Series(trend_data).rolling(window=min(10, len(trend_data))).mean()
                ax.plot(x_positions, moving_avg, label=f'前区{num:02d}', 
                       color=colors[i % len(colors)], linewidth=2, marker='o', markersize=3)
        
        ax.set_title(f'{lottery_type}热门号码走势图（移动平均）', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('开奖期数', fontsize=12)
        ax.set_ylabel('出现概率', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 设置x轴标签（显示部分期号）
        step = max(1, len(periods) // 10)
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=45)
    
    def _create_sum_trend(self, ax, history_data: List[Dict], lottery_type: str, colors: List[str]):
        """创建号码和值走势图"""
        periods = []
        sums = []
        
        for record in history_data:
            periods.append(record.get('period', ''))
            
            if lottery_type == "双色球":
                red_nums = record.get('numbers', {}).get('red', '').split(',')
                try:
                    red_sum = sum(int(num) for num in red_nums if num.isdigit())
                    sums.append(red_sum)
                except ValueError:
                    sums.append(0)
            else:  # 大乐透
                front_nums = record.get('numbers', {}).get('front', '').split(',')
                try:
                    front_sum = sum(int(num) for num in front_nums if num.isdigit())
                    sums.append(front_sum)
                except ValueError:
                    sums.append(0)
        
        x_positions = range(len(periods))
        
        # 绘制和值走势
        ax.plot(x_positions, sums, color=colors[0], linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        # 添加移动平均线
        if len(sums) > 5:
            moving_avg = pd.Series(sums).rolling(window=min(10, len(sums))).mean()
            ax.plot(x_positions, moving_avg, color=colors[1], linewidth=3, 
                   linestyle='--', label='移动平均线', alpha=0.8)
        
        # 添加平均线
        avg_sum = np.mean(sums) if sums else 0
        ax.axhline(y=avg_sum, color=colors[2], linestyle=':', linewidth=2, 
                  label=f'平均值: {avg_sum:.1f}', alpha=0.8)
        
        title_suffix = "红球" if lottery_type == "双色球" else "前区"
        ax.set_title(f'{lottery_type}{title_suffix}号码和值走势图', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('开奖期数', fontsize=12)
        ax.set_ylabel('号码和值', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置x轴标签
        step = max(1, len(periods) // 10)
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=45)
    
    def _create_pattern_trend(self, ax, history_data: List[Dict], lottery_type: str, colors: List[str]):
        """创建形态走势图（奇偶比、大小比等）"""
        periods = []
        odd_ratios = []  # 奇数比例
        big_ratios = []  # 大数比例
        
        for record in history_data:
            periods.append(record.get('period', ''))
            
            if lottery_type == "双色球":
                red_nums = record.get('numbers', {}).get('red', '').split(',')
                nums = [int(num) for num in red_nums if num.isdigit()]
            else:  # 大乐透
                front_nums = record.get('numbers', {}).get('front', '').split(',')
                nums = [int(num) for num in front_nums if num.isdigit()]
            
            if nums:
                # 计算奇数比例
                odd_count = sum(1 for num in nums if num % 2 == 1)
                odd_ratio = odd_count / len(nums)
                odd_ratios.append(odd_ratio)
                
                # 计算大数比例（>中位数的号码）
                max_num = 33 if lottery_type == "双色球" else 35
                big_count = sum(1 for num in nums if num > max_num // 2)
                big_ratio = big_count / len(nums)
                big_ratios.append(big_ratio)
            else:
                odd_ratios.append(0)
                big_ratios.append(0)
        
        x_positions = range(len(periods))
        
        # 绘制奇偶比走势
        ax.plot(x_positions, odd_ratios, color=colors[0], linewidth=2, 
               marker='o', markersize=4, label='奇数比例', alpha=0.8)
        
        # 绘制大小比走势
        ax.plot(x_positions, big_ratios, color=colors[1], linewidth=2, 
               marker='s', markersize=4, label='大数比例', alpha=0.8)
        
        # 添加理论平均线
        ax.axhline(y=0.5, color=colors[2], linestyle='--', linewidth=2, 
                  label='理论平均值 (0.5)', alpha=0.6)
        
        ax.set_title(f'{lottery_type}号码形态走势图', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('开奖期数', fontsize=12)
        ax.set_ylabel('比例', fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置x轴标签
        step = max(1, len(periods) // 10)
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=45)
    
    def create_prediction_accuracy_chart(self, accuracy_data: Dict) -> plt.Figure:
        """
        创建预测准确率图表
        
        Args:
            accuracy_data: 准确率数据
            
        Returns:
            matplotlib图表对象
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        colors = self.color_schemes[self.current_scheme]
        
        # 1. 总体准确率饼图
        if 'by_lottery_type' in accuracy_data and accuracy_data['by_lottery_type']:
            lottery_types = list(accuracy_data['by_lottery_type'].keys())
            counts = [data['count'] for data in accuracy_data['by_lottery_type'].values()]
            
            ax1.pie(counts, labels=lottery_types, autopct='%1.1f%%', colors=colors[:len(counts)])
            ax1.set_title('预测次数分布', fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, '暂无预测数据', ha='center', va='center', fontsize=12)
            ax1.set_title('预测次数分布', fontsize=14, fontweight='bold')
        
        # 2. 平均命中数柱状图
        if 'by_lottery_type' in accuracy_data and accuracy_data['by_lottery_type']:
            lottery_types = list(accuracy_data['by_lottery_type'].keys())
            avg_hits = [data.get('avg_hits', 0) or 0 for data in accuracy_data['by_lottery_type'].values()]
            
            bars = ax2.bar(lottery_types, avg_hits, color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
            ax2.set_title('平均命中号码数', fontsize=14, fontweight='bold')
            ax2.set_ylabel('平均命中数')
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(0.5, 0.5, '暂无命中数据', ha='center', va='center', fontsize=12)
            ax2.set_title('平均命中号码数', fontsize=14, fontweight='bold')
        
        # 3. 准确率分布
        if 'by_lottery_type' in accuracy_data and accuracy_data['by_lottery_type']:
            lottery_types = list(accuracy_data['by_lottery_type'].keys())
            accuracies = [data.get('avg_accuracy', 0) or 0 for data in accuracy_data['by_lottery_type'].values()]
            
            bars = ax3.bar(lottery_types, accuracies, color=colors[2], alpha=0.8, edgecolor='black', linewidth=1)
            ax3.set_title('平均准确率', fontsize=14, fontweight='bold')
            ax3.set_ylabel('准确率 (%)')
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        else:
            ax3.text(0.5, 0.5, '暂无准确率数据', ha='center', va='center', fontsize=12)
            ax3.set_title('平均准确率', fontsize=14, fontweight='bold')
        
        # 4. 统计摘要
        ax4.axis('off')
        stats_text = f"""
        📊 预测统计摘要
        
        总预测次数: {accuracy_data.get('total_predictions', 0)}
        平均命中数: {accuracy_data.get('avg_hit_count', 0):.2f}
        最高命中数: {accuracy_data.get('max_hit_count', 0)}
        平均准确率: {accuracy_data.get('avg_accuracy_score', 0):.2f}%
        
        💡 提示：
        • 命中数越高，预测效果越好
        • 准确率反映整体预测质量
        • 持续记录有助于改进预测算法
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor=colors[0], alpha=0.1))
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_analysis_chart(self, analysis_data: Dict) -> plt.Figure:
        """
        创建综合分析图表
        
        Args:
            analysis_data: 综合分析数据
            
        Returns:
            matplotlib图表对象
        """
        fig = plt.figure(figsize=(20, 14))
        colors = self.color_schemes[self.current_scheme]
        
        # 创建子图网格
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 冷热号分析 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_hot_cold_analysis(ax1, analysis_data.get('hot_cold', {}), colors)
        
        # 2. 遗漏值分析 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_missing_analysis(ax2, analysis_data.get('missing', {}), colors)
        
        # 3. 奇偶分析 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_odd_even_analysis(ax3, analysis_data.get('odd_even', {}), colors)
        
        # 4. 大小分析 (左中)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_big_small_analysis(ax4, analysis_data.get('big_small', {}), colors)
        
        # 5. 和值分析 (中中)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_sum_analysis(ax5, analysis_data.get('sum_analysis', {}), colors)
        
        # 6. 跨度分析 (右中)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_span_analysis(ax6, analysis_data.get('span', {}), colors)
        
        # 7. 连号分析 (左下)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_consecutive_analysis(ax7, analysis_data.get('consecutive', {}), colors)
        
        # 8. 重号分析 (中下)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_repeat_analysis(ax8, analysis_data.get('repeat', {}), colors)
        
        # 9. 综合评分 (右下)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_comprehensive_score(ax9, analysis_data.get('scores', {}), colors)
        
        # 添加总标题
        fig.suptitle('彩票号码综合分析报告', fontsize=20, fontweight='bold', y=0.95)
        
        return fig
    
    def _plot_hot_cold_analysis(self, ax, data: Dict, colors: List[str]):
        """绘制冷热号分析"""
        if not data:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center')
            ax.set_title('冷热号分析')
            return
        
        hot_numbers = data.get('hot', [])
        cold_numbers = data.get('cold', [])
        
        categories = ['热号', '冷号']
        counts = [len(hot_numbers), len(cold_numbers)]
        
        bars = ax.bar(categories, counts, color=[colors[0], colors[1]], alpha=0.8)
        ax.set_title('冷热号分布', fontweight='bold')
        ax.set_ylabel('号码数量')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
    
    def _plot_missing_analysis(self, ax, data: Dict, colors: List[str]):
        """绘制遗漏值分析"""
        if not data:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center')
            ax.set_title('遗漏值分析')
            return
        
        missing_ranges = ['0-5期', '6-10期', '11-20期', '20期以上']
        counts = [
            data.get('range_0_5', 0),
            data.get('range_6_10', 0),
            data.get('range_11_20', 0),
            data.get('range_20_plus', 0)
        ]
        
        bars = ax.bar(missing_ranges, counts, color=colors[2], alpha=0.8)
        ax.set_title('遗漏期数分布', fontweight='bold')
        ax.set_ylabel('号码数量')
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
    
    def _plot_odd_even_analysis(self, ax, data: Dict, colors: List[str]):
        """绘制奇偶分析"""
        if not data:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center')
            ax.set_title('奇偶分析')
            return
        
        categories = ['奇数', '偶数']
        counts = [data.get('odd_count', 0), data.get('even_count', 0)]
        
        wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1.1f%%', 
                                         colors=[colors[3], colors[4]])
        ax.set_title('奇偶号码分布', fontweight='bold')
    
    def _plot_big_small_analysis(self, ax, data: Dict, colors: List[str]):
        """绘制大小号分析"""
        if not data:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center')
            ax.set_title('大小号分析')
            return
        
        categories = ['小号', '大号']
        counts = [data.get('small_count', 0), data.get('big_count', 0)]
        
        bars = ax.bar(categories, counts, color=[colors[0], colors[2]], alpha=0.8)
        ax.set_title('大小号分布', fontweight='bold')
        ax.set_ylabel('号码数量')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
    
    def _plot_sum_analysis(self, ax, data: Dict, colors: List[str]):
        """绘制和值分析"""
        if not data:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center')
            ax.set_title('和值分析')
            return
        
        recent_sums = data.get('recent_sums', [])
        if recent_sums:
            ax.plot(range(len(recent_sums)), recent_sums, color=colors[1], 
                   marker='o', linewidth=2, markersize=4)
            ax.axhline(y=np.mean(recent_sums), color=colors[3], 
                      linestyle='--', label=f'平均值: {np.mean(recent_sums):.1f}')
            ax.legend()
        
        ax.set_title('近期和值走势', fontweight='bold')
        ax.set_xlabel('期数')
        ax.set_ylabel('和值')
        ax.grid(True, alpha=0.3)
    
    def _plot_span_analysis(self, ax, data: Dict, colors: List[str]):
        """绘制跨度分析"""
        if not data:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center')
            ax.set_title('跨度分析')
            return
        
        span_ranges = ['0-10', '11-20', '21-30', '30+']
        counts = [
            data.get('span_0_10', 0),
            data.get('span_11_20', 0),
            data.get('span_21_30', 0),
            data.get('span_30_plus', 0)
        ]
        
        bars = ax.bar(span_ranges, counts, color=colors[4], alpha=0.8)
        ax.set_title('跨度分布', fontweight='bold')
        ax.set_ylabel('出现次数')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
    
    def _plot_consecutive_analysis(self, ax, data: Dict, colors: List[str]):
        """绘制连号分析"""
        if not data:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center')
            ax.set_title('连号分析')
            return
        
        consecutive_counts = data.get('consecutive_distribution', {})
        if consecutive_counts:
            numbers = list(consecutive_counts.keys())
            counts = list(consecutive_counts.values())
            
            bars = ax.bar(numbers, counts, color=colors[0], alpha=0.8)
            ax.set_title('连号出现频率', fontweight='bold')
            ax.set_xlabel('连号个数')
            ax.set_ylabel('出现次数')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom')
    
    def _plot_repeat_analysis(self, ax, data: Dict, colors: List[str]):
        """绘制重号分析"""
        if not data:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center')
            ax.set_title('重号分析')
            return
        
        repeat_counts = data.get('repeat_distribution', {})
        if repeat_counts:
            numbers = list(repeat_counts.keys())
            counts = list(repeat_counts.values())
            
            bars = ax.bar(numbers, counts, color=colors[1], alpha=0.8)
            ax.set_title('重号出现频率', fontweight='bold')
            ax.set_xlabel('重号个数')
            ax.set_ylabel('出现次数')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom')
    
    def _plot_comprehensive_score(self, ax, data: Dict, colors: List[str]):
        """绘制综合评分"""
        if not data:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center')
            ax.set_title('综合评分')
            return
        
        categories = ['规律性', '随机性', '热度', '稳定性', '综合']
        scores = [
            data.get('regularity_score', 0),
            data.get('randomness_score', 0),
            data.get('hotness_score', 0),
            data.get('stability_score', 0),
            data.get('overall_score', 0)
        ]
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        scores_closed = scores + [scores[0]]  # 闭合图形
        angles_closed = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles_closed, scores_closed, 'o-', linewidth=2, color=colors[2])
        ax.fill(angles_closed, scores_closed, alpha=0.25, color=colors[2])
        ax.set_xticks(angles)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('综合评分雷达图', fontweight='bold')
        ax.grid(True)
    
    def save_chart(self, fig: plt.Figure, filename: str, dpi: int = 300, 
                  format: str = 'png') -> str:
        """
        保存图表
        
        Args:
            fig: matplotlib图表对象
            filename: 文件名
            dpi: 分辨率
            format: 文件格式
            
        Returns:
            保存的文件路径
        """
        # 确保输出目录存在
        output_dir = Path("charts_output")
        output_dir.mkdir(exist_ok=True)
        
        # 生成完整文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.{format}"
        filepath = output_dir / full_filename
        
        # 保存图表
        fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        return str(filepath)


# 使用示例
if __name__ == "__main__":
    # 创建可视化实例
    viz = LotteryVisualization()
    
    # 示例数据
    sample_frequency_data = {
        'red_balls': {str(i): np.random.randint(5, 25) for i in range(1, 34)},
        'blue_balls': {str(i): np.random.randint(3, 15) for i in range(1, 17)}
    }
    
    # 创建频率图表
    fig = viz.create_number_frequency_chart(sample_frequency_data, "双色球", "bar")
    
    # 保存图表
    filepath = viz.save_chart(fig, "frequency_chart_demo")
    print(f"图表已保存至: {filepath}")
    
    plt.show()

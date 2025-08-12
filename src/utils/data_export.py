"""
AI彩票预测系统 - 数据导出模块
提供多种格式的数据导出功能：Excel、PDF、图片、HTML等
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import base64
from io import BytesIO
import logging

try:
    from core.database_adapter import DatabaseAdapter
except ImportError:
    from ..core.database_adapter import DatabaseAdapter
try:
    from analysis.prediction_statistics import PredictionStatistics
    from analysis.lottery_analysis import LotteryAnalysis
    from analysis.lottery_visualization import LotteryVisualization
except ImportError:
    from ..analysis.prediction_statistics import PredictionStatistics
    from ..analysis.lottery_analysis import LotteryAnalysis
    from ..analysis.lottery_visualization import LotteryVisualization

logger = logging.getLogger(__name__)


class DataExporter:
    """数据导出器"""
    
    def __init__(self, db_adapter: DatabaseAdapter = None):
        """
        初始化数据导出器
        
        Args:
            db_adapter: 数据库适配器实例
        """
        self.db_adapter = db_adapter or DatabaseAdapter()
        self.statistics = PredictionStatistics()
        self.analysis = LotteryAnalysis()
        self.visualization = LotteryVisualization()
        
        # 创建导出目录
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.export_dir / "excel").mkdir(exist_ok=True)
        (self.export_dir / "pdf").mkdir(exist_ok=True)
        (self.export_dir / "images").mkdir(exist_ok=True)
        (self.export_dir / "html").mkdir(exist_ok=True)
        (self.export_dir / "json").mkdir(exist_ok=True)
    
    def export_prediction_data(self, export_format: str = 'excel', 
                             model_name: str = None, lottery_type: str = None,
                             days: int = 30) -> str:
        """
        导出预测数据
        
        Args:
            export_format: 导出格式 ('excel', 'pdf', 'json', 'html')
            model_name: 模型名称筛选
            lottery_type: 彩票类型筛选
            days: 统计天数
            
        Returns:
            导出文件路径
        """
        try:
            # 获取预测数据
            prediction_data = self._get_prediction_data(model_name, lottery_type, days)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"prediction_data_{model_name or 'all'}_{lottery_type or 'all'}_{days}days_{timestamp}"
            
            if export_format == 'excel':
                return self._export_to_excel(prediction_data, base_filename)
            elif export_format == 'pdf':
                return self._export_prediction_to_pdf(prediction_data, base_filename)
            elif export_format == 'json':
                return self._export_to_json(prediction_data, base_filename)
            elif export_format == 'html':
                return self._export_prediction_to_html(prediction_data, base_filename)
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
                
        except Exception as e:
            logger.error(f"导出预测数据失败: {e}")
            raise
    
    def export_analysis_report(self, lottery_type: str, period_range: str = "最近100期",
                             export_format: str = 'pdf', include_charts: bool = True) -> str:
        """
        导出分析报告
        
        Args:
            lottery_type: 彩票类型
            period_range: 期数范围
            export_format: 导出格式
            include_charts: 是否包含图表
            
        Returns:
            导出文件路径
        """
        try:
            # 获取分析结果
            analysis_result = self.analysis.comprehensive_analysis(lottery_type, period_range)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"analysis_report_{lottery_type}_{period_range}_{timestamp}"
            
            if export_format == 'pdf':
                return self._export_analysis_to_pdf(analysis_result, base_filename, include_charts)
            elif export_format == 'html':
                return self._export_analysis_to_html(analysis_result, base_filename, include_charts)
            elif export_format == 'excel':
                return self._export_analysis_to_excel(analysis_result, base_filename)
            elif export_format == 'json':
                return self._export_to_json(analysis_result, base_filename)
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
                
        except Exception as e:
            logger.error(f"导出分析报告失败: {e}")
            raise
    
    def export_charts(self, chart_data: Dict[str, Any], export_format: str = 'png',
                     dpi: int = 300, size: tuple = (12, 8)) -> List[str]:
        """
        导出图表
        
        Args:
            chart_data: 图表数据
            export_format: 导出格式 ('png', 'pdf', 'svg', 'jpg')
            dpi: 分辨率
            size: 图表尺寸
            
        Returns:
            导出文件路径列表
        """
        try:
            exported_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 根据图表类型生成不同的图表
            chart_types = chart_data.get('types', ['frequency', 'trend', 'analysis'])
            
            for chart_type in chart_types:
                try:
                    fig = self._create_chart_by_type(chart_type, chart_data)
                    
                    if fig:
                        filename = f"chart_{chart_type}_{timestamp}.{export_format}"
                        filepath = self.export_dir / "images" / filename
                        
                        fig.savefig(filepath, format=export_format, dpi=dpi, 
                                   bbox_inches='tight', facecolor='white')
                        
                        exported_files.append(str(filepath))
                        plt.close(fig)
                        
                except Exception as e:
                    logger.error(f"导出图表 {chart_type} 失败: {e}")
                    continue
            
            return exported_files
            
        except Exception as e:
            logger.error(f"导出图表失败: {e}")
            raise
    
    def export_comprehensive_report(self, lottery_type: str, model_name: str = None,
                                  period_range: str = "最近100期", days: int = 30) -> str:
        """
        导出综合报告（包含预测、分析、图表）
        
        Args:
            lottery_type: 彩票类型
            model_name: 模型名称
            period_range: 期数范围
            days: 统计天数
            
        Returns:
            导出文件路径
        """
        try:
            # 收集所有数据
            prediction_data = self._get_prediction_data(model_name, lottery_type, days)
            analysis_result = self.analysis.comprehensive_analysis(lottery_type, period_range)
            statistics_data = self.statistics.get_comprehensive_statistics(days)
            
            # 生成图表
            charts_data = self._prepare_charts_data(lottery_type, analysis_result, statistics_data)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_report_{lottery_type}_{model_name or 'all'}_{timestamp}.pdf"
            filepath = self.export_dir / "pdf" / filename
            
            # 创建PDF报告
            self._create_comprehensive_pdf(
                filepath, lottery_type, prediction_data, 
                analysis_result, statistics_data, charts_data
            )
            
            # 记录导出历史
            self.db_adapter.save_export_record(
                export_type='pdf',
                export_content='comprehensive_report',
                file_path=str(filepath),
                file_size=os.path.getsize(filepath) if os.path.exists(filepath) else None,
                export_params={
                    'lottery_type': lottery_type,
                    'model_name': model_name,
                    'period_range': period_range,
                    'days': days
                }
            )
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"导出综合报告失败: {e}")
            raise
    
    def _get_prediction_data(self, model_name: str = None, lottery_type: str = None, 
                           days: int = 30) -> Dict[str, Any]:
        """获取预测数据"""
        try:
            # 获取预测历史
            prediction_history = self.db_adapter.get_prediction_history(lottery_type)
            
            # 获取准确率统计
            accuracy_stats = self.statistics.get_comprehensive_statistics(days)
            
            # 获取模型性能
            model_performance = self.statistics.get_model_performance(model_name, lottery_type, days)
            
            return {
                'prediction_history': prediction_history,
                'accuracy_stats': accuracy_stats,
                'model_performance': model_performance,
                'summary': {
                    'total_predictions': len(prediction_history),
                    'model_name': model_name,
                    'lottery_type': lottery_type,
                    'period_days': days,
                    'export_time': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"获取预测数据失败: {e}")
            return {}
    
    def _export_to_excel(self, data: Dict[str, Any], base_filename: str) -> str:
        """导出到Excel格式"""
        try:
            filename = f"{base_filename}.xlsx"
            filepath = self.export_dir / "excel" / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 写入摘要信息
                if 'summary' in data:
                    summary_df = pd.DataFrame([data['summary']])
                    summary_df.to_excel(writer, sheet_name='摘要', index=False)
                
                # 写入预测历史
                if 'prediction_history' in data and data['prediction_history']:
                    pred_df = pd.DataFrame(data['prediction_history'])
                    pred_df.to_excel(writer, sheet_name='预测历史', index=False)
                
                # 写入准确率统计
                if 'accuracy_stats' in data:
                    stats = data['accuracy_stats']
                    if 'overall' in stats:
                        overall_df = pd.DataFrame([stats['overall']])
                        overall_df.to_excel(writer, sheet_name='总体统计', index=False)
                    
                    if 'by_model' in stats and stats['by_model']:
                        model_df = pd.DataFrame.from_dict(stats['by_model'], orient='index')
                        model_df.to_excel(writer, sheet_name='模型对比', index=True)
                
                # 写入模型性能
                if 'model_performance' in data and data['model_performance']:
                    perf_df = pd.DataFrame([data['model_performance']])
                    perf_df.to_excel(writer, sheet_name='模型性能', index=False)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"导出Excel失败: {e}")
            raise
    
    def _export_to_json(self, data: Dict[str, Any], base_filename: str) -> str:
        """导出到JSON格式"""
        try:
            filename = f"{base_filename}.json"
            filepath = self.export_dir / "json" / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"导出JSON失败: {e}")
            raise
    
    def _export_prediction_to_pdf(self, data: Dict[str, Any], base_filename: str) -> str:
        """导出预测数据到PDF"""
        try:
            filename = f"{base_filename}.pdf"
            filepath = self.export_dir / "pdf" / filename
            
            doc = SimpleDocTemplate(str(filepath), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # 添加标题
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # 居中
            )
            story.append(Paragraph("AI彩票预测数据报告", title_style))
            story.append(Spacer(1, 20))
            
            # 添加摘要信息
            if 'summary' in data:
                summary = data['summary']
                story.append(Paragraph("报告摘要", styles['Heading2']))
                summary_text = f"""
                • 模型名称: {summary.get('model_name', '全部模型')}
                • 彩票类型: {summary.get('lottery_type', '全部类型')}
                • 统计天数: {summary.get('period_days', 30)}天
                • 预测总数: {summary.get('total_predictions', 0)}次
                • 导出时间: {summary.get('export_time', 'N/A')}
                """
                story.append(Paragraph(summary_text, styles['Normal']))
                story.append(Spacer(1, 20))
            
            # 添加模型性能表格
            if 'model_performance' in data and data['model_performance']:
                story.append(Paragraph("模型性能分析", styles['Heading2']))
                
                perf = data['model_performance']
                perf_data = [
                    ['指标', '数值'],
                    ['预测总数', str(perf.get('total_predictions', 0))],
                    ['平均准确率', f"{perf.get('avg_accuracy', 0):.2f}%"],
                    ['最高准确率', f"{perf.get('max_accuracy', 0):.2f}%"],
                    ['平均命中数', f"{perf.get('avg_hits', 0):.2f}"],
                    ['最高命中数', str(perf.get('max_hits', 0))],
                    ['成功率', f"{perf.get('success_rate', 0):.2f}%"]
                ]
                
                perf_table = Table(perf_data)
                perf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(perf_table)
                story.append(Spacer(1, 20))
            
            # 构建PDF
            doc.build(story)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"导出预测PDF失败: {e}")
            raise
    
    def _export_analysis_to_pdf(self, analysis_result: Dict[str, Any], 
                              base_filename: str, include_charts: bool = True) -> str:
        """导出分析结果到PDF"""
        try:
            filename = f"{base_filename}.pdf"
            filepath = self.export_dir / "pdf" / filename
            
            doc = SimpleDocTemplate(str(filepath), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # 添加标题
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1
            )
            story.append(Paragraph(f"{analysis_result.get('lottery_type', '')}数据分析报告", title_style))
            story.append(Spacer(1, 20))
            
            # 添加基本信息
            basic_info = f"""
            • 彩票类型: {analysis_result.get('lottery_type', 'N/A')}
            • 分析期数: {analysis_result.get('period_range', 'N/A')}
            • 数据样本: {analysis_result.get('data_count', 0)}期
            • 置信度评分: {analysis_result.get('confidence_score', 0):.1f}/100
            • 分析时间: {analysis_result.get('analysis_date', 'N/A')}
            """
            story.append(Paragraph("基本信息", styles['Heading2']))
            story.append(Paragraph(basic_info, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # 添加综合评分
            if 'scores' in analysis_result:
                scores = analysis_result['scores']
                story.append(Paragraph("综合评分", styles['Heading2']))
                
                scores_data = [
                    ['评分项目', '分数'],
                    ['规律性评分', f"{scores.get('regularity_score', 0):.1f}/100"],
                    ['随机性评分', f"{scores.get('randomness_score', 0):.1f}/100"],
                    ['热度评分', f"{scores.get('hotness_score', 0):.1f}/100"],
                    ['稳定性评分', f"{scores.get('stability_score', 0):.1f}/100"],
                    ['综合评分', f"{scores.get('overall_score', 0):.1f}/100"]
                ]
                
                scores_table = Table(scores_data)
                scores_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(scores_table)
                story.append(Spacer(1, 20))
            
            # 添加分析建议
            if 'recommendations' in analysis_result and analysis_result['recommendations']:
                story.append(Paragraph("分析建议", styles['Heading2']))
                recommendations_text = ""
                for i, rec in enumerate(analysis_result['recommendations'], 1):
                    recommendations_text += f"{i}. {rec}<br/>"
                
                story.append(Paragraph(recommendations_text, styles['Normal']))
                story.append(Spacer(1, 20))
            
            # 如果包含图表，添加图表
            if include_charts:
                try:
                    # 创建分析图表
                    fig = self.visualization.create_comprehensive_analysis_chart(analysis_result)
                    
                    # 保存图表为临时文件
                    temp_chart_path = self.export_dir / "images" / f"temp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    fig.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
                    
                    # 添加图表到PDF
                    story.append(Paragraph("综合分析图表", styles['Heading2']))
                    chart_img = Image(str(temp_chart_path), width=6*inch, height=4*inch)
                    story.append(chart_img)
                    
                    plt.close(fig)
                    
                    # 删除临时文件
                    temp_chart_path.unlink(missing_ok=True)
                    
                except Exception as e:
                    logger.error(f"添加图表到PDF失败: {e}")
            
            # 构建PDF
            doc.build(story)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"导出分析PDF失败: {e}")
            raise
    
    def _export_prediction_to_html(self, data: Dict[str, Any], base_filename: str) -> str:
        """导出预测数据到HTML"""
        try:
            filename = f"{base_filename}.html"
            filepath = self.export_dir / "html" / filename
            
            # 构建HTML内容
            html_content = f"""
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AI彩票预测数据报告</title>
                <style>
                    body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
                    .header {{ text-align: center; color: #2c3e50; }}
                    .section {{ margin: 20px 0; }}
                    .section h2 {{ color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #3498db; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🎯 AI彩票预测数据报告</h1>
                    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            # 添加摘要信息
            if 'summary' in data:
                summary = data['summary']
                html_content += f"""
                <div class="section">
                    <h2>📊 报告摘要</h2>
                    <div class="summary">
                        <p><strong>模型名称:</strong> {summary.get('model_name', '全部模型')}</p>
                        <p><strong>彩票类型:</strong> {summary.get('lottery_type', '全部类型')}</p>
                        <p><strong>统计天数:</strong> {summary.get('period_days', 30)}天</p>
                        <p><strong>预测总数:</strong> {summary.get('total_predictions', 0)}次</p>
                    </div>
                </div>
                """
            
            # 添加模型性能表格
            if 'model_performance' in data and data['model_performance']:
                perf = data['model_performance']
                html_content += f"""
                <div class="section">
                    <h2>📈 模型性能分析</h2>
                    <table>
                        <tr><th>指标</th><th>数值</th></tr>
                        <tr><td>预测总数</td><td>{perf.get('total_predictions', 0)}</td></tr>
                        <tr><td>平均准确率</td><td>{perf.get('avg_accuracy', 0):.2f}%</td></tr>
                        <tr><td>最高准确率</td><td>{perf.get('max_accuracy', 0):.2f}%</td></tr>
                        <tr><td>平均命中数</td><td>{perf.get('avg_hits', 0):.2f}</td></tr>
                        <tr><td>最高命中数</td><td>{perf.get('max_hits', 0)}</td></tr>
                        <tr><td>成功率</td><td>{perf.get('success_rate', 0):.2f}%</td></tr>
                    </table>
                </div>
                """
            
            # 添加预测历史表格
            if 'prediction_history' in data and data['prediction_history']:
                html_content += """
                <div class="section">
                    <h2>📋 预测历史记录</h2>
                    <table>
                        <tr><th>ID</th><th>彩票类型</th><th>模型</th><th>预测类型</th><th>创建时间</th></tr>
                """
                
                for record in data['prediction_history'][:20]:  # 只显示前20条
                    html_content += f"""
                        <tr>
                            <td>{record.get('id', '')}</td>
                            <td>{record.get('lottery_type', '')}</td>
                            <td>{record.get('model_name', '')}</td>
                            <td>{record.get('prediction_type', '')}</td>
                            <td>{record.get('created_at', '')}</td>
                        </tr>
                    """
                
                html_content += "</table></div>"
            
            html_content += """
            </body>
            </html>
            """
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"导出HTML失败: {e}")
            raise
    
    def _export_analysis_to_html(self, analysis_result: Dict[str, Any], 
                               base_filename: str, include_charts: bool = True) -> str:
        """导出分析结果到HTML"""
        try:
            filename = f"{base_filename}.html"
            filepath = self.export_dir / "html" / filename
            
            # 构建HTML内容
            html_content = f"""
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{analysis_result.get('lottery_type', '')}数据分析报告</title>
                <style>
                    body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
                    .header {{ text-align: center; color: #2c3e50; }}
                    .section {{ margin: 20px 0; }}
                    .section h2 {{ color: #e74c3c; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }}
                    .info-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                    .score-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
                    .score-item {{ background-color: #3498db; color: white; padding: 10px; border-radius: 5px; text-align: center; }}
                    ul {{ list-style-type: none; padding: 0; }}
                    li {{ background-color: #f8f9fa; margin: 5px 0; padding: 10px; border-left: 4px solid #3498db; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🔍 {analysis_result.get('lottery_type', '')}数据分析报告</h1>
                    <p>分析时间: {analysis_result.get('analysis_date', 'N/A')}</p>
                </div>
            """
            
            # 添加基本信息
            html_content += f"""
            <div class="section">
                <h2>📊 基本信息</h2>
                <div class="info-box">
                    <p><strong>彩票类型:</strong> {analysis_result.get('lottery_type', 'N/A')}</p>
                    <p><strong>分析期数:</strong> {analysis_result.get('period_range', 'N/A')}</p>
                    <p><strong>数据样本:</strong> {analysis_result.get('data_count', 0)}期</p>
                    <p><strong>置信度评分:</strong> {analysis_result.get('confidence_score', 0):.1f}/100</p>
                </div>
            </div>
            """
            
            # 添加综合评分
            if 'scores' in analysis_result:
                scores = analysis_result['scores']
                html_content += f"""
                <div class="section">
                    <h2>🎯 综合评分</h2>
                    <div class="score-grid">
                        <div class="score-item">
                            <h4>规律性</h4>
                            <p>{scores.get('regularity_score', 0):.1f}/100</p>
                        </div>
                        <div class="score-item">
                            <h4>随机性</h4>
                            <p>{scores.get('randomness_score', 0):.1f}/100</p>
                        </div>
                        <div class="score-item">
                            <h4>热度指数</h4>
                            <p>{scores.get('hotness_score', 0):.1f}/100</p>
                        </div>
                        <div class="score-item">
                            <h4>稳定性</h4>
                            <p>{scores.get('stability_score', 0):.1f}/100</p>
                        </div>
                    </div>
                </div>
                """
            
            # 添加分析建议
            if 'recommendations' in analysis_result and analysis_result['recommendations']:
                html_content += """
                <div class="section">
                    <h2>💡 分析建议</h2>
                    <ul>
                """
                
                for rec in analysis_result['recommendations']:
                    html_content += f"<li>{rec}</li>"
                
                html_content += "</ul></div>"
            
            # 如果包含图表，添加图表
            if include_charts:
                try:
                    # 创建图表并转换为base64
                    fig = self.visualization.create_comprehensive_analysis_chart(analysis_result)
                    
                    # 保存为base64字符串
                    buffer = BytesIO()
                    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    buffer.close()
                    
                    html_content += f"""
                    <div class="section">
                        <h2>📈 综合分析图表</h2>
                        <img src="data:image/png;base64,{image_base64}" style="width: 100%; max-width: 800px;" alt="综合分析图表">
                    </div>
                    """
                    
                    plt.close(fig)
                    
                except Exception as e:
                    logger.error(f"添加图表到HTML失败: {e}")
            
            html_content += """
            </body>
            </html>
            """
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"导出分析HTML失败: {e}")
            raise
    
    def _export_analysis_to_excel(self, analysis_result: Dict[str, Any], base_filename: str) -> str:
        """导出分析结果到Excel"""
        try:
            filename = f"{base_filename}.xlsx"
            filepath = self.export_dir / "excel" / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 基本信息
                basic_info = {
                    '彩票类型': analysis_result.get('lottery_type', 'N/A'),
                    '分析期数': analysis_result.get('period_range', 'N/A'),
                    '数据样本': analysis_result.get('data_count', 0),
                    '置信度评分': analysis_result.get('confidence_score', 0),
                    '分析时间': analysis_result.get('analysis_date', 'N/A')
                }
                basic_df = pd.DataFrame([basic_info])
                basic_df.to_excel(writer, sheet_name='基本信息', index=False)
                
                # 综合评分
                if 'scores' in analysis_result:
                    scores_df = pd.DataFrame([analysis_result['scores']])
                    scores_df.to_excel(writer, sheet_name='综合评分', index=False)
                
                # 频率分析
                if 'frequency' in analysis_result and 'error' not in analysis_result['frequency']:
                    freq_data = analysis_result['frequency']
                    if 'red_frequency' in freq_data:
                        red_freq_df = pd.DataFrame(list(freq_data['red_frequency'].items()), 
                                                 columns=['号码', '频率'])
                        red_freq_df.to_excel(writer, sheet_name='红球频率', index=False)
                    
                    if 'blue_frequency' in freq_data:
                        blue_freq_df = pd.DataFrame(list(freq_data['blue_frequency'].items()), 
                                                  columns=['号码', '频率'])
                        blue_freq_df.to_excel(writer, sheet_name='蓝球频率', index=False)
                
                # 分析建议
                if 'recommendations' in analysis_result and analysis_result['recommendations']:
                    rec_df = pd.DataFrame({'建议': analysis_result['recommendations']})
                    rec_df.to_excel(writer, sheet_name='分析建议', index=False)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"导出分析Excel失败: {e}")
            raise
    
    def _create_chart_by_type(self, chart_type: str, chart_data: Dict[str, Any]):
        """根据类型创建图表"""
        try:
            lottery_type = chart_data.get('lottery_type', '双色球')
            
            if chart_type == 'frequency':
                # 创建频率图表
                frequency_data = chart_data.get('frequency_data', {})
                if not frequency_data:
                    # 生成模拟数据
                    if lottery_type == "双色球":
                        frequency_data = {
                            'red_balls': {str(i): np.random.randint(5, 25) for i in range(1, 34)},
                            'blue_balls': {str(i): np.random.randint(3, 15) for i in range(1, 17)}
                        }
                    else:
                        frequency_data = {
                            'front_balls': {str(i): np.random.randint(4, 20) for i in range(1, 36)},
                            'back_balls': {str(i): np.random.randint(3, 12) for i in range(1, 13)}
                        }
                
                return self.visualization.create_number_frequency_chart(
                    frequency_data, lottery_type, 'bar'
                )
            
            elif chart_type == 'trend':
                # 创建趋势图表
                history_data = chart_data.get('history_data', [])
                return self.visualization.create_trend_chart(
                    history_data, lottery_type, 'frequency'
                )
            
            elif chart_type == 'analysis':
                # 创建分析图表
                analysis_data = chart_data.get('analysis_data', {})
                return self.visualization.create_comprehensive_analysis_chart(analysis_data)
            
            else:
                return None
                
        except Exception as e:
            logger.error(f"创建图表失败: {e}")
            return None
    
    def _prepare_charts_data(self, lottery_type: str, analysis_result: Dict[str, Any], 
                           statistics_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备图表数据"""
        return {
            'lottery_type': lottery_type,
            'frequency_data': analysis_result.get('frequency', {}),
            'analysis_data': analysis_result,
            'statistics_data': statistics_data
        }
    
    def _create_comprehensive_pdf(self, filepath: Path, lottery_type: str,
                                prediction_data: Dict[str, Any], analysis_result: Dict[str, Any],
                                statistics_data: Dict[str, Any], charts_data: Dict[str, Any]):
        """创建综合PDF报告"""
        try:
            doc = SimpleDocTemplate(str(filepath), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # 标题页
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                alignment=1
            )
            story.append(Paragraph(f"🎯 {lottery_type}综合分析报告", title_style))
            story.append(Spacer(1, 30))
            
            # 报告概述
            overview_text = f"""
            本报告包含了{lottery_type}的全面数据分析，涵盖预测性能、历史数据分析、
            统计学指标等多个方面。通过AI技术和统计学方法，为用户提供专业的
            数据洞察和参考建议。
            
            报告生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
            """
            story.append(Paragraph("报告概述", styles['Heading2']))
            story.append(Paragraph(overview_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # 添加预测性能部分
            if prediction_data.get('model_performance'):
                story.append(Paragraph("预测性能分析", styles['Heading2']))
                # 添加性能表格...
            
            # 添加数据分析部分
            if analysis_result:
                story.append(Paragraph("历史数据分析", styles['Heading2']))
                # 添加分析内容...
            
            # 添加图表
            try:
                if charts_data:
                    story.append(Paragraph("可视化图表", styles['Heading2']))
                    
                    # 创建并添加图表
                    fig = self.visualization.create_comprehensive_analysis_chart(analysis_result)
                    temp_chart_path = self.export_dir / "images" / f"temp_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    fig.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
                    
                    chart_img = Image(str(temp_chart_path), width=7*inch, height=5*inch)
                    story.append(chart_img)
                    
                    plt.close(fig)
                    temp_chart_path.unlink(missing_ok=True)
                    
            except Exception as e:
                logger.error(f"添加图表到综合PDF失败: {e}")
            
            # 构建PDF
            doc.build(story)
            
        except Exception as e:
            logger.error(f"创建综合PDF失败: {e}")
            raise
    
    def get_export_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取导出历史记录"""
        try:
            # 从数据库获取导出历史
            conn = self.db_adapter.db_manager.get_connection()
            cursor = conn.execute('''
                SELECT * FROM export_history 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            history = []
            for row in cursor:
                record = dict(row)
                if record['export_params']:
                    record['export_params'] = json.loads(record['export_params'])
                history.append(record)
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"获取导出历史失败: {e}")
            return []
    
    def clean_old_exports(self, days: int = 30):
        """清理旧的导出文件"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 获取旧的导出记录
            conn = self.db_adapter.db_manager.get_connection()
            cursor = conn.execute('''
                SELECT file_path FROM export_history 
                WHERE created_at < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_count = 0
            for row in cursor:
                file_path = Path(row['file_path'])
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"删除文件失败 {file_path}: {e}")
            
            # 删除数据库记录
            cursor = conn.execute('''
                DELETE FROM export_history 
                WHERE created_at < ?
            ''', (cutoff_date.isoformat(),))
            
            conn.commit()
            conn.close()
            
            logger.info(f"清理完成: 删除 {deleted_count} 个文件, {cursor.rowcount} 条记录")
            
        except Exception as e:
            logger.error(f"清理旧导出文件失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 创建导出器实例
    exporter = DataExporter()
    
    # 导出预测数据
    try:
        excel_file = exporter.export_prediction_data('excel', 'deepseek-chat', '双色球', 30)
        print(f"Excel文件已导出: {excel_file}")
        
        pdf_file = exporter.export_analysis_report('双色球', '最近100期', 'pdf', True)
        print(f"PDF报告已导出: {pdf_file}")
        
        comprehensive_file = exporter.export_comprehensive_report('双色球', 'deepseek-chat')
        print(f"综合报告已导出: {comprehensive_file}")
        
    except Exception as e:
        print(f"导出失败: {e}")

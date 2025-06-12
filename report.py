#!/usr/bin/env python3
"""
Advanced Data Analysis Report Generator
=====================================

This module provides comprehensive reporting capabilities for data analysis results,
including statistical summaries, visualizations, and insights generation.

Author: Data Analysis Tool Team
Date: 2025-06-08
Version: 2.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import other modules from the project
try:
    from statistical_analysis import StatisticalAnalyzer
except ImportError:
    StatisticalAnalyzer = None
    # Removed print statement to reduce console noise

try:
    from visualization import AdvancedVisualizer
except ImportError:
    AdvancedVisualizer = None
    # Removed print statement to reduce console noise

try:
    from advanced_insights import AdvancedInsightGenerator
except ImportError:
    AdvancedInsightGenerator = None
    # Removed print statement to reduce console noise

@dataclass
class ReportConfig:
    """Configuration class for report generation"""
    title: str = "Data Analysis Report"
    author: str = "Data Analysis Tool"
    date: str = None
    include_summary: bool = True
    include_visualizations: bool = True
    include_statistics: bool = True
    include_insights: bool = True
    output_format: str = "html"  # html, pdf, markdown
    theme: str = "default"
    
    def __post_init__(self):
        if self.date is None:
            self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class ReportGenerator:
    """Advanced report generator for data analysis results"""
    
    def __init__(self, config: ReportConfig = None):
        """
        Initialize the report generator
        
        Args:
            config: ReportConfig object with report settings
        """
        self.config = config or ReportConfig()
        self.data = None
        self.analysis_results = {}
        self.visualizations = []
        self.insights = []
        
        # Initialize analyzers safely
        self.stats_analyzer = None
        self.visualizer = None
        self.insight_generator = None
        
        # Only initialize if the class is available and needed
        if StatisticalAnalyzer:
            try:
                # Initialize with empty DataFrame for now, will be updated when data is loaded
                pass  # Will initialize when needed
            except Exception:
                pass  # Silently fail
                
        if AdvancedVisualizer:
            try:
                # Initialize with empty DataFrame for now, will be updated when data is loaded
                pass  # Will initialize when needed
            except Exception:
                pass  # Silently fail
                
        if AdvancedInsightGenerator:
            try:
                # Initialize with empty DataFrame for now, will be updated when data is loaded
                pass  # Will initialize when needed
            except Exception:
                pass  # Silently fail
    
    def load_data(self, data_source):
        """
        Load data from various sources
        
        Args:
            data_source: Can be DataFrame, file path, or dict
        """
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source
        elif isinstance(data_source, str):
            # File path
            if data_source.endswith('.csv'):
                self.data = pd.read_csv(data_source)
            elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
                self.data = pd.read_excel(data_source)
            elif data_source.endswith('.json'):
                self.data = pd.read_json(data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        elif isinstance(data_source, dict):
            self.data = pd.DataFrame(data_source)
        else:
            raise ValueError("Unsupported data source type")
        
        # Initialize analyzers with actual data
        if StatisticalAnalyzer and self.data is not None:
            try:
                self.stats_analyzer = StatisticalAnalyzer(self.data)
            except Exception:
                pass  # Silently fail
                
        if AdvancedVisualizer and self.data is not None:
            try:
                self.visualizer = AdvancedVisualizer(self.data)
            except Exception:
                pass  # Silently fail
                
        if AdvancedInsightGenerator and self.data is not None:
            try:
                self.insight_generator = AdvancedInsightGenerator(self.data)
            except Exception:
                pass  # Silently fail
        
        # Only log in debug mode or when explicitly requested
        # print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
    
    def generate_data_summary(self) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        if self.data is None:
            return {}
        
        summary = {
            'basic_info': {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'memory_usage': self.data.memory_usage(deep=True).sum(),
                'data_types': self.data.dtypes.value_counts().to_dict()
            },
            'missing_data': {
                'total_missing': self.data.isnull().sum().sum(),
                'missing_by_column': self.data.isnull().sum().to_dict(),
                'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
            },
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = self.data[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols:
                cat_summary[col] = {
                    'unique_values': self.data[col].nunique(),
                    'most_frequent': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                    'frequency_count': self.data[col].value_counts().head(5).to_dict()
                }
            summary['categorical_summary'] = cat_summary
        
        return summary
    
    def generate_statistical_analysis(self) -> Dict[str, Any]:
        """Generate statistical analysis results"""
        if self.data is None:
            return {}
            
        # Use basic pandas operations for analysis instead of requiring external modules
        try:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            results = {
                'basic_stats': {},
                'correlations': {},
                'data_quality': {}
            }
            
            # Basic statistics
            if len(numeric_cols) > 0:
                results['basic_stats'] = self.data[numeric_cols].describe().to_dict()
            
            # Correlation matrix
            if len(numeric_cols) > 1:
                results['correlations'] = self.data[numeric_cols].corr().to_dict()
            
            # Data quality metrics
            results['data_quality'] = {
                'missing_values': self.data.isnull().sum().to_dict(),
                'duplicate_rows': self.data.duplicated().sum(),
                'data_types': self.data.dtypes.astype(str).to_dict()
            }
            
            return results
            
        except Exception as e:
            # Only print in debug mode
            # print(f"Statistical analysis error: {e}")
            return {}
    
    def generate_visualizations(self) -> List[Dict[str, Any]]:
        """Generate visualizations for the report"""
        visualizations = []
        
        if self.data is None:
            return visualizations
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Data overview visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Data Overview', fontsize=16)
            
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
            
            # Missing data heatmap
            if self.data.isnull().sum().sum() > 0:
                missing_data = self.data.isnull()
                if len(missing_data.columns) > 20:  # Too many columns for heatmap
                    missing_summary = self.data.isnull().sum().sort_values(ascending=False)[:20]
                    missing_summary.plot(kind='bar', ax=axes[0,0])
                    axes[0,0].set_title('Top 20 Missing Data Columns')
                    axes[0,0].tick_params(axis='x', rotation=45)
                else:
                    sns.heatmap(missing_data, ax=axes[0,0], cbar=True, yticklabels=False)
                    axes[0,0].set_title('Missing Data Pattern')
            else:
                axes[0,0].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                              transform=axes[0,0].transAxes, fontsize=14)
                axes[0,0].set_title('Missing Data Status')
            
            # Data types distribution
            dtype_counts = self.data.dtypes.value_counts()
            if len(dtype_counts) > 0:
                axes[0,1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
                axes[0,1].set_title('Data Types Distribution')
            
            # Numeric columns distribution
            if len(numeric_cols) > 0:
                if len(numeric_cols) <= 4:  # Show individual histograms
                    for i, col in enumerate(numeric_cols[:4]):
                        self.data[col].hist(ax=axes[1,0], alpha=0.7, label=col, bins=20)
                    axes[1,0].legend()
                    axes[1,0].set_title('Numeric Columns Distribution')
                else:  # Show summary statistics
                    summary_stats = self.data[numeric_cols].describe().T
                    y_pos = np.arange(len(summary_stats))
                    axes[1,0].barh(y_pos, summary_stats['mean'])
                    axes[1,0].set_yticks(y_pos)
                    axes[1,0].set_yticklabels(summary_stats.index)
                    axes[1,0].set_title('Mean Values of Numeric Columns')
            else:
                axes[1,0].text(0.5, 0.5, 'No Numeric Data', ha='center', va='center', 
                              transform=axes[1,0].transAxes, fontsize=14)
                axes[1,0].set_title('Numeric Data Status')
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                corr_matrix = self.data[numeric_cols].corr()
                mask = np.triu(np.ones_like(corr_matrix))  # Show only lower triangle
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                           center=0, ax=axes[1,1], fmt='.2f', square=True)
                axes[1,1].set_title('Correlation Matrix')
            else:
                axes[1,1].text(0.5, 0.5, 'Insufficient Numeric\nColumns for Correlation', 
                              ha='center', va='center', transform=axes[1,1].transAxes, fontsize=14)
                axes[1,1].set_title('Correlation Status')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = f"report_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Convert to base64 for embedding in HTML
            import base64
            with open(viz_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            
            # Clean up the temporary file
            import os
            try:
                os.remove(viz_path)
            except:
                pass  # If file removal fails, continue anyway
            
            visualizations.append({
                'title': 'Data Overview',
                'type': 'overview',
                'path': viz_path,
                'base64_data': img_data,
                'description': 'Comprehensive overview of the dataset including missing data patterns, data types, distributions, and correlations'
            })
            
        except Exception as e:
            # Create a simple text-based visualization info
            visualizations.append({
                'title': 'Data Overview',
                'type': 'text',
                'path': None,
                'base64_data': None,
                'description': f'Visualization generation failed: {str(e)}. Dataset has {self.data.shape[0]} rows and {self.data.shape[1]} columns.'
            })
        
        return visualizations
    
    def generate_insights(self) -> List[Dict[str, Any]]:
        """Generate automated insights from data analysis"""
        insights = []
        
        if self.data is None:
            return insights
        
        try:
            # Data quality insights
            missing_pct = (self.data.isnull().sum() / len(self.data) * 100)
            high_missing_cols = missing_pct[missing_pct > 20].index.tolist()
            
            if high_missing_cols:
                insights.append({
                    'category': 'Data Quality',
                    'title': 'High Missing Data Warning',
                    'description': f"Columns with >20% missing data: {', '.join(high_missing_cols)}",
                    'severity': 'warning',
                    'recommendation': 'Consider data imputation or column removal strategies'
                })
            
            # Statistical insights
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = self.data[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                if high_corr_pairs:
                    insights.append({
                        'category': 'Statistical',
                        'title': 'High Correlation Detected',
                        'description': f"Found {len(high_corr_pairs)} pairs with correlation >0.8",
                        'severity': 'info',
                        'recommendation': 'Consider multicollinearity in modeling'
                    })
            
            # Data distribution insights
            for col in numeric_cols:
                skewness = self.data[col].skew()
                if abs(skewness) > 2:
                    insights.append({
                        'category': 'Distribution',
                        'title': f'Highly Skewed Distribution: {col}',
                        'description': f"Skewness: {skewness:.2f}",
                        'severity': 'info',
                        'recommendation': 'Consider data transformation (log, sqrt, etc.)'
                    })
            
        except Exception as e:
            # Only print in debug mode
            # print(f"Insight generation error: {e}")
            pass
        
        return insights
    
    def generate_html_report(self) -> str:
        """Generate HTML report"""
        # Generate all components
        summary = self.generate_data_summary()
        stats = self.generate_statistical_analysis()
        visualizations = self.generate_visualizations()
        insights = self.generate_insights()
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.config.title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #007acc;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #007acc;
                    margin: 0;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .section h2 {{
                    color: #333;
                    border-left: 4px solid #007acc;
                    padding-left: 15px;
                }}
                .insight-card {{
                    background-color: #f8f9fa;
                    border-left: 4px solid #007acc;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                .warning {{ border-left-color: #ff6b6b; }}
                .info {{ border-left-color: #4ecdc4; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #007acc;
                    color: white;
                }}
                .visualization {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .visualization img {{
                    max-width: 100%;
                    border-radius: 5px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{self.config.title}</h1>
                    <p><strong>Author:</strong> {self.config.author}</p>
                    <p><strong>Generated:</strong> {self.config.date}</p>
                </div>
        """
        
        # Add data summary section
        if self.config.include_summary and summary:
            html_content += f"""
                <div class="section">
                    <h2>📊 Data Summary</h2>
                    <h3>Basic Information</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Rows</td><td>{summary['basic_info']['rows']:,}</td></tr>
                        <tr><td>Total Columns</td><td>{summary['basic_info']['columns']:,}</td></tr>
                        <tr><td>Memory Usage</td><td>{summary['basic_info']['memory_usage'] / 1024 / 1024:.2f} MB</td></tr>
                        <tr><td>Missing Values</td><td>{summary['missing_data']['total_missing']:,}</td></tr>
                    </table>
                </div>
            """
        
        # Add insights section
        if self.config.include_insights and insights:
            html_content += """
                <div class="section">
                    <h2>💡 Key Insights</h2>
            """
            for insight in insights:
                severity_class = insight.get('severity', 'info')
                html_content += f"""
                    <div class="insight-card {severity_class}">
                        <h4>{insight['title']}</h4>
                        <p>{insight['description']}</p>
                        <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                    </div>
                """
            html_content += "</div>"
        
        # Add visualizations section
        if self.config.include_visualizations and visualizations:
            html_content += """
                <div class="section">
                    <h2>📈 Visualizations</h2>
            """
            for viz in visualizations:
                if viz.get('base64_data'):
                    html_content += f"""
                        <div class="visualization">
                            <h3>{viz['title']}</h3>
                            <img src="data:image/png;base64,{viz['base64_data']}" alt="{viz['title']}" style="max-width: 100%; height: auto;">
                            <p>{viz['description']}</p>
                        </div>
                    """
                else:
                    html_content += f"""
                        <div class="visualization">
                            <h3>{viz['title']}</h3>
                            <div style="padding: 20px; background-color: #f8f9fa; border-radius: 5px; text-align: center;">
                                <p>{viz['description']}</p>
                            </div>
                        </div>
                    """
            html_content += "</div>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def save_report(self, filename: str = None) -> str:
        """Save the report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_analysis_report_{timestamp}.html"
        
        html_content = self.generate_html_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Only print in debug mode
        # print(f"Report saved successfully: {filename}")
        return filename
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics for quick overview"""
        if self.data is None:
            return {}
        
        return {
            'dataset_shape': self.data.shape,
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_data_percentage': (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100,
            'numeric_columns': len(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(self.data.select_dtypes(include=['datetime64']).columns)
        }

def create_quick_report(data_source, title: str = "Quick Data Report") -> str:
    """
    Create a quick report with minimal configuration
    
    Args:
        data_source: Data source (DataFrame, file path, etc.)
        title: Report title
    
    Returns:
        str: Path to generated report file
    """
    config = ReportConfig(title=title)
    generator = ReportGenerator(config)
    generator.load_data(data_source)
    return generator.save_report()

def create_detailed_report(data_source, config: ReportConfig = None) -> str:
    """
    Create a detailed report with full analysis
    
    Args:
        data_source: Data source (DataFrame, file path, etc.)
        config: Report configuration
    
    Returns:
        str: Path to generated report file
    """
    if config is None:
        config = ReportConfig(
            title="Detailed Data Analysis Report",
            include_summary=True,
            include_visualizations=True,
            include_statistics=True,
            include_insights=True
        )
    
    generator = ReportGenerator(config)
    generator.load_data(data_source)
    return generator.save_report()

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Data Analysis Report Generator v2.0")
    print("=" * 50)
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'sales': np.random.normal(1000, 200, 100),
        'profit': np.random.normal(150, 50, 100),
        'customers': np.random.poisson(50, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D')
    })
    
    # Add some missing values for testing
    sample_data.loc[np.random.choice(sample_data.index, 10), 'profit'] = np.nan
    
    print("Sample data created for testing")
    print(f"Data shape: {sample_data.shape}")
    
    # Generate quick report
    try:
        report_path = create_quick_report(sample_data, "Sample Data Analysis Report")
        print(f"Quick report generated: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
    
    print("Report generation completed!")

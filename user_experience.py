"""
User Experience Enhancement Module
- Progress tracking
- Error handling and user guides
- Result saving/export functionality
- Settings management
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for serializing NumPy types to JSON"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return super(NumpyEncoder, self).default(obj)


class ProgressTracker:
    """Progress tracking class"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
    def update(self, step: int, message: str = ""):
        """Update progress"""
        self.current_step = step
        progress = self.current_step / self.total_steps
        self.progress_bar.progress(progress)
        
        if message:
            self.status_text.text(f"{self.description}: {message} ({self.current_step}/{self.total_steps})")
        else:
            self.status_text.text(f"{self.description}: {self.current_step}/{self.total_steps} completed")
    
    def complete(self, message: str = "Completed!"):
        """Mark as completed"""
        self.progress_bar.progress(1.0)
        self.status_text.text(message)
    
    def error(self, message: str = "Error occurred"):
        """Display error"""
        self.status_text.error(message)


class ErrorHandler:
    """Error handling class"""
    
    @staticmethod
    def handle_error(func):
        """Error handling decorator"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error occurred: {str(e)}"
                st.error(error_msg)
                
                with st.expander("Detailed Error Information"):
                    st.code(traceback.format_exc())
                
                # User guide
                ErrorHandler.show_error_guide(e)
                
                return None
        return wrapper
    
    @staticmethod
    def show_error_guide(error: Exception):
        """Show error-specific user guide"""
        error_type = type(error).__name__
        
        guides = {
            'FileNotFoundError': "File not found. Please check the file path.",
            'ValueError': "Invalid input value. Please check the data format.",
            'KeyError': "Required column not found. Please check the data structure.",
            'MemoryError': "Insufficient memory. Please use smaller dataset or apply filtering.",
            'TypeError': "Data type mismatch. Please check numeric/string data types."
        }
        
        if error_type in guides:
            st.info(f"💡 Solution: {guides[error_type]}")
        else:
            st.info("💡 General solution: Please check your data and try again.")


class DataExporter:
    """Data export class"""
    
    @staticmethod
    def to_csv(df: pd.DataFrame) -> str:
        """Convert DataFrame to CSV"""
        return df.to_csv(index=False)
    
    @staticmethod
    def to_excel(df: pd.DataFrame) -> bytes:
        """Convert DataFrame to Excel"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
        return output.getvalue()
    
    @staticmethod
    def create_download_link(data: Any, filename: str, file_type: str = "text/csv") -> str:
        """Create download link"""
        if isinstance(data, str):
            b64 = base64.b64encode(data.encode()).decode()
        else:
            b64 = base64.b64encode(data).decode()
        
        return f'<a href="data:{file_type};base64,{b64}" download="{filename}">💾 Download {filename}</a>'
    
    @staticmethod
    def export_report(report_data: Dict[str, Any]) -> str:
        """Export analysis report to JSON - supports NumPy types"""
        return json.dumps(report_data, ensure_ascii=False, indent=2, cls=NumpyEncoder)


class SettingsManager:
    """Settings management class"""
    
    @staticmethod
    def save_settings(settings: Dict[str, Any]):
        """Save settings - supports NumPy types"""
        settings_json = json.dumps(settings, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        st.session_state['saved_settings'] = settings
        
        # Provide download link
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eda_settings_{timestamp}.json"
        
        st.download_button(
            label="⚙️ Download Settings File",
            data=settings_json,
            file_name=filename,
            mime="application/json"
        )
    
    @staticmethod
    def load_settings(uploaded_file) -> Optional[Dict[str, Any]]:
        """Load settings"""
        if uploaded_file is not None:
            try:
                settings = json.load(uploaded_file)
                st.session_state['saved_settings'] = settings
                st.success("✅ Settings loaded successfully.")
                return settings
            except Exception as e:
                st.error(f"Error loading settings file: {str(e)}")
                return None
        return None
    
    @staticmethod
    def apply_settings(settings: Dict[str, Any]):
        """Apply settings"""
        for key, value in settings.items():
            if key in st.session_state:
                st.session_state[key] = value


def safe_json_convert(obj):
    """Safely convert data to JSON serializable format"""
    if isinstance(obj, dict):
        return {key: safe_json_convert(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return obj.fillna('null').to_dict()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj) or pd.isnull(obj):
        return None
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj


def render_data_export_ui(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None):
    """Data export UI"""
    st.header("💾 Data Export")
    
    export_tabs = st.tabs(["Data Export", "Report Export", "Settings Management"])
    
    with export_tabs[0]:
        st.subheader("📊 Export Processed Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as CSV"):
                csv_data = DataExporter.to_csv(df)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_data_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export as Excel"):
                excel_data = DataExporter.to_excel(df)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_data_{timestamp}.xlsx"
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Data preview
        st.write("**Data Preview**")
        st.dataframe(df.head(10))
        st.info(f"Total {len(df)} rows, {len(df.columns)} columns")
    
    with export_tabs[1]:
        st.subheader("📑 Export Analysis Report")
        
        # Collect analysis results from session state
        collected_analysis = {}
        
        # Preprocessing results
        if 'preprocessor' in st.session_state:
            try:
                preprocessor = st.session_state['preprocessor']
                collected_analysis['preprocessing'] = preprocessor.get_preprocessing_summary()
            except:
                pass
        
        # Always include basic data information (safe JSON conversion)
        try:
            # Safely calculate memory usage
            memory_usage = float(df.memory_usage(deep=True).sum() / 1024 / 1024)
            
            # Safely convert data types
            data_types = {str(k): int(v) for k, v in df.dtypes.value_counts().items()}
            
            # Safely convert missing data
            missing_by_column = {str(k): int(v) for k, v in df.isnull().sum().items()}
            missing_percentage = {str(k): float(v) for k, v in (df.isnull().sum() / len(df) * 100).items()}
            
            collected_analysis['data_summary'] = {
                "basic_info": {
                    "rows": int(len(df)),
                    "columns": int(len(df.columns)),
                    "memory_usage_mb": memory_usage,
                    "data_types": data_types
                },
                "missing_data": {
                    "total_missing": int(df.isnull().sum().sum()),
                    "missing_by_column": missing_by_column,
                    "missing_percentage": missing_percentage
                },
                "numeric_summary": {},
                "categorical_summary": {}
            }
            
            # Safely convert numeric summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_summary = {}
                describe_df = df[numeric_cols].describe()
                for col in describe_df.columns:
                    numeric_summary[str(col)] = {
                        str(stat): float(value) if not pd.isna(value) else None 
                        for stat, value in describe_df[col].items()
                    }
                collected_analysis['data_summary']['numeric_summary'] = numeric_summary
                
        except Exception as e:
            st.warning(f"Error occurred while generating data summary: {str(e)}")
            # Save basic information only
            collected_analysis['data_summary'] = {
                "basic_info": {
                    "rows": int(len(df)),
                    "columns": int(len(df.columns))
                }
            }
        
        # Categorical summary (safe conversion)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols:
                try:
                    # Safely handle mode
                    mode_values = df[col].mode()
                    most_frequent = str(mode_values.iloc[0]) if not mode_values.empty else None
                    
                    # Safely convert frequency counts
                    frequency_count = {str(k): int(v) for k, v in df[col].value_counts().head(5).items()}
                    
                    cat_summary[str(col)] = {
                        'unique_values': int(df[col].nunique()),
                        'most_frequent': most_frequent,
                        'frequency_count': frequency_count
                    }
                except Exception as col_error:
                    # Skip individual column errors
                    continue
                    
            if 'data_summary' in collected_analysis:
                collected_analysis['data_summary']['categorical_summary'] = cat_summary
        
        # Include other analysis results from session state
        if analysis_results:
            collected_analysis.update(analysis_results)
        
        # Always enable report generation
        if collected_analysis:
            try:
                # Generate report (using safe conversion)
                report = {
                    "generated_at": datetime.now().isoformat(),
                    "data_info": {
                        "rows": int(len(df)),
                        "columns": int(len(df.columns)),
                        "numeric_columns": [str(col) for col in df.select_dtypes(include=[np.number]).columns],
                        "categorical_columns": [str(col) for col in df.select_dtypes(include=['object', 'category']).columns]
                    },
                    "analysis_results": safe_json_convert(collected_analysis)
                }
                
                report_json = DataExporter.export_report(report)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_report_{timestamp}.json"
                
                st.download_button(
                    label="📥 Download Report (JSON)",
                    data=report_json,
                    file_name=filename,
                    mime="application/json"
                )
                
                # Report preview
                with st.expander("Report Preview"):
                    st.json(report)
                    
            except Exception as e:
                st.error(f"Error occurred during report generation: {str(e)}")
                st.info("Generating basic report.")
                
                # Generate basic report on error
                basic_report = {
                    "generated_at": datetime.now().isoformat(),
                    "data_info": {
                        "rows": int(len(df)),
                        "columns": int(len(df.columns))
                    },
                    "error": "Full report generation failed, basic info only"
                }
                
                report_json = DataExporter.export_report(basic_report)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"basic_report_{timestamp}.json"
                
                st.download_button(
                    label="📥 Download Basic Report (JSON)",
                    data=report_json,
                    file_name=filename,
                    mime="application/json"
                )
                
            # Display included analysis results
            st.write("**Included Analysis Results:**")
            analysis_types = []
            if 'preprocessing' in collected_analysis:
                analysis_types.append("✅ Data preprocessing results")
            if 'data_summary' in collected_analysis:
                analysis_types.append("✅ Basic data summary")
            if analysis_results:
                analysis_types.append("✅ Additional analysis results")
            
            for analysis_type in analysis_types:
                st.write(f"• {analysis_type}")
                
            if not analysis_types:
                st.info("💡 To perform more analysis, visit these tabs:")
                st.write("• 📊 Statistical Analysis: Statistical tests and correlation analysis")
                st.write("• 🚀 Advanced Analysis: Time series, text, and modeling analysis")
                st.write("• 🔍 Insights: Automatic insight generation")
        else:
            st.info("Basic data summary can be included in the report.") 
            
            # Enable basic report generation
            basic_report = {
                "generated_at": datetime.now().isoformat(),
                "data_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
                },
                "basic_summary": collected_analysis.get('data_summary', {})
            }
            
            report_json = DataExporter.export_report(basic_report)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"basic_report_{timestamp}.json"
            
            st.download_button(
                label="📥 Download Basic Report (JSON)",
                data=report_json,
                file_name=filename,
                mime="application/json"
            )
    
    with export_tabs[2]:
        st.subheader("⚙️ Settings Management")
        
        # Save current settings
        if st.button("Save Current Settings"):
            current_settings = {
                "correlation_threshold": st.session_state.get('correlation_threshold', 0.7),
                "outlier_contamination": st.session_state.get('outlier_contamination', 0.1),
                "visualization_settings": st.session_state.get('visualization_settings', {}),
                "preprocessing_options": st.session_state.get('preprocessing_options', {})
            }
            SettingsManager.save_settings(current_settings)
        
        # Load settings
        uploaded_settings = st.file_uploader(
            "Upload Settings File",
            type=['json'],
            help="Upload a previously saved settings file."
        )
        
        if uploaded_settings:
            settings = SettingsManager.load_settings(uploaded_settings)
            if settings and st.button("Apply Settings"):
                SettingsManager.apply_settings(settings)
                st.rerun()


def show_help_guide():
    """Display help guide"""
    with st.expander("📚 User Guide", expanded=False):
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        1. **Upload Data**: Upload CSV or Excel files from the left sidebar.
        2. **Data Preprocessing**: Clean your data in the 'Data Preprocessing' tab.
        3. **Exploratory Analysis**: Explore your data in 'Overview' and 'Visualization' tabs.
        4. **Statistical Analysis**: Perform in-depth analysis in the 'Statistical Analysis' tab.
        5. **Generate Insights**: Check automatically generated insights in the 'Insights' tab.
        6. **Create Report**: Generate final analysis report in the 'Report' tab.
        
        ### 💡 Key Features
        
        - **Automatic Data Type Conversion**: Automatically converts numeric strings to numbers
        - **Outlier Detection**: Detect and handle outliers using IQR or Z-Score methods
        - **Missing Value Handling**: Handle missing values with various methods
        - **Advanced Visualization**: 3D plots, Pair plots, Q-Q plots, etc.
        - **Statistical Testing**: t-test, ANOVA, chi-square test, etc.
        - **Predictive Modeling**: Linear regression and clustering
        """)


def show_data_quality_check(df: pd.DataFrame):
    """Display data quality check"""
    with st.expander("🔍 Data Quality Check", expanded=True):
        quality_issues = []
        
        # Check missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            quality_issues.append(f"Columns with missing values: {', '.join(missing_cols)}")
        
        # Check duplicate data
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"Duplicate rows: {duplicates}")
        
        # Check data types
        object_cols = df.select_dtypes(include=['object']).columns
        numeric_like = []
        for col in object_cols:
            try:
                pd.to_numeric(df[col], errors='coerce')
                numeric_like.append(col)
            except:
                pass
        
        if numeric_like:
            quality_issues.append(f"String columns convertible to numeric: {', '.join(numeric_like)}")
        
        # Display results
        if quality_issues:
            st.warning("⚠️ Data quality issues found:")
            for issue in quality_issues:
                st.write(f"• {issue}")
            st.info("💡 You can resolve these issues in the 'Data Preprocessing' tab.")
        else:
            st.success("✅ Data quality looks good!")

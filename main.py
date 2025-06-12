import streamlit as st
import pandas as pd
import numpy as np

from overview import render_overview
import visualization as vis
# Import new report module
from report import ReportGenerator, ReportConfig, create_quick_report, create_detailed_report

from data_preprocessing import render_preprocessing_ui
from statistical_analysis import render_statistical_analysis
from user_experience import (
    ProgressTracker, ErrorHandler, render_data_export_ui, 
    show_help_guide, show_data_quality_check
)
from performance_optimization import (
    render_performance_settings, optimize_large_dataset,
    MemoryMonitor
)
from advanced_analysis import render_advanced_analysis
from advanced_insights import render_advanced_insights

# Page configuration
st.set_page_config(page_title="Auto EDA & Insight", layout="wide")

# Global styles
import seaborn as sns
sns.set_style("whitegrid")

# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'date_cols' not in st.session_state:
    st.session_state.date_cols = []

st.title("Auto EDA & Insight Platform")

# Show help guide
show_help_guide()

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])
if not uploaded_file:
    st.info("Please upload a file from the sidebar first.")
    st.stop()

# Data loading with progress indicator
with st.spinner("Loading data..."):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
        
        # Check for empty dataframe
        if df.empty:
            st.error("❌ Uploaded file is empty.")
            st.stop()
        
        # Save to session state
        st.session_state.df_original = df.copy()
        st.session_state.df_filtered = df.copy()
        
        st.success(f"✅ Data loaded successfully! ({len(df):,} rows × {len(df.columns)} columns)")
        
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='cp949')
            if df.empty:
                st.error("❌ Uploaded file is empty.")
                st.stop()
            st.session_state.df_original = df.copy()
            st.session_state.df_filtered = df.copy()
            st.success(f"✅ Data loaded successfully! ({len(df):,} rows × {len(df.columns)} columns)")
        except Exception as e:
            st.error(f"❌ Encoding error: {str(e)}. Please use UTF-8 encoding.")
            st.stop()
    except pd.errors.EmptyDataError:
        st.error("❌ Empty file. Please upload a file with data.")
        st.stop()
    except MemoryError:
        st.error("❌ Insufficient memory: File is too large. Please split into smaller files and upload.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Data loading failed: {str(e)}")
        st.stop()

# Automatic date column recognition
date_cols = []
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            date_cols.append(col)
        except:
            pass

# Save to session state
st.session_state.date_cols = date_cols
st.session_state.df_filtered = df.copy()

# Data quality check
show_data_quality_check(df)

# Tab configuration
tabs = st.tabs([
    "Data Preprocessing", "Overview", "Visualization", 
    "Statistical Analysis", "Advanced Analysis", "Insights", 
    "Report", "Export & Settings"
])

# Get data from session state
df_filtered = st.session_state.df_filtered.copy()
date_cols = st.session_state.date_cols

# Data preprocessing
with tabs[0]:
    try:
        df_preprocessed = render_preprocessing_ui(df_filtered)
        if df_preprocessed is not None and not df_preprocessed.empty:
            st.session_state.df_filtered = df_preprocessed.copy()
            df_filtered = df_preprocessed.copy()
        else:
            st.warning("Preprocessing result is empty. Using original data.")
    except Exception as e:
        st.error(f"Error during data preprocessing: {str(e)}")
        st.info("Continuing with original data.")

# Overview
with tabs[1]:
    try:
        df_overview = render_overview(df_filtered, date_cols)
        if df_overview is not None and not df_overview.empty:
            st.session_state.df_filtered = df_overview.copy()
            df_filtered = df_overview.copy()
    except Exception as e:
        st.error(f"Error in Overview tab: {str(e)}")
        st.info("Continuing with current data.")

# Visualization
with tabs[2]:
    try:
        vis.render_all_visualizations(df_filtered)
    except Exception as e:
        st.error(f"Error during visualization generation: {str(e)}")

# Statistical analysis
with tabs[3]:
    try:
        render_statistical_analysis(df_filtered)
        # Mark that statistical analysis has been performed
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        st.session_state.analysis_results['statistical_analysis_performed'] = True
    except Exception as e:
        st.error(f"Error during statistical analysis: {str(e)}")

# Advanced analysis
with tabs[4]:
    try:
        render_advanced_analysis(df_filtered)
        # Mark that advanced analysis has been performed
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        st.session_state.analysis_results['advanced_analysis_performed'] = True
    except Exception as e:
        st.error(f"Error during advanced analysis: {str(e)}")

# Insights
with tabs[5]:
    try:
        render_advanced_insights(df_filtered)
        # Mark that insights have been generated
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        st.session_state.analysis_results['insights_generated'] = True
    except Exception as e:
        st.error(f"Error during insight generation: {str(e)}")

# Report
with tabs[6]:
    try:
        st.header("📊 Analysis Report Generation")
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Report Settings")
            report_title = st.text_input("Report Title", value="Data Analysis Report")
            report_author = st.text_input("Author", value="Data Analysis Tool")
            
            st.subheader("Sections to Include")
            include_summary = st.checkbox("Data Summary", value=True)
            include_visualizations = st.checkbox("Visualizations", value=True)
            include_statistics = st.checkbox("Statistical Analysis", value=True)
            include_insights = st.checkbox("Auto Insights", value=True)
        
        with col2:
            st.subheader("Report Preview")
            
            if st.button("🎯 Report Statistics Preview"):
                try:
                    generator = ReportGenerator()
                    generator.load_data(df_filtered)
                    stats = generator.generate_summary_stats()
                    
                    st.write("**Report Preview:**")
                    st.write(f"- Dataset size: {stats['dataset_shape'][0]:,} rows × {stats['dataset_shape'][1]} columns")
                    st.write(f"- Memory usage: {stats['memory_usage_mb']:.2f} MB")
                    st.write(f"- Missing data: {stats['missing_data_percentage']:.1f}%")
                    st.write(f"- Numeric columns: {stats['numeric_columns']} columns")
                    st.write(f"- Categorical columns: {stats['categorical_columns']} columns")
                except Exception as e:
                    st.error(f"Error generating report preview: {str(e)}")
        
        st.divider()
        
        # Generate report button (single unified button)
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("📋 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Generating report..."):
                    try:
                        config = ReportConfig(
                            title=report_title,
                            author=report_author,
                            include_summary=include_summary,
                            include_visualizations=include_visualizations,
                            include_statistics=include_statistics,
                            include_insights=include_insights
                        )
                        
                        generator = ReportGenerator(config)
                        generator.load_data(df_filtered)
                        report_path = generator.save_report()
                        
                        st.success(f"✅ Report generated successfully: {report_path}")
                        
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                        st.download_button(
                            label="📥 Download Report",
                            data=report_content,
                            file_name=report_path,
                            mime="text/html",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Report generation error: {str(e)}")
        

        
        # Report management
        st.divider()
        st.subheader("📁 Report Management")
        
        import os
        import glob
        
        try:
            report_files = glob.glob("data_analysis_report_*.html")
            if report_files:
                st.write("**Recent Reports:**")
                for report_file in sorted(report_files, reverse=True)[:5]:
                    file_size = os.path.getsize(report_file) / 1024
                    file_time = os.path.getmtime(report_file)
                    from datetime import datetime
                    file_time_str = datetime.fromtimestamp(file_time).strftime("%Y-%m-%d %H:%M:%S")
                    
                    col_file, col_download = st.columns([3, 1])
                    with col_file:
                        st.write(f"📄 {report_file} ({file_size:.1f} KB) - {file_time_str}")
                    with col_download:
                        with open(report_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.download_button(
                            label="⬇️",
                            data=content,
                            file_name=report_file,
                            mime="text/html",
                            key=f"download_{report_file}"
                        )
            else:
                st.info("No reports generated yet. Create your first report above!")
        except Exception as e:
            st.error(f"Error in report management: {str(e)}")
        
        # Advanced report options
        with st.expander("🔧 Advanced Report Options"):
            st.write("**Coming in future updates:**")
            st.write("- 📊 Custom chart selection")
            st.write("- 🎨 Report themes and styling")
            st.write("- 📧 Scheduled email reports")
            st.write("- 🔗 Report sharing links")
            st.write("- 📈 Comparative analysis reports")
            st.write("- 💾 PowerPoint/Word export")
    
    except Exception as e:
        st.error(f"Error in Report tab: {str(e)}")

# Export and settings
with tabs[7]:
    try:
        analysis_results = st.session_state.get('analysis_results', {})
        render_data_export_ui(df_filtered, analysis_results)
        
        st.divider()
        perf_settings = render_performance_settings()
        
        if st.session_state.get('optimize_memory', False):
            df_filtered = optimize_large_dataset(df_filtered, perf_settings)
            st.session_state['optimize_memory'] = False
    except Exception as e:
        st.error(f"Error in Export and Settings tab: {str(e)}")

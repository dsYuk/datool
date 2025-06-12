"""
Integrated Visualization Module
- Basic visualization functions
- Advanced visualization classes
- Interactive filtering
- Multivariate relationship visualization
- Statistical visualization
- Customization options
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Optional, Dict, Any

# --------------------- Basic Visualization Functions ---------------------

def plot_distribution_plotly(df: pd.DataFrame, column: str):
    """Distribution histogram plot"""
    fig = px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}")
    st.plotly_chart(fig, use_container_width=True, key=f"dist_{column}")

def plot_correlation_heatmap_plotly(df: pd.DataFrame):
    """Correlation heatmap plot"""
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")

def plot_grouped_bar_plotly(df: pd.DataFrame, group_col: str, value_col: str):
    """Grouped bar chart"""
    grouped = df.groupby(group_col)[value_col].mean().reset_index()
    fig = px.bar(grouped, x=group_col, y=value_col, title=f"Average {value_col} by {group_col}")
    st.plotly_chart(fig, use_container_width=True, key=f"grouped_bar_{group_col}_{value_col}")

def plot_time_series_plotly(df: pd.DataFrame, date_col: str, value_col: str):
    """Time series plot"""
    df[date_col] = pd.to_datetime(df[date_col])
    fig = px.line(df, x=date_col, y=value_col, title=f"{value_col} over Time")
    st.plotly_chart(fig, use_container_width=True, key=f"timeseries_{date_col}_{value_col}")

# --------------------- Categorical Visualization Functions ---------------------

def plot_group_count_by_category(df, value_col, category_col, ascending=False, key_suffix=""):
    """Count bar chart by category"""
    grouped = df.groupby(category_col)[value_col].count().reset_index()
    grouped = grouped.sort_values(by=value_col, ascending=ascending)
    fig = px.bar(grouped, x=category_col, y=value_col,
                 title=f"Count of {value_col} by {category_col} (Sort: {'Ascending' if ascending else 'Descending'})")
    st.plotly_chart(fig, use_container_width=True, key=f"count_bar_{category_col}_{value_col}{key_suffix}")

def plot_group_proportion_by_category(df, value_col, category_col, ascending=False, key_suffix=""):
    """Proportion bar chart by category"""
    grouped = df.groupby(category_col)[value_col].count().reset_index()
    grouped[value_col] = grouped[value_col] / grouped[value_col].sum() * 100
    grouped = grouped.sort_values(by=value_col, ascending=ascending)
    fig = px.bar(grouped, x=category_col, y=value_col,
                 title=f"Proportion (%) of {value_col} by {category_col} (Sort: {'Ascending' if ascending else 'Descending'})")
    st.plotly_chart(fig, use_container_width=True, key=f"prop_bar_{category_col}_{value_col}{key_suffix}")

def plot_category_pie(df, column: str, key_suffix=""):
    """Pie chart"""
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    fig = px.pie(counts, names=column, values='count', title=f"Pie Chart of {column}")
    st.plotly_chart(fig, use_container_width=True, key=f"pie_{column}{key_suffix}")

def plot_category_treemap(df, column: str, key_suffix=""):
    """Treemap"""
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    fig = px.treemap(counts, path=[column], values='count', title=f"Treemap of {column}")
    st.plotly_chart(fig, use_container_width=True, key=f"treemap_{column}{key_suffix}")

# --------------------- Advanced Visualization Classes ---------------------

class AdvancedVisualizer:
    """Advanced visualization class"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def create_interactive_scatter(self, x_col: str, y_col: str, 
                                 color_col: Optional[str] = None,
                                 size_col: Optional[str] = None,
                                 hover_data: Optional[List[str]] = None) -> go.Figure:
        """Create interactive scatter plot"""
        fig = px.scatter(
            self.df, 
            x=x_col, 
            y=y_col,
            color=color_col,
            size=size_col,
            hover_data=hover_data or self.df.columns.tolist(),
            title=f"{x_col} vs {y_col}"
        )
        
        # Add regression line option
        if x_col in self.numeric_cols and y_col in self.numeric_cols:
            # Calculate regression line
            x_data = self.df[x_col].dropna()
            y_data = self.df[y_col].dropna()
            
            # Use common indices
            common_idx = x_data.index.intersection(y_data.index)
            x_data = x_data[common_idx]
            y_data = y_data[common_idx]
            
            if len(x_data) > 0:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=p(x_data),
                        mode='lines',
                        name='Regression Line',
                        line=dict(color='red', dash='dash')
                    )
                )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode='closest'
        )
        
        return fig
    
    def create_3d_scatter(self, x_col: str, y_col: str, z_col: str,
                         color_col: Optional[str] = None) -> go.Figure:
        """Create 3D scatter plot"""
        fig = px.scatter_3d(
            self.df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}"
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
        return fig
    
    def create_pair_plot(self, columns: List[str], 
                        hue_col: Optional[str] = None) -> plt.Figure:
        """Create pair plot"""
        plot_df = self.df[columns].copy()
        
        if hue_col and hue_col in self.df.columns:
            plot_df[hue_col] = self.df[hue_col]
            g = sns.pairplot(plot_df, hue=hue_col, diag_kind='kde')
        else:
            g = sns.pairplot(plot_df, diag_kind='kde')
        
        return g.fig
    
    def create_qq_plot(self, column: str) -> go.Figure:
        """Create Q-Q plot"""
        data = self.df[column].dropna()
        
        # Calculate theoretical and sample quantiles
        sorted_data = np.sort(data)
        n = len(sorted_data)
        theoretical_quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
        
        fig = go.Figure()
        
        # Q-Q plot points
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                name='Data',
                marker=dict(color='blue', size=5)
            )
        )
        
        # Ideal line
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Distribution Line',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(
            title=f"Q-Q Plot: {column}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles"
        )
        
        return fig
    
    def create_residual_plot(self, x_col: str, y_col: str) -> go.Figure:
        """Create residual plot"""
        # Prepare data
        x_data = self.df[x_col].dropna()
        y_data = self.df[y_col].dropna()
        
        # Common indices
        common_idx = x_data.index.intersection(y_data.index)
        x_data = x_data[common_idx]
        y_data = y_data[common_idx]
        
        # Linear regression
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        y_pred = p(x_data)
        
        # Calculate residuals
        residuals = y_data - y_pred
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Residual Plot', 'Residual Histogram'))
        
        # Residual plot
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='blue', size=5)
            ),
            row=1, col=1
        )
        
        # Zero reference line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Residual histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residual Distribution',
                nbinsx=30
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(
            title=f"Residual Analysis: {x_col} vs {y_col}",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def create_interactive_heatmap(self, columns: Optional[List[str]] = None,
                                 method: str = 'pearson') -> go.Figure:
        """Create interactive correlation heatmap"""
        if columns is None:
            columns = self.numeric_cols
        
        corr_matrix = self.df[columns].corr(method=method)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Correlation Coefficient")
        ))
        
        fig.update_layout(
            title=f"Correlation Heatmap ({method})",
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'},
            width=800,
            height=800
        )
        
        return fig
    
    def create_parallel_coordinates(self, columns: List[str],
                                  color_col: Optional[str] = None) -> go.Figure:
        """Create parallel coordinates plot"""
        # Set dimensions
        dimensions = []
        for col in columns:
            if col in self.numeric_cols:
                dimensions.append(
                    dict(
                        range=[self.df[col].min(), self.df[col].max()],
                        label=col,
                        values=self.df[col]
                    )
                )
            else:
                # Handle categorical variables
                unique_vals = self.df[col].unique()
                dimensions.append(
                    dict(
                        range=[0, len(unique_vals)-1],
                        tickvals=list(range(len(unique_vals))),
                        ticktext=unique_vals,
                        label=col,
                        values=pd.Categorical(self.df[col]).codes
                    )
                )
        
        # Color settings
        if color_col and color_col in self.df.columns:
            if color_col in self.numeric_cols:
                color_values = self.df[color_col]
            else:
                color_values = pd.Categorical(self.df[color_col]).codes
        else:
            color_values = np.arange(len(self.df))
        
        fig = go.Figure(data=
            go.Parcoords(
                dimensions=dimensions,
                line=dict(
                    color=color_values,
                    colorscale='Viridis',
                    showscale=True
                )
            )
        )
        
        fig.update_layout(
            title="Parallel Coordinates Plot",
            height=600
        )
        
        return fig

# --------------------- Rendering Functions ---------------------

def render_advanced_visualizations(df: pd.DataFrame):
    """Advanced visualization UI"""
    st.subheader("📊 Advanced Visualizations")
    
    if df is None or df.empty:
        st.warning("Data is empty.")
        return
    
    viz = AdvancedVisualizer(df)
    
    # Select visualization type
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Interactive Scatter Plot", "3D Scatter Plot", "Pair Plot", "Q-Q Plot", 
         "Residual Plot", "Correlation Heatmap", "Parallel Coordinates Plot"]
    )
    
    if viz_type == "Interactive Scatter Plot":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_col = st.selectbox("X-axis", viz.numeric_cols, key="adv_scatter_x")
        with col2:
            y_col = st.selectbox("Y-axis", [col for col in viz.numeric_cols if col != x_col], key="adv_scatter_y")
        with col3:
            color_col = st.selectbox(
                "Color By", 
                [None] + viz.categorical_cols + viz.numeric_cols,
                key="adv_scatter_color"
            )
        with col4:
            size_col = st.selectbox(
                "Size By",
                [None] + viz.numeric_cols,
                key="adv_scatter_size"
            )
        
        # Select hover data
        hover_cols = st.multiselect(
            "Columns to show in hover info",
            df.columns.tolist(),
            default=df.columns[:5].tolist()
        )
        
        fig = viz.create_interactive_scatter(x_col, y_col, color_col, size_col, hover_cols)
        st.plotly_chart(fig, use_container_width=True, key="adv_interactive_scatter")
        
    elif viz_type == "3D Scatter Plot":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_col = st.selectbox("X-axis", viz.numeric_cols, key="adv_3d_x")
        with col2:
            y_col = st.selectbox("Y-axis", [col for col in viz.numeric_cols if col != x_col], key="adv_3d_y")
        with col3:
            z_col = st.selectbox(
                "Z-axis", 
                [col for col in viz.numeric_cols if col not in [x_col, y_col]],
                key="adv_3d_z"
            )
        with col4:
            color_col = st.selectbox(
                "Color By",
                [None] + viz.categorical_cols + viz.numeric_cols,
                key="adv_3d_color"
            )
        
        fig = viz.create_3d_scatter(x_col, y_col, z_col, color_col)
        st.plotly_chart(fig, use_container_width=True, key="adv_3d_scatter")
        
    elif viz_type == "Pair Plot":
        selected_cols = st.multiselect(
            "Select columns to analyze (max 5 recommended)",
            viz.numeric_cols,
            default=viz.numeric_cols[:min(4, len(viz.numeric_cols))]
        )
        
        if len(selected_cols) >= 2:
            hue_col = st.selectbox(
                "Color grouping basis",
                [None] + viz.categorical_cols,
                key="adv_pair_hue"
            )
            
            with st.spinner("Creating pair plot..."):
                fig = viz.create_pair_plot(selected_cols, hue_col)
                st.pyplot(fig)
        else:
            st.warning("Please select at least 2 columns.")
            
    elif viz_type == "Q-Q Plot":
        col = st.selectbox("Column to analyze", viz.numeric_cols, key="adv_qq_col")
        
        fig = viz.create_qq_plot(col)
        st.plotly_chart(fig, use_container_width=True, key="adv_qq_plot")
        
        # Add normality test
        data = df[col].dropna()
        statistic, p_value = stats.shapiro(data)
        st.info(f"Shapiro-Wilk normality test: statistic={statistic:.4f}, p-value={p_value:.4f}")
        
    elif viz_type == "Residual Plot":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Independent Variable (X)", viz.numeric_cols, key="adv_residual_x")
        with col2:
            y_col = st.selectbox(
                "Dependent Variable (Y)", 
                [col for col in viz.numeric_cols if col != x_col],
                key="adv_residual_y"
            )
        
        fig = viz.create_residual_plot(x_col, y_col)
        st.plotly_chart(fig, use_container_width=True, key="adv_residual_plot")
        
    elif viz_type == "Correlation Heatmap":
        col1, col2 = st.columns(2)
        
        with col1:
            selected_cols = st.multiselect(
                "Select columns to analyze",
                viz.numeric_cols,
                default=viz.numeric_cols
            )
        with col2:
            method = st.selectbox(
                "Correlation method",
                ['pearson', 'spearman', 'kendall'],
                key="adv_heatmap_method"
            )
        
        if len(selected_cols) >= 2:
            fig = viz.create_interactive_heatmap(selected_cols, method)
            st.plotly_chart(fig, use_container_width=True, key="adv_interactive_heatmap")
        else:
            st.warning("Please select at least 2 columns.")
            
    elif viz_type == "Parallel Coordinates Plot":
        selected_cols = st.multiselect(
            "Select columns to analyze",
            df.columns.tolist(),
            default=df.columns[:min(5, len(df.columns))].tolist()
        )
        
        if len(selected_cols) >= 2:
            color_col = st.selectbox(
                "Color basis",
                [None] + df.columns.tolist(),
                key="adv_parallel_color"
            )
            
            fig = viz.create_parallel_coordinates(selected_cols, color_col)
            st.plotly_chart(fig, use_container_width=True, key="adv_parallel_coords")
        else:
            st.warning("Please select at least 2 columns.")


def render_visualization_customization():
    """Visualization customization options"""
    with st.expander("📐 Visualization Customization Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Color Settings")
            color_palette = st.selectbox(
                "Color Palette",
                ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                 'turbo', 'rainbow', 'jet', 'hot', 'cool'],
                key="viz_custom_palette"
            )
            opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.1)
            
        with col2:
            st.subheader("Layout")
            width = st.number_input("Width", 400, 1200, 800, 50)
            height = st.number_input("Height", 300, 1000, 600, 50)
            show_grid = st.checkbox("Show Grid", True)
            
        with col3:
            st.subheader("Other Options")
            font_size = st.slider("Font Size", 8, 20, 12)
            marker_size = st.slider("Marker Size", 2, 20, 6)
            line_width = st.slider("Line Width", 1, 10, 2)
        
        return {
            'color_palette': color_palette,
            'opacity': opacity,
            'width': width,
            'height': height,
            'show_grid': show_grid,
            'font_size': font_size,
            'marker_size': marker_size,
            'line_width': line_width
        }


def render_categorical_visualization(df: pd.DataFrame, key_prefix: str = ""):
    """Categorical variable visualization"""
    st.subheader("📂 Categorical Variable Visualization")
    
    if df is None or df.empty:
        st.warning("Data is empty.")
        return

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if not num_cols or not cat_cols:
        st.warning("Insufficient numeric or categorical columns.")
        return

    value_col = st.selectbox("X-axis variable (numeric aggregation basis)", num_cols, key=f"{key_prefix}cat_viz_value")
    category_col = st.selectbox("Y-axis variable (categorical)", cat_cols, key=f"{key_prefix}cat_viz_category")
    chart_type = st.radio("Visualization type", ["Bar Chart (Count)", "Bar Chart (Proportion)", "Pie Chart", "Treemap"], key=f"{key_prefix}cat_viz_type")

    ascending = False
    if "Bar Chart" in chart_type:
        order = st.radio("Sort order", ["Descending", "Ascending"], key=f"{key_prefix}cat_viz_order")
        ascending = order == "Ascending"

    if chart_type == "Bar Chart (Count)":
        plot_group_count_by_category(df, value_col, category_col, ascending, key_suffix=f"_{key_prefix}")
    elif chart_type == "Bar Chart (Proportion)":
        plot_group_proportion_by_category(df, value_col, category_col, ascending, key_suffix=f"_{key_prefix}")
    elif chart_type == "Pie Chart":
        plot_category_pie(df, category_col, key_suffix=f"_{key_prefix}")
    elif chart_type == "Treemap":
        plot_category_treemap(df, category_col, key_suffix=f"_{key_prefix}")


def render_all_visualizations(df: pd.DataFrame):
    """Render all visualizations"""
    # Create visualization tabs (simplified to 3 tabs)
    viz_tabs = st.tabs(["Basic Visualizations", "Advanced Visualizations", "Categorical Visualizations"])
    
    with viz_tabs[0]:
        st.subheader("📊 Basic Visualizations")
        
        if df is None or df.empty:
            st.warning("Data is empty.")
            return

        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])]

        # Customization options
        custom_options = render_visualization_customization()

        if num_cols:
            st.markdown("### 📌 Numeric Variable Distributions")
            for col in num_cols:
                plot_distribution_plotly(df, col)

            st.markdown("### 🔥 Correlations Between Numeric Variables")
            plot_correlation_heatmap_plotly(df)

        if date_cols and num_cols:
            st.markdown("### 📈 Time Series Plot (First Date + First Numeric)")
            plot_time_series_plotly(df, date_cols[0], num_cols[0])
    
    with viz_tabs[1]:
        render_advanced_visualizations(df)
    
    with viz_tabs[2]:
        render_categorical_visualization(df, key_prefix="all_viz_")

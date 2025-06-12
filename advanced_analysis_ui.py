"""
Advanced Analysis UI Functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from advanced_analysis import (
    TimeSeriesAnalyzer, TextAnalyzer, ModelRecommender, ABTestAnalyzer
)


def render_advanced_analysis(df: pd.DataFrame):
    """Advanced analysis UI"""
    st.header("🚀 Advanced Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Time Series Analysis", "Text Analysis", "Model Recommendation", "A/B Test"]
    )
    
    if analysis_type == "Time Series Analysis":
        render_time_series_analysis(df)
    elif analysis_type == "Text Analysis":
        render_text_analysis(df)
    elif analysis_type == "Model Recommendation":
        render_model_recommendation(df)
    elif analysis_type == "A/B Test":
        render_ab_test_analysis(df)


def render_time_series_analysis(df: pd.DataFrame):
    """Time series analysis UI"""
    st.subheader("📈 Time Series Analysis")
    
    # Select date and value columns
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not date_cols or not numeric_cols:
        st.warning("Time series analysis requires both date columns and numeric columns.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Date Column", date_cols)
    with col2:
        value_col = st.selectbox("Value Column", numeric_cols)
    
    # Create time series analyzer
    ts_analyzer = TimeSeriesAnalyzer(df, date_col, value_col)
    
    # Analysis options
    analysis_options = st.multiselect(
        "Select Analysis Items",
        ["Stationarity Test", "Time Series Decomposition", "Seasonality Detection", "ARIMA Forecast"],
        default=["Stationarity Test", "Time Series Decomposition"]
    )
    
    # Stationarity test
    if "Stationarity Test" in analysis_options:
        st.write("### Stationarity Test")
        stationarity = ts_analyzer.check_stationarity()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ADF Test**")
            st.metric("Statistic", f"{stationarity['adf']['statistic']:.4f}")
            st.metric("p-value", f"{stationarity['adf']['p_value']:.4f}")
            st.info("Stationary: " + ("Yes" if stationarity['adf']['is_stationary'] else "No"))
        
        with col2:
            st.write("**KPSS Test**")
            st.metric("Statistic", f"{stationarity['kpss']['statistic']:.4f}")
            st.metric("p-value", f"{stationarity['kpss']['p_value']:.4f}")
            st.info("Stationary: " + ("Yes" if stationarity['kpss']['is_stationary'] else "No"))
    
    # Time series decomposition
    if "Time Series Decomposition" in analysis_options:
        st.write("### Time Series Decomposition")
        
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox("Decomposition Model", ["additive", "multiplicative"])
        with col2:
            period = st.number_input("Period", min_value=2, value=12)
        
        decomposition = ts_analyzer.decompose_time_series(model=model, period=period)
        
        # Visualization
        fig = make_subplots(rows=4, cols=1, 
                           subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
        
        fig.add_trace(go.Scatter(x=ts_analyzer.ts.index, y=ts_analyzer.ts, name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition['trend'].index, y=decomposition['trend'], name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition['seasonal'].index, y=decomposition['seasonal'], name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition['residual'].index, y=decomposition['residual'], name='Residual'), row=4, col=1)
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonality detection
    if "Seasonality Detection" in analysis_options:
        st.write("### Seasonality Detection")
        seasonality = ts_analyzer.detect_seasonality()
        
        st.write(f"**Seasonality Strength**: {seasonality['seasonality_strength']:.3f}")
        st.write("**Major Periods**:")
        for i, period in enumerate(seasonality['dominant_periods'][:3]):
            st.write(f"  {i+1}. {period:.1f} periods")
    
    # ARIMA forecast
    if "ARIMA Forecast" in analysis_options:
        st.write("### ARIMA Forecast")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            p = st.number_input("p (AR)", min_value=0, max_value=5, value=1)
        with col2:
            d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
        with col3:
            q = st.number_input("q (MA)", min_value=0, max_value=5, value=1)
        with col4:
            forecast_periods = st.number_input("Forecast Periods", min_value=1, value=30)
        
        if st.button("Run Forecast"):
            forecast_result = ts_analyzer.forecast_arima(
                order=(p, d, q), 
                forecast_periods=forecast_periods
            )
            
            if 'error' not in forecast_result:
                # Forecast result visualization
                fig = go.Figure()
                
                # Original data
                fig.add_trace(go.Scatter(
                    x=ts_analyzer.ts.index,
                    y=ts_analyzer.ts,
                    name='Actual',
                    mode='lines'
                ))
                
                # Forecast values
                last_date = ts_analyzer.ts.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_periods
                )
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_result['forecast'],
                    name='Forecast',
                    mode='lines',
                    line=dict(color='red')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_result['confidence_interval'].iloc[:, 0],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_result['confidence_interval'].iloc[:, 1],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='95% Confidence Interval'
                ))
                
                fig.update_layout(title="ARIMA Forecast Results")
                st.plotly_chart(fig, use_container_width=True)
                
                # Model information
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AIC", f"{forecast_result['aic']:.2f}")
                with col2:
                    st.metric("BIC", f"{forecast_result['bic']:.2f}")
            else:
                st.error(f"Forecast failed: {forecast_result['error']}")


def render_text_analysis(df: pd.DataFrame):
    """Text analysis UI"""
    st.subheader("📝 Text Analysis")
    
    # Select text column
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not text_cols:
        st.warning("No text columns available for analysis.")
        return
    
    text_col = st.selectbox("Select Text Column", text_cols)
    
    # Prepare text data
    texts = df[text_col].dropna().tolist()
    
    if len(texts) == 0:
        st.warning("No text data found in the selected column.")
        return
    
    # Create text analyzer
    text_analyzer = TextAnalyzer(texts)
    
    # Analysis options
    analysis_tabs = st.tabs(["Keyword Extraction", "Topic Modeling", "Sentiment Analysis", "Word Cloud"])
    
    with analysis_tabs[0]:
        st.write("### Keyword Extraction")
        
        col1, col2 = st.columns(2)
        with col1:
            n_keywords = st.slider("Number of Keywords", 5, 20, 10)
        with col2:
            method = st.selectbox("Extraction Method", ["tfidf", "frequency"])
        
        keywords = text_analyzer.extract_keywords(n_keywords, method)
        
        # Keyword visualization
        keyword_df = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
        fig = px.bar(keyword_df, x='Score', y='Keyword', orientation='h',
                    title=f"Top {n_keywords} Keywords")
        st.plotly_chart(fig)
    
    with analysis_tabs[1]:
        st.write("### Topic Modeling")
        
        col1, col2 = st.columns(2)
        with col1:
            n_topics = st.slider("Number of Topics", 2, 10, 5)
        with col2:
            n_words = st.slider("Words per Topic", 5, 15, 10)
        
        if st.button("Run Topic Modeling"):
            topics_result = text_analyzer.topic_modeling(n_topics, n_words)
            
            # Display main words for each topic
            for topic in topics_result['topics']:
                st.write(f"**Topic {topic['topic_id'] + 1}**")
                words_str = ", ".join([f"{word} ({score:.2f})" 
                                     for word, score in zip(topic['words'], topic['scores'])])
                st.write(words_str)
            
            st.metric("Perplexity", f"{topics_result['perplexity']:.2f}")
    
    with analysis_tabs[2]:
        st.write("### Sentiment Analysis")
        
        if st.button("Run Sentiment Analysis"):
            sentiment_result = text_analyzer.sentiment_analysis()
            
            # Overall sentiment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_df = pd.DataFrame(list(sentiment_result['summary'].items()), 
                                          columns=['Sentiment', 'Count'])
                fig = px.pie(sentiment_df, values='Count', names='Sentiment',
                           title="Sentiment Distribution")
                st.plotly_chart(fig)
            
            with col2:
                st.metric("Average Sentiment Score", f"{sentiment_result['average_score']:.3f}")
                
                # Count by sentiment
                for sentiment, count in sentiment_result['summary'].items():
                    st.write(f"{sentiment}: {count} items")
    
    with analysis_tabs[3]:
        st.write("### Word Cloud")
        
        if st.button("Generate Word Cloud"):
            # Combine all texts
            all_text = ' '.join(text_analyzer.processed_texts)
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis'
            ).generate(all_text)
            
            # Display
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)


def render_model_recommendation(df: pd.DataFrame):
    """Model recommendation UI"""
    st.subheader("🤖 Machine Learning Model Recommendation")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Model recommendation requires at least 2 numeric columns.")
        return
    
    # Feature and target selection
    st.write("### Data Configuration")
    
    target_col = st.selectbox("Select Target Variable", df.columns)
    feature_cols = st.multiselect(
        "Select Feature Variables",
        [col for col in df.columns if col != target_col],
        default=[col for col in numeric_cols if col != target_col][:5]
    )
    
    if not feature_cols:
        st.warning("Please select feature variables.")
        return
    
    # Prepare data
    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]
    
    # Handle missing values in target
    if y.isna().any():
        st.warning("Target variable has missing values. Removing missing values.")
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
    
    # Create model recommender
    recommender = ModelRecommender(X, y)
    
    st.write(f"**Task Type**: {recommender.task_type}")
    st.write(f"**Data Size**: {len(X)} samples, {len(feature_cols)} features")
    
    # Model evaluation
    if st.button("Start Model Evaluation"):
        with st.spinner("Evaluating models..."):
            evaluation = recommender.evaluate_models()
        
        # Display results
        st.write("### Model Evaluation Results")
        
        results_df = pd.DataFrame(evaluation['results'])
        
        # Performance chart
        if recommender.task_type == 'regression':
            fig = px.bar(results_df, x='model', y='rmse',
                        title="RMSE by Model (Lower is Better)",
                        color='complexity')
        else:
            fig = px.bar(results_df, x='model', y='accuracy',
                        title="Accuracy by Model (Higher is Better)",
                        color='complexity')
        
        st.plotly_chart(fig)
        
        # Detailed results
        st.write("### Detailed Evaluation Results")
        st.dataframe(results_df)
        
        # Best model
        best = evaluation['best_model']
        st.success(f"🏆 Recommended Model: **{best['model']}**")
        
        # Recommendation reasons
        st.write("### Recommendation Rationale")
        if best['complexity'] == 'low':
            st.info("Simple and interpretable model. Suitable for understanding basic patterns.")
        elif best['complexity'] == 'medium':
            st.info("Balanced performance and complexity. Good choice for most cases.")
        else:
            st.info("High performance but complex model. Suitable when prediction accuracy is critical.")


def render_ab_test_analysis(df: pd.DataFrame):
    """A/B test analysis UI"""
    st.subheader("🔬 A/B Test Analysis")
    
    # Select group and metric columns
    col1, col2 = st.columns(2)
    
    with col1:
        group_col = st.selectbox(
            "Select Group Column",
            df.columns,
            help="Column that distinguishes A/B groups"
        )
    
    with col2:
        metric_col = st.selectbox(
            "Select Metric Column",
            df.select_dtypes(include=[np.number]).columns,
            help="Numeric metric to compare"
        )
    
    # Check groups
    groups = df[group_col].unique()
    
    if len(groups) != 2:
        st.error("A/B testing requires exactly 2 groups.")
        return
    
    # Split data
    control_data = df[df[group_col] == groups[0]][metric_col].dropna()
    treatment_data = df[df[group_col] == groups[1]][metric_col].dropna()
    
    st.write(f"**Control Group (A)**: {groups[0]} - {len(control_data)} samples")
    st.write(f"**Treatment Group (B)**: {groups[1]} - {len(treatment_data)} samples")
    
    # Create A/B test analyzer
    ab_analyzer = ABTestAnalyzer(control_data, treatment_data)
    
    # Basic statistics
    stats_data = ab_analyzer.calculate_statistics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Control Group Statistics**")
        st.metric("Mean", f"{stats_data['control']['mean']:.3f}")
        st.metric("Standard Deviation", f"{stats_data['control']['std']:.3f}")
    
    with col2:
        st.write("**Treatment Group Statistics**")
        st.metric("Mean", f"{stats_data['treatment']['mean']:.3f}")
        st.metric("Standard Deviation", f"{stats_data['treatment']['std']:.3f}")
    
    # Distribution visualization
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=control_data,
        name=f"Control ({groups[0]})",
        opacity=0.7
    ))
    fig.add_trace(go.Histogram(
        x=treatment_data,
        name=f"Treatment ({groups[1]})",
        opacity=0.7
    ))
    fig.update_layout(
        barmode='overlay',
        title="Distribution by Group",
        xaxis_title=metric_col,
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig)
    
    # Statistical testing
    st.write("### Statistical Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_type = st.selectbox(
            "Test Method",
            ['t-test', 'mann-whitney'],
            help="Normal distribution assumption: t-test, Non-parametric: mann-whitney"
        )
    
    with col2:
        alternative = st.selectbox(
            "Alternative Hypothesis",
            ['two-sided', 'greater', 'less'],
            help="two-sided: A≠B, greater: A>B, less: A<B"
        )
    
    if st.button("Run Test"):
        test_result = ab_analyzer.perform_test(test_type, alternative)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test Statistic", f"{test_result['statistic']:.4f}")
        
        with col2:
            st.metric("p-value", f"{test_result['p_value']:.4f}")
        
        with col3:
            st.metric("Effect Size (Cohen's d)", f"{test_result['effect_size']:.3f}")
        
        # Confidence interval
        ci_lower, ci_upper = test_result['confidence_interval']
        st.info(f"95% Confidence Interval for Mean Difference: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # Result interpretation
        if test_result['is_significant']:
            st.success("✅ Statistically significant difference found (p < 0.05)")
            
            if test_result['effect_size'] < 0.2:
                effect_desc = "very small"
            elif test_result['effect_size'] < 0.5:
                effect_desc = "small"
            elif test_result['effect_size'] < 0.8:
                effect_desc = "medium"
            else:
                effect_desc = "large"
            
            st.write(f"Effect size is at a {effect_desc} level.")
        else:
            st.warning("❌ No statistically significant difference found (p ≥ 0.05)")
    
    # Sample size calculation
    st.write("### Sample Size Calculation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        desired_effect = st.number_input(
            "Desired Effect Size to Detect",
            0.1, 2.0, 0.5, 0.1,
            help="0.2: small, 0.5: medium, 0.8: large"
        )
    
    with col2:
        desired_power = st.number_input(
            "Statistical Power",
            0.5, 0.99, 0.8, 0.05,
            help="Typically 0.8 is used"
        )
    
    with col3:
        alpha = st.number_input(
            "Significance Level",
            0.01, 0.1, 0.05, 0.01,
            help="Typically 0.05 is used"
        )
    
    required_n = ab_analyzer.calculate_sample_size(desired_effect, desired_power, alpha)
    st.info(f"Required Sample Size per Group: **{required_n} participants**")
    st.write(f"Total Required Sample Size: {required_n * 2} participants")

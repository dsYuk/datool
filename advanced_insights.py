"""
Advanced Insights Generation Module
- Automatic pattern detection
- Business-oriented insights
- Predictive modeling
- Anomaly pattern detection
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedInsightGenerator:
    """Advanced insight generation class"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.insights = []
    
    def detect_correlations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Detect strong correlations"""
        insights = []
        
        if len(self.numeric_cols) < 2:
            return insights
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        for i in range(len(self.numeric_cols)):
            for j in range(i+1, len(self.numeric_cols)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                    
                    # Calculate additional statistics
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        self.df[col1].dropna(), 
                        self.df[col2].dropna()
                    )
                    
                    insight = {
                        'type': 'correlation',
                        'level': 'high' if abs(corr_value) >= 0.8 else 'moderate',
                        'columns': [col1, col2],
                        'correlation': corr_value,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'description': f"**Strong {'positive' if corr_value > 0 else 'negative'} correlation found**: "
                                     f"{col1} and {col2} show a correlation coefficient of {abs(corr_value):.2f}.",
                        'business_insight': self._generate_correlation_business_insight(col1, col2, corr_value)
                    }
                    insights.append(insight)
        
        return insights
    
    def detect_outliers(self, contamination: float = 0.1) -> Dict[str, Any]:
        """Outlier detection (Isolation Forest)"""
        if len(self.numeric_cols) == 0:
            return {}
        
        # Prepare data
        X = self.df[self.numeric_cols].dropna()
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(X_scaled)
        
        # Outlier indices
        outlier_indices = X.index[outliers == -1].tolist()
        
        # Outlier scores
        outlier_scores = iso_forest.score_samples(X_scaled)
        
        # Analyze outlier contribution by variable
        outlier_details = {}
        for col in self.numeric_cols:
            col_outliers = X.loc[outlier_indices, col]
            if len(col_outliers) > 0:
                outlier_details[col] = {
                    'count': len(col_outliers),
                    'mean': col_outliers.mean(),
                    'std': col_outliers.std(),
                    'min': col_outliers.min(),
                    'max': col_outliers.max()
                }
        
        return {
            'type': 'anomaly',
            'total_outliers': len(outlier_indices),
            'outlier_ratio': len(outlier_indices) / len(X),
            'outlier_indices': outlier_indices,
            'outlier_scores': outlier_scores,
            'outlier_details': outlier_details,
            'description': f"**Anomaly patterns detected**: {len(outlier_indices)/len(X)*100:.1f}%"
                         f"({len(outlier_indices)} rows) of the data were detected as outliers.",
            'business_insight': self._generate_outlier_business_insight(outlier_details)
        }
    
    def detect_trends(self, date_col: str, value_col: str) -> Dict[str, Any]:
        """Time series trend analysis"""
        if date_col not in self.df.columns or value_col not in self.numeric_cols:
            return {}
        
        # Aggregate by date
        ts_data = self.df.groupby(date_col)[value_col].mean().sort_index()
        
        if len(ts_data) < 10:
            return {}
        
        # Linear trend
        x = np.arange(len(ts_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_data.values)
        
        # Trend direction
        if p_value < 0.05:
            if slope > 0:
                trend_direction = "upward"
                trend_strength = "strong" if abs(r_value) > 0.7 else "moderate"
            else:
                trend_direction = "downward"
                trend_strength = "strong" if abs(r_value) > 0.7 else "moderate"
        else:
            trend_direction = "none"
            trend_strength = "weak"
        
        # Volatility analysis
        volatility = ts_data.pct_change().std()
        
        # Seasonality test (if sufficient data is available)
        seasonality = None
        if len(ts_data) >= 24:  # At least 2 years of monthly data
            try:
                # ADF test
                adf_result = adfuller(ts_data.dropna())
                is_stationary = adf_result[1] < 0.05
                
                # Seasonal decomposition
                decomposition = seasonal_decompose(ts_data, model='additive', period=12)
                seasonal_strength = decomposition.seasonal.std() / ts_data.std()
                
                seasonality = {
                    'is_stationary': is_stationary,
                    'seasonal_strength': seasonal_strength,
                    'has_seasonality': seasonal_strength > 0.1
                }
            except:
                pass
        
        return {
            'type': 'trend',
            'date_column': date_col,
            'value_column': value_col,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'volatility': volatility,
            'seasonality': seasonality,
            'description': f"**Time series trend**: {value_col} shows a {trend_strength} {trend_direction} "
                         f"trend (R²={r_value**2:.3f}).",
            'business_insight': self._generate_trend_business_insight(
                value_col, trend_direction, volatility, seasonality
            )
        }
    
    def detect_patterns_in_categories(self) -> List[Dict[str, Any]]:
        """Pattern detection in categorical variables"""
        insights = []
        
        for cat_col in self.categorical_cols:
            value_counts = self.df[cat_col].value_counts()
            
            # Imbalance detection
            if len(value_counts) > 1:
                imbalance_ratio = value_counts.iloc[0] / value_counts.sum()
                
                if imbalance_ratio > 0.7:
                    insight = {
                        'type': 'imbalance',
                        'column': cat_col,
                        'dominant_category': value_counts.index[0],
                        'dominant_ratio': imbalance_ratio,
                        'description': f"**Category imbalance**: In {cat_col}, '{value_counts.index[0]}' "
                                     f"category represents {imbalance_ratio*100:.1f}% of the total.",
                        'business_insight': self._generate_imbalance_business_insight(
                            cat_col, value_counts.index[0], imbalance_ratio
                        )
                    }
                    insights.append(insight)
            
            # Rare category detection
            rare_threshold = 0.01  # Less than 1%
            rare_categories = value_counts[value_counts / value_counts.sum() < rare_threshold]
            
            if len(rare_categories) > 0:
                insight = {
                    'type': 'rare_categories',
                    'column': cat_col,
                    'rare_count': len(rare_categories),
                    'rare_categories': rare_categories.index.tolist()[:5],  # Top 5 only
                    'description': f"**Rare categories found**: {cat_col} has {len(rare_categories)} "
                                 f"rare categories (less than 1%).",
                    'business_insight': "Consider data quality review or category consolidation."
                }
                insights.append(insight)
        
        return insights
    
    def perform_predictive_analysis(self, target_col: str, 
                                  feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Predictive analysis"""
        if target_col not in self.numeric_cols:
            return {}
        
        if feature_cols is None:
            feature_cols = [col for col in self.numeric_cols if col != target_col]
        
        if len(feature_cols) == 0:
            return {}
        
        # Prepare data
        data = self.df[feature_cols + [target_col]].dropna()
        X = data[feature_cols]
        y = data[target_col]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Performance evaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Residuals
        residuals = y_test - y_pred
        
        # Calculate coefficients and p-values
        n = len(X)
        p = len(feature_cols)
        
        # MSE
        mse_val = np.sum(residuals**2) / (n - p - 1)
        
        # Standard errors
        var_b = mse_val * np.linalg.inv(X.T @ X).diagonal()
        sd_b = np.sqrt(var_b)
        
        # t-statistics
        ts_b = model.coef_ / sd_b
        
        # p-values
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), n - p - 1)) for t in ts_b]
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return {
            'type': 'predictive',
            'target': target_col,
            'features': feature_cols,
            'r2_score': r2,
            'rmse': rmse,
            'feature_importance': feature_importance,
            'top_features': feature_importance.head(3)['feature'].tolist(),
            'description': f"**Predictive model performance**: The model for predicting {target_col} has an R² of {r2:.3f}.",
            'business_insight': self._generate_predictive_business_insight(
                target_col, r2, feature_importance.head(3)
            )
        }
    
    def _generate_correlation_business_insight(self, col1: str, col2: str, 
                                             correlation: float) -> str:
        """Generate business insight for correlations"""
        if abs(correlation) > 0.8:
            return f"{col1} and {col2} show a very strong relationship. " \
                   f"Changes in one variable are likely to directly impact the other."
        else:
            return f"The relationship between {col1} and {col2} can be leveraged for " \
                   f"building predictive models or business decision making."
    
    def _generate_outlier_business_insight(self, outlier_details: Dict[str, Any]) -> str:
        """Generate business insight for outliers"""
        if outlier_details:
            top_col = max(outlier_details, key=lambda x: outlier_details[x]['count'])
            return f"Many outliers were found especially in {top_col}. " \
                   f"Data quality review, special event verification, or separate analysis may be needed."
        return "Analyze outlier patterns to identify unusual transactions or special cases."
    
    def _generate_trend_business_insight(self, col: str, direction: str, 
                                       volatility: float, seasonality: Optional[Dict]) -> str:
        """Generate business insight for trends"""
        insight = f"The {direction} trend in {col} "
        
        if direction == "upward":
            insight += "suggests growth opportunities. "
        elif direction == "downward":
            insight += "may require attention or intervention. "
        
        if volatility > 0.1:
            insight += "High volatility indicates risk management is important. "
        
        if seasonality and seasonality.get('has_seasonality'):
            insight += "Seasonality patterns suggest seasonal strategies are needed."
        
        return insight
    
    def _generate_imbalance_business_insight(self, col: str, dominant: str, 
                                           ratio: float) -> str:
        """Generate business insight for imbalance"""
        return f"The concentration of '{dominant}' in {col} is {ratio*100:.0f}%, which is very high. " \
               f"Diversification strategies or concentration risk management may be needed."
    
    def _generate_predictive_business_insight(self, target: str, r2: float, 
                                            top_features: pd.DataFrame) -> str:
        """Generate business insight for predictive models"""
        if r2 > 0.7:
            quality = "with high accuracy"
        elif r2 > 0.5:
            quality = "at an appropriate level"
        else:
            quality = "with limited accuracy"
        
        top_feature = top_features.iloc[0]['feature']
        return f"{target} can be predicted {quality}, " \
               f"with {top_feature} being the most important predictor variable."


def render_advanced_insights(df: pd.DataFrame):
    """Advanced insights UI"""
    st.header("🔍 Advanced Insight Analysis")
    
    generator = AdvancedInsightGenerator(df)
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_types = st.multiselect(
            "Select Analysis Types",
            ["Correlation Patterns", "Outlier Detection", "Time Series Trends", 
             "Category Patterns", "Predictive Analysis"],
            default=["Correlation Patterns", "Outlier Detection"]
        )
    
    with col2:
        correlation_threshold = st.slider(
            "Correlation Threshold",
            0.5, 0.9, 0.7, 0.05
        )
    
    insights = []
    
    # Correlation patterns
    if "Correlation Patterns" in analysis_types:
        corr_insights = generator.detect_correlations(correlation_threshold)
        insights.extend(corr_insights)
    
    # Outlier detection
    if "Outlier Detection" in analysis_types:
        outlier_insight = generator.detect_outliers()
        if outlier_insight:
            insights.append(outlier_insight)
    
    # Time series trends
    if "Time Series Trends" in analysis_types and generator.datetime_cols:
        for date_col in generator.datetime_cols:
            for value_col in generator.numeric_cols:
                trend_insight = generator.detect_trends(date_col, value_col)
                if trend_insight:
                    insights.append(trend_insight)
                    break  # Analyze only the first valid combination
    
    # Category patterns
    if "Category Patterns" in analysis_types:
        cat_insights = generator.detect_patterns_in_categories()
        insights.extend(cat_insights)
    
    # Predictive analysis
    if "Predictive Analysis" in analysis_types and len(generator.numeric_cols) > 1:
        target_col = st.selectbox(
            "Select Target Variable for Prediction",
            generator.numeric_cols
        )
        pred_insight = generator.perform_predictive_analysis(target_col)
        if pred_insight:
            insights.append(pred_insight)
    
    # Display insights
    if insights:
        st.subheader(f"🎯 Discovered Insights ({len(insights)} found)")
        
        for i, insight in enumerate(insights):
            with st.expander(f"{i+1}. {insight['description']}", expanded=True):
                # Business insight
                st.info(f"💡 **Business Insight**: {insight['business_insight']}")
                
                # Details
                if insight['type'] == 'correlation':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Correlation Coefficient", f"{insight['correlation']:.3f}")
                    with col2:
                        st.metric("R²", f"{insight['r_squared']:.3f}")
                    with col3:
                        st.metric("p-value", f"{insight['p_value']:.4f}")
                    
                    # Scatter plot
                    fig = px.scatter(df, x=insight['columns'][0], y=insight['columns'][1],
                                   trendline="ols", title="Correlation Visualization")
                    st.plotly_chart(fig)
                
                elif insight['type'] == 'anomaly':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Number of Outliers", insight['total_outliers'])
                    with col2:
                        st.metric("Outlier Ratio", f"{insight['outlier_ratio']*100:.1f}%")
                    
                    # Outlier details
                    if insight['outlier_details']:
                        st.write("**Outlier Distribution by Variable**")
                        outlier_df = pd.DataFrame(insight['outlier_details']).T
                        st.dataframe(outlier_df)
                
                elif insight['type'] == 'trend':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trend Direction", insight['trend_direction'])
                    with col2:
                        st.metric("R²", f"{insight['r_squared']:.3f}")
                    with col3:
                        st.metric("Volatility", f"{insight['volatility']:.3f}")
                    
                    if insight['seasonality']:
                        st.write("**Seasonality Analysis**")
                        st.write(f"- Stationarity: {'Yes' if insight['seasonality']['is_stationary'] else 'No'}")
                        st.write(f"- Seasonality Strength: {insight['seasonality']['seasonal_strength']:.3f}")
                
                elif insight['type'] == 'predictive':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{insight['r2_score']:.3f}")
                    with col2:
                        st.metric("RMSE", f"{insight['rmse']:.3f}")
                    
                    st.write("**Key Predictor Variables**")
                    st.dataframe(insight['feature_importance'].head())
    else:
        st.info("No insights were found for the selected analysis types.")

"""
Statistical Analysis Module
- Basic statistical tests (t-test, chi-square, etc.)
- Distribution goodness-of-fit tests
- Regression analysis
- Clustering analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any, List, Optional


class StatisticalAnalyzer:
    """Statistical analysis class"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def perform_normality_tests(self, column: str) -> Dict[str, Any]:
        """Normality tests"""
        data = self.df[column].dropna()
        
        results = {}
        
        # Shapiro-Wilk test
        if len(data) <= 5000:
            stat, p = stats.shapiro(data)
            results['shapiro_wilk'] = {'statistic': stat, 'p_value': p}
        
        # Kolmogorov-Smirnov test
        stat, p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        results['kolmogorov_smirnov'] = {'statistic': stat, 'p_value': p}
        
        # Anderson-Darling test
        result = stats.anderson(data, dist='norm')
        results['anderson_darling'] = {
            'statistic': result.statistic,
            'critical_values': dict(zip(['15%', '10%', '5%', '2.5%', '1%'], result.critical_values))
        }
        
        # Jarque-Bera test
        stat, p = stats.jarque_bera(data)
        results['jarque_bera'] = {'statistic': stat, 'p_value': p}
        
        return results
    
    def perform_t_test(self, group_col: str, value_col: str, 
                      test_type: str = 'independent') -> Dict[str, Any]:
        """Perform t-test"""
        groups = self.df[group_col].unique()
        
        if len(groups) != 2:
            return {'error': 'Exactly 2 groups are required.'}
        
        group1_data = self.df[self.df[group_col] == groups[0]][value_col].dropna()
        group2_data = self.df[self.df[group_col] == groups[1]][value_col].dropna()
        
        # Test for equal variances (Levene's test)
        levene_stat, levene_p = stats.levene(group1_data, group2_data)
        
        if test_type == 'independent':
            # Independent samples t-test
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, 
                                             equal_var=(levene_p > 0.05))
        else:
            # Paired samples t-test
            if len(group1_data) != len(group2_data):
                return {'error': 'For paired t-test, both groups must have the same size.'}
            t_stat, p_value = stats.ttest_rel(group1_data, group2_data)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((group1_data.std()**2 + group2_data.std()**2) / 2)
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
        
        return {
            'groups': {
                groups[0]: {'mean': group1_data.mean(), 'std': group1_data.std(), 'n': len(group1_data)},
                groups[1]: {'mean': group2_data.mean(), 'std': group2_data.std(), 'n': len(group2_data)}
            },
            'levene_test': {'statistic': levene_stat, 'p_value': levene_p},
            't_test': {'statistic': t_stat, 'p_value': p_value},
            'effect_size': cohens_d
        }
    
    def perform_anova(self, group_col: str, value_col: str) -> Dict[str, Any]:
        """One-way ANOVA"""
        groups = []
        group_names = []
        
        for group in self.df[group_col].unique():
            group_data = self.df[self.df[group_col] == group][value_col].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(group)
        
        if len(groups) < 2:
            return {'error': 'At least 2 groups are required.'}
        
        # ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Group statistics
        group_stats = {}
        for name, data in zip(group_names, groups):
            group_stats[name] = {
                'mean': data.mean(),
                'std': data.std(),
                'n': len(data)
            }
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'group_statistics': group_stats
        }
    
    def perform_chi_square_test(self, col1: str, col2: str) -> Dict[str, Any]:
        """Chi-square test"""
        # Create contingency table
        crosstab = pd.crosstab(self.df[col1], self.df[col2])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(crosstab)
        
        # Cramér's V (effect size)
        n = crosstab.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'crosstab': crosstab,
            'expected_frequencies': pd.DataFrame(expected, 
                                               index=crosstab.index, 
                                               columns=crosstab.columns)
        }
    
    def perform_correlation_test(self, col1: str, col2: str, 
                               method: str = 'pearson') -> Dict[str, Any]:
        """Correlation test"""
        data1 = self.df[col1].dropna()
        data2 = self.df[col2].dropna()
        
        # Use only common indices
        common_idx = data1.index.intersection(data2.index)
        data1 = data1[common_idx]
        data2 = data2[common_idx]
        
        if method == 'pearson':
            corr, p_value = stats.pearsonr(data1, data2)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(data1, data2)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(data1, data2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'n': len(data1),
            'method': method
        }
    
    def perform_linear_regression(self, predictors: List[str], 
                                target: str) -> Dict[str, Any]:
        """Linear regression analysis"""
        # Prepare data
        X = self.df[predictors].dropna()
        y = self.df[target].dropna()
        
        # Common indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predictions
        y_pred = model.predict(X)
        
        # R² calculation
        r2 = r2_score(y, y_pred)
        
        # Residuals
        residuals = y - y_pred
        
        # Calculate coefficients and p-values
        n = len(X)
        p = len(predictors)
        
        # MSE
        mse = np.sum(residuals**2) / (n - p - 1)
        
        # Standard errors
        var_b = mse * np.linalg.inv(X.T @ X).diagonal()
        sd_b = np.sqrt(var_b)
        
        # t-statistics
        ts_b = model.coef_ / sd_b
        
        # p-values
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), n - p - 1)) for t in ts_b]
        
        return {
            'coefficients': dict(zip(predictors, model.coef_)),
            'intercept': model.intercept_,
            'r_squared': r2,
            'p_values': dict(zip(predictors, p_values)),
            'predictions': y_pred,
            'residuals': residuals,
            'model': model
        }
    
    def perform_kmeans_clustering(self, features: List[str], 
                                n_clusters: int = 3) -> Dict[str, Any]:
        """K-means clustering"""
        # Prepare data
        X = self.df[features].dropna()
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Silhouette score
        if n_clusters > 1:
            silhouette = silhouette_score(X_scaled, clusters)
        else:
            silhouette = None
        
        # Cluster centers
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Cluster sizes
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        
        # Cluster means
        cluster_means = {}
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_means[f'Cluster_{i}'] = X[cluster_mask].mean().to_dict()
        
        return {
            'clusters': clusters,
            'centers': centers,
            'silhouette_score': silhouette,
            'cluster_sizes': cluster_sizes.to_dict(),
            'cluster_means': cluster_means,
            'inertia': kmeans.inertia_
        }


def render_statistical_analysis(df: pd.DataFrame):
    """Statistical analysis UI"""
    st.header("📈 Statistical Analysis")
    
    analyzer = StatisticalAnalyzer(df)
    
    # Select analysis type
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Normality Test", "t-test", "ANOVA", "Chi-square Test", 
         "Correlation Test", "Regression Analysis", "Clustering"]
    )
    
    if analysis_type == "Normality Test":
        col = st.selectbox("Variable to Test", analyzer.numeric_cols)
        
        if st.button("Run Normality Test"):
            results = analyzer.perform_normality_tests(col)
            
            # Display results
            st.subheader("📊 Normality Test Results")
            
            for test_name, result in results.items():
                st.write(f"**{test_name.replace('_', ' ').title()}**")
                
                if 'p_value' in result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Statistic", f"{result['statistic']:.4f}")
                    with col2:
                        st.metric("p-value", f"{result['p_value']:.4f}")
                    with col3:
                        is_normal = result['p_value'] > 0.05
                        st.metric("Result", "Normal" if is_normal else "Non-normal")
                else:
                    # Anderson-Darling test
                    st.write(f"Statistic: {result['statistic']:.4f}")
                    st.write("Critical values:")
                    for level, value in result['critical_values'].items():
                        st.write(f"  - {level}: {value:.4f}")
            
            # Distribution visualization
            fig = px.histogram(df[col], nbins=30, title=f"Distribution of {col}")
            fig.add_vline(x=df[col].mean(), line_dash="dash", 
                         line_color="red", annotation_text="Mean")
            st.plotly_chart(fig)
    
    elif analysis_type == "t-test":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            group_col = st.selectbox("Group Variable", analyzer.categorical_cols)
        with col2:
            value_col = st.selectbox("Value Variable", analyzer.numeric_cols)
        with col3:
            test_type = st.selectbox("Test Type", ["independent", "paired"])
        
        if st.button("Run t-test"):
            results = analyzer.perform_t_test(group_col, value_col, test_type)
            
            if 'error' in results:
                st.error(results['error'])
            else:
                st.subheader("📊 t-test Results")
                
                # Group statistics
                st.write("**Group Statistics**")
                group_df = pd.DataFrame(results['groups']).T
                st.dataframe(group_df)
                
                # Test results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("t-statistic", f"{results['t_test']['statistic']:.4f}")
                with col2:
                    st.metric("p-value", f"{results['t_test']['p_value']:.4f}")
                with col3:
                    st.metric("Cohen's d", f"{results['effect_size']:.4f}")
                
                # Levene's test
                st.write("**Equal Variance Test (Levene's test)**")
                st.write(f"Statistic: {results['levene_test']['statistic']:.4f}, "
                        f"p-value: {results['levene_test']['p_value']:.4f}")
                
                # Visualization
                fig = px.box(df, x=group_col, y=value_col, 
                            title=f"Distribution of {value_col} by {group_col}")
                st.plotly_chart(fig)
    
    elif analysis_type == "ANOVA":
        col1, col2 = st.columns(2)
        
        with col1:
            group_col = st.selectbox("Group Variable", analyzer.categorical_cols)
        with col2:
            value_col = st.selectbox("Value Variable", analyzer.numeric_cols)
        
        if st.button("Run ANOVA"):
            results = analyzer.perform_anova(group_col, value_col)
            
            if 'error' in results:
                st.error(results['error'])
            else:
                st.subheader("📊 ANOVA Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("F-statistic", f"{results['f_statistic']:.4f}")
                with col2:
                    st.metric("p-value", f"{results['p_value']:.4f}")
                
                # Group statistics
                st.write("**Group Statistics**")
                group_df = pd.DataFrame(results['group_statistics']).T
                st.dataframe(group_df)
                
                # Visualization
                fig = px.box(df, x=group_col, y=value_col,
                            title=f"Distribution of {value_col} by {group_col}")
                st.plotly_chart(fig)
    
    elif analysis_type == "Chi-square Test":
        col1, col2 = st.columns(2)
        
        with col1:
            var1 = st.selectbox("Variable 1", analyzer.categorical_cols)
        with col2:
            var2 = st.selectbox("Variable 2", 
                               [col for col in analyzer.categorical_cols if col != var1])
        
        if st.button("Run Chi-square Test"):
            results = analyzer.perform_chi_square_test(var1, var2)
            
            st.subheader("📊 Chi-square Test Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("χ² statistic", f"{results['chi2_statistic']:.4f}")
            with col2:
                st.metric("p-value", f"{results['p_value']:.4f}")
            with col3:
                st.metric("Degrees of Freedom", results['degrees_of_freedom'])
            with col4:
                st.metric("Cramér's V", f"{results['cramers_v']:.4f}")
            
            # Contingency table
            st.write("**Contingency Table (Observed Frequencies)**")
            st.dataframe(results['crosstab'])
            
            # Expected frequencies
            with st.expander("View Expected Frequencies"):
                st.dataframe(results['expected_frequencies'])
            
            # Heatmap
            fig = px.imshow(results['crosstab'], 
                           text_auto=True,
                           title=f"{var1} vs {var2} Contingency Table")
            st.plotly_chart(fig)
    
    elif analysis_type == "Correlation Test":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            var1 = st.selectbox("Variable 1", analyzer.numeric_cols)
        with col2:
            var2 = st.selectbox("Variable 2", 
                               [col for col in analyzer.numeric_cols if col != var1])
        with col3:
            method = st.selectbox("Method", ['pearson', 'spearman', 'kendall'])
        
        if st.button("Run Correlation Test"):
            results = analyzer.perform_correlation_test(var1, var2, method)
            
            st.subheader("📊 Correlation Test Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correlation Coefficient", f"{results['correlation']:.4f}")
            with col2:
                st.metric("p-value", f"{results['p_value']:.4f}")
            with col3:
                st.metric("Sample Size", results['n'])
            
            # Scatter plot
            fig = px.scatter(df, x=var1, y=var2, 
                            trendline="ols",
                            title=f"{var1} vs {var2}")
            st.plotly_chart(fig)
    
    elif analysis_type == "Regression Analysis":
        st.subheader("Linear Regression Analysis")
        
        # Select predictor variables
        predictors = st.multiselect("Select Predictor Variables", analyzer.numeric_cols)
        
        # Select target variable
        available_targets = [col for col in analyzer.numeric_cols if col not in predictors]
        target = st.selectbox("Target Variable", available_targets)
        
        if predictors and st.button("Run Regression Analysis"):
            results = analyzer.perform_linear_regression(predictors, target)
            
            st.subheader("📊 Regression Analysis Results")
            
            # Display R²
            st.metric("R² (R-squared)", f"{results['r_squared']:.4f}")
            
            # Display coefficients
            st.write("**Regression Coefficients**")
            coef_df = pd.DataFrame({
                'Coefficient': results['coefficients'],
                'p-value': results['p_values']
            })
            coef_df.loc['Intercept'] = [results['intercept'], np.nan]
            st.dataframe(coef_df)
            
            # Residual plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Predicted vs Actual
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=results['predictions'],
                    y=df[target].iloc[results['residuals'].index],
                    mode='markers',
                    name='Data'
                ))
                fig1.add_trace(go.Scatter(
                    x=[results['predictions'].min(), results['predictions'].max()],
                    y=[results['predictions'].min(), results['predictions'].max()],
                    mode='lines',
                    name='Perfect Prediction Line',
                    line=dict(color='red', dash='dash')
                ))
                fig1.update_layout(
                    title="Predicted vs Actual Values",
                    xaxis_title="Predicted Values",
                    yaxis_title="Actual Values"
                )
                st.plotly_chart(fig1)
            
            with col2:
                # Residual distribution
                fig2 = px.histogram(results['residuals'], 
                                   title="Residual Distribution",
                                   nbins=30)
                st.plotly_chart(fig2)
    
    elif analysis_type == "Clustering":
        st.subheader("K-means Clustering")
        
        # Select features
        features = st.multiselect("Select Features for Clustering", 
                                 analyzer.numeric_cols,
                                 default=analyzer.numeric_cols[:3])
        
        # Number of clusters
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        
        if len(features) >= 2 and st.button("Run Clustering"):
            results = analyzer.perform_kmeans_clustering(features, n_clusters)
            
            st.subheader("📊 Clustering Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Silhouette Score", 
                         f"{results['silhouette_score']:.4f}" if results['silhouette_score'] else "N/A")
            with col2:
                st.metric("Inertia", f"{results['inertia']:.2f}")
            
            # Cluster sizes
            st.write("**Cluster Sizes**")
            cluster_size_df = pd.DataFrame.from_dict(
                results['cluster_sizes'], 
                orient='index', 
                columns=['Size']
            )
            st.dataframe(cluster_size_df)
            
            # Cluster means
            st.write("**Cluster Mean Values**")
            cluster_means_df = pd.DataFrame(results['cluster_means']).T
            st.dataframe(cluster_means_df)
            
            # Visualization (2D or 3D)
            if len(features) == 2:
                # 2D scatter plot
                plot_df = df[features].copy()
                plot_df['Cluster'] = results['clusters']
                
                fig = px.scatter(plot_df, x=features[0], y=features[1],
                               color='Cluster',
                               title="Clustering Results (2D)")
                
                # Add cluster centers
                centers_df = pd.DataFrame(results['centers'], columns=features)
                fig.add_trace(go.Scatter(
                    x=centers_df[features[0]],
                    y=centers_df[features[1]],
                    mode='markers',
                    marker=dict(size=15, symbol='star', color='red'),
                    name='Cluster Centers'
                ))
                st.plotly_chart(fig)
                
            elif len(features) >= 3:
                # 3D scatter plot or PCA
                plot_df = df[features].copy()
                plot_df['Cluster'] = results['clusters']
                
                if len(features) == 3:
                    # 3D scatter plot
                    fig = px.scatter_3d(plot_df, 
                                      x=features[0], 
                                      y=features[1],
                                      z=features[2],
                                      color='Cluster',
                                      title="Clustering Results (3D)")
                else:
                    # PCA dimensionality reduction
                    st.info("3 or more features selected, reducing to 3D using PCA.")
                    
                    # Perform PCA
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df[features])
                    
                    pca = PCA(n_components=3)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
                    pca_df['Cluster'] = results['clusters']
                    
                    fig = px.scatter_3d(pca_df,
                                      x='PC1',
                                      y='PC2', 
                                      z='PC3',
                                      color='Cluster',
                                      title="Clustering Results (PCA 3D)")
                    
                    # Explained variance ratio
                    st.write(f"**PCA Explained Variance Ratio**: {pca.explained_variance_ratio_}")
                    st.write(f"**Total Explained Variance**: {sum(pca.explained_variance_ratio_):.2%}")
                
                st.plotly_chart(fig)

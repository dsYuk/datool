"""
Advanced Analysis Features Module
- Time series analysis (seasonality, trend analysis)
- Text data analysis
- Automatic machine learning model recommendation
- A/B test analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class TimeSeriesAnalyzer:
    """Time series analysis class"""
    
    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str):
        self.df = df.copy()
        self.date_col = date_col
        self.value_col = value_col
        
        # Set date index
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.set_index(date_col).sort_index()
        self.ts = self.df[value_col]
    
    def check_stationarity(self) -> Dict[str, Any]:
        """Stationarity test"""
        # ADF Test
        adf_result = adfuller(self.ts.dropna())
        
        # KPSS Test
        kpss_result = kpss(self.ts.dropna())
        
        return {
            'adf': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            },
            'kpss': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
        }
    
    def decompose_time_series(self, model: str = 'additive', period: Optional[int] = None):
        """Time series decomposition"""
        if period is None:
            # Automatic period detection
            if self.ts.index.freq == 'D':
                period = 7  # Weekly pattern
            elif self.ts.index.freq == 'M':
                period = 12  # Annual pattern
            else:
                period = 4  # Default value
        
        decomposition = seasonal_decompose(self.ts, model=model, period=period)
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'model': model,
            'period': period
        }
    
    def forecast_arima(self, order: Tuple[int, int, int] = (1, 1, 1), 
                      forecast_periods: int = 30) -> Dict[str, Any]:
        """ARIMA forecasting"""
        try:
            # Train ARIMA model
            model = ARIMA(self.ts, order=order)
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=forecast_periods)
            
            # Confidence interval
            forecast_df = fitted_model.get_forecast(steps=forecast_periods)
            confidence_int = forecast_df.conf_int()
            
            return {
                'forecast': forecast,
                'confidence_interval': confidence_int,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'model_summary': fitted_model.summary()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_seasonality(self) -> Dict[str, Any]:
        """Seasonality pattern detection"""
        # Periodicity detection using FFT
        fft = np.fft.fft(self.ts.dropna())
        frequencies = np.fft.fftfreq(len(self.ts.dropna()))
        
        # Find major frequencies
        power = np.abs(fft) ** 2
        top_frequencies_idx = np.argsort(power)[-10:]
        top_frequencies = frequencies[top_frequencies_idx]
        
        # Calculate periods
        periods = [1 / f for f in top_frequencies if f > 0]
        
        return {
            'dominant_periods': sorted(periods)[:5],
            'seasonality_strength': self._calculate_seasonality_strength()
        }
    
    def _calculate_seasonality_strength(self) -> float:
        """Calculate seasonality strength"""
        decomposition = self.decompose_time_series()
        seasonal_variance = decomposition['seasonal'].var()
        total_variance = self.ts.var()
        
        return seasonal_variance / total_variance if total_variance > 0 else 0


class TextAnalyzer:
    """Text data analysis class"""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.processed_texts = self._preprocess_texts(texts)
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Text preprocessing"""
        processed = []
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            # Remove special characters
            text = re.sub(r'[^a-zA-Z0-9가-힣\s]', '', text)
            # Clean whitespace
            text = ' '.join(text.split())
            processed.append(text)
        return processed
    
    def extract_keywords(self, n_keywords: int = 10, method: str = 'tfidf') -> List[Tuple[str, float]]:
        """Keyword extraction"""
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=n_keywords, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(self.processed_texts)
            
            # Calculate average TF-IDF score for each word
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.mean(axis=0).A1
            
            keywords = [(feature_names[i], scores[i]) 
                       for i in scores.argsort()[::-1][:n_keywords]]
        else:
            # Simple frequency-based
            vectorizer = CountVectorizer(max_features=n_keywords, stop_words='english')
            count_matrix = vectorizer.fit_transform(self.processed_texts)
            
            feature_names = vectorizer.get_feature_names_out()
            counts = count_matrix.sum(axis=0).A1
            
            keywords = [(feature_names[i], counts[i]) 
                       for i in counts.argsort()[::-1][:n_keywords]]
        
        return keywords
    
    def topic_modeling(self, n_topics: int = 5, n_words: int = 10) -> Dict[str, Any]:
        """Topic modeling (LDA)"""
        # Create document-term matrix
        vectorizer = CountVectorizer(max_features=100, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(self.processed_texts)
        
        # LDA model
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        # Main words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[::-1][:n_words]
            top_words = [feature_names[i] for i in top_words_idx]
            top_scores = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'scores': top_scores
            })
        
        # Document-topic distribution
        doc_topic_dist = lda.transform(doc_term_matrix)
        
        return {
            'topics': topics,
            'doc_topic_distribution': doc_topic_dist,
            'perplexity': lda.perplexity(doc_term_matrix)
        }
    
    def sentiment_analysis(self) -> Dict[str, Any]:
        """Simple sentiment analysis (rule-based)"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'wonderful', 
                         'fantastic', 'happy', 'love', 'best', 'amazing']
        negative_words = ['bad', 'terrible', 'poor', 'negative', 'awful', 
                         'horrible', 'hate', 'worst', 'disappointing']
        
        sentiments = []
        for text in self.processed_texts:
            words = text.split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            if pos_count > neg_count:
                sentiment = 'positive'
                score = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
            elif neg_count > pos_count:
                sentiment = 'negative'
                score = -neg_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
            else:
                sentiment = 'neutral'
                score = 0
            
            sentiments.append({
                'sentiment': sentiment,
                'score': score,
                'positive_words': pos_count,
                'negative_words': neg_count
            })
        
        # Overall statistics
        sentiment_counts = pd.DataFrame(sentiments)['sentiment'].value_counts()
        
        return {
            'sentiments': sentiments,
            'summary': sentiment_counts.to_dict(),
            'average_score': np.mean([s['score'] for s in sentiments])
        }


class ModelRecommender:
    """Machine learning model recommendation class"""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'auto'):
        self.X = X
        self.y = y
        self.task_type = self._determine_task_type(y) if task_type == 'auto' else task_type
    
    def _determine_task_type(self, y: pd.Series) -> str:
        """Automatically determine task type"""
        if y.dtype in ['object', 'category'] or y.nunique() < 10:
            return 'classification'
        else:
            return 'regression'
    
    def recommend_models(self) -> List[Dict[str, Any]]:
        """Model recommendation"""
        if self.task_type == 'regression':
            models = [
                {'name': 'Linear Regression', 'model': LinearRegression(), 'complexity': 'low'},
                {'name': 'Random Forest', 'model': RandomForestRegressor(n_estimators=100, random_state=42), 'complexity': 'medium'},
                {'name': 'SVR', 'model': SVR(kernel='rbf'), 'complexity': 'high'}
            ]
        else:
            models = [
                {'name': 'Logistic Regression', 'model': LogisticRegression(random_state=42), 'complexity': 'low'},
                {'name': 'Random Forest', 'model': RandomForestClassifier(n_estimators=100, random_state=42), 'complexity': 'medium'},
                {'name': 'SVC', 'model': SVC(kernel='rbf', random_state=42), 'complexity': 'high'}
            ]
        
        return models
    
    def evaluate_models(self, test_size: float = 0.2) -> Dict[str, Any]:
        """Model evaluation"""
        # Data split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        models = self.recommend_models()
        results = []
        
        for model_info in models:
            model = model_info['model']
            
            # Training
            model.fit(X_train, y_train)
            
            # Prediction
            y_pred = model.predict(X_test)
            
            # Evaluation
            if self.task_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Cross validation
                cv_scores = cross_val_score(model, self.X, self.y, 
                                          cv=5, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                results.append({
                    'model': model_info['name'],
                    'complexity': model_info['complexity'],
                    'rmse': rmse,
                    'cv_rmse': cv_rmse,
                    'performance': 1 / (1 + rmse)  # Convert to 0-1 value
                })
            else:
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross validation
                cv_scores = cross_val_score(model, self.X, self.y, cv=5)
                cv_accuracy = cv_scores.mean()
                
                results.append({
                    'model': model_info['name'],
                    'complexity': model_info['complexity'],
                    'accuracy': accuracy,
                    'cv_accuracy': cv_accuracy,
                    'performance': accuracy
                })
        
        # Find best performing model
        best_model = max(results, key=lambda x: x['performance'])
        
        return {
            'results': results,
            'best_model': best_model,
            'task_type': self.task_type
        }


class ABTestAnalyzer:
    """A/B test analysis class"""
    
    def __init__(self, control_data: pd.Series, treatment_data: pd.Series):
        self.control = control_data
        self.treatment = treatment_data
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistics"""
        return {
            'control': {
                'mean': self.control.mean(),
                'std': self.control.std(),
                'count': len(self.control),
                'se': self.control.std() / np.sqrt(len(self.control))
            },
            'treatment': {
                'mean': self.treatment.mean(),
                'std': self.treatment.std(),
                'count': len(self.treatment),
                'se': self.treatment.std() / np.sqrt(len(self.treatment))
            }
        }
    
    def perform_test(self, test_type: str = 't-test', 
                    alternative: str = 'two-sided') -> Dict[str, Any]:
        """Perform statistical test"""
        if test_type == 't-test':
            # t-test
            statistic, p_value = stats.ttest_ind(self.treatment, self.control, 
                                                alternative=alternative)
        elif test_type == 'mann-whitney':
            # Non-parametric test
            statistic, p_value = stats.mannwhitneyu(self.treatment, self.control, 
                                                   alternative=alternative)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Calculate effect size
        effect_size = self._calculate_effect_size()
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval()
        
        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': confidence_interval,
            'is_significant': p_value < 0.05
        }
    
    def _calculate_effect_size(self) -> float:
        """Calculate Cohen's d"""
        pooled_std = np.sqrt((self.control.std()**2 + self.treatment.std()**2) / 2)
        return (self.treatment.mean() - self.control.mean()) / pooled_std
    
    def _calculate_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean difference"""
        diff = self.treatment.mean() - self.control.mean()
        
        # Standard error
        se = np.sqrt(self.control.var()/len(self.control) + 
                    self.treatment.var()/len(self.treatment))
        
        # t-distribution critical value
        df = len(self.control) + len(self.treatment) - 2
        t_critical = stats.t.ppf((1 + confidence) / 2, df)
        
        margin = t_critical * se
        
        return (diff - margin, diff + margin)
    
    def calculate_sample_size(self, effect_size: float = 0.5, 
                            power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculate required sample size"""
        from statsmodels.stats.power import tt_ind_solve_power
        
        sample_size = tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1,
            alternative='two-sided'
        )
        
        return int(np.ceil(sample_size))


# Import UI functions from separate file
from advanced_analysis_ui import render_advanced_analysis

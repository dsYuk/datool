"""
Data Preprocessing and Validation Module
- Outlier detection and handling
- Automatic data type conversion
- Duplicate data handling
- Missing value handling
"""

import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing class"""
    
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("Cannot preprocess empty dataframe.")
        
        self.df = df.copy()
        self.original_df = df.copy()
        self.preprocessing_report = {
            'outliers': {},
            'duplicates': {},
            'missing_values': {},
            'type_conversions': {}
        }
    
    def detect_outliers(self, method: str = 'IQR', threshold: float = 1.5) -> Dict[str, List[int]]:
        """Outlier detection"""
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return outliers
        
        for col in numeric_cols:
            try:
                col_data = self.df[col].dropna()
                if len(col_data) < 10:
                    continue
                
                if method == 'IQR':
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR == 0:
                        continue
                        
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outlier_condition = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                    
                elif method == 'Z-Score':
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    
                    if std_val == 0:
                        continue
                    
                    z_scores = np.abs((self.df[col] - mean_val) / std_val)
                    outlier_condition = z_scores > threshold
                
                outlier_condition = outlier_condition & self.df[col].notna()
                outlier_indices = self.df[outlier_condition].index.tolist()
                
                if outlier_indices:
                    outliers[col] = outlier_indices
                    
            except Exception as e:
                st.warning(f"Error detecting outliers in column '{col}': {str(e)}")
                continue
        
        self.preprocessing_report['outliers'] = outliers
        return outliers
    
    def handle_outliers(self, method: str = 'remove', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle outliers"""
        if not columns:
            columns = list(self.preprocessing_report['outliers'].keys())
        
        removed_total = 0
        
        for col in columns:
            if col in self.preprocessing_report['outliers'] and col in self.df.columns:
                outlier_indices = self.preprocessing_report['outliers'][col]
                
                try:
                    if method == 'remove':
                        before_len = len(self.df)
                        self.df = self.df.drop(outlier_indices)
                        removed_total += before_len - len(self.df)
                        
                    elif method == 'cap':
                        lower = self.df[col].quantile(0.05)
                        upper = self.df[col].quantile(0.95)
                        self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                        
                    elif method == 'mean':
                        if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                            mean_val = self.df[col].mean()
                            self.df.loc[outlier_indices, col] = mean_val
                            
                    elif method == 'median':
                        if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                            median_val = self.df[col].median()
                            self.df.loc[outlier_indices, col] = median_val
                            
                except Exception as e:
                    st.warning(f"Error handling outliers in column '{col}': {str(e)}")
                    continue
        
        if method == 'remove' and removed_total > 0:
            st.info(f"Total {removed_total} outlier rows were removed.")
        
        return self.df
    
    def analyze_type_conversion_options(self) -> Dict[str, Dict[str, Any]]:
        """Analyze data type conversion possibilities"""
        conversion_options = {}
        
        for col in self.df.columns:
            original_type = str(self.df[col].dtype)
            options = {
                'current_type': original_type,
                'sample_values': self.df[col].dropna().head(5).tolist(),
                'unique_count': self.df[col].nunique(),
                'null_count': self.df[col].isnull().sum(),
                'possible_conversions': []
            }
            
            if self.df[col].dtype == 'object':
                try:
                    temp_numeric = pd.to_numeric(self.df[col], errors='coerce')
                    numeric_ratio = temp_numeric.notna().sum() / len(self.df[col])
                    if numeric_ratio > 0.5:
                        options['possible_conversions'].append({
                            'target_type': 'numeric',
                            'success_rate': f"{numeric_ratio:.1%}",
                            'description': 'Convert to numeric (int64/float64)'
                        })
                except:
                    pass
                
                try:
                    temp_datetime = pd.to_datetime(self.df[col], errors='coerce')
                    datetime_ratio = temp_datetime.notna().sum() / len(self.df[col])
                    if datetime_ratio > 0.5:
                        options['possible_conversions'].append({
                            'target_type': 'datetime',
                            'success_rate': f"{datetime_ratio:.1%}",
                            'description': 'Convert to datetime type'
                        })
                except:
                    pass
                
                unique_count = self.df[col].nunique()
                unique_ratio = unique_count / len(self.df)
                if unique_ratio < 0.3 and unique_count < 50:
                    options['possible_conversions'].append({
                        'target_type': 'category',
                        'success_rate': '100%',
                        'description': f'Convert to categorical ({unique_count} unique values)'
                    })
            
            elif self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                unique_count = self.df[col].nunique()
                if unique_count < 20:
                    options['possible_conversions'].append({
                        'target_type': 'category',
                        'success_rate': '100%',
                        'description': f'Convert to categorical ({unique_count} unique values)'
                    })
            
            conversion_options[col] = options
        
        return conversion_options
    
    def convert_column_type(self, column: str, target_type: str) -> bool:
        """Convert specific column data type"""
        try:
            original_type = str(self.df[column].dtype)
            
            if target_type == 'numeric':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            elif target_type == 'datetime':
                self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
            elif target_type == 'category':
                self.df[column] = self.df[column].astype('category')
            elif target_type == 'string':
                self.df[column] = self.df[column].astype('string')
            else:
                return False
            
            if 'type_conversions' not in self.preprocessing_report:
                self.preprocessing_report['type_conversions'] = {}
            
            self.preprocessing_report['type_conversions'][column] = f"{original_type} → {target_type}"
            return True
            
        except Exception as e:
            st.error(f"Error converting column '{column}' type: {str(e)}")
            return False
    
    def detect_duplicates(self) -> pd.DataFrame:
        """Detect duplicate data"""
        try:
            duplicates = self.df[self.df.duplicated()]
            self.preprocessing_report['duplicates'] = {
                'count': len(duplicates),
                'indices': duplicates.index.tolist()
            }
            return duplicates
        except Exception as e:
            st.error(f"Error detecting duplicates: {str(e)}")
            return pd.DataFrame()
    
    def handle_duplicates(self, keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate data"""
        try:
            before_count = len(self.df)
            self.df = self.df.drop_duplicates(keep=keep)
            after_count = len(self.df)
            
            removed_count = before_count - after_count
            self.preprocessing_report['duplicates']['removed'] = removed_count
            
            if removed_count > 0:
                st.info(f"{removed_count} duplicate rows were removed.")
            
        except Exception as e:
            st.error(f"Error removing duplicates: {str(e)}")
        
        return self.df
    
    def detect_missing_values(self) -> Dict[str, Dict[str, Any]]:
        """Missing value detection"""
        missing_info = {}
        
        for col in self.df.columns:
            try:
                missing_count = self.df[col].isna().sum()
                if missing_count > 0:
                    missing_info[col] = {
                        'count': missing_count,
                        'percentage': (missing_count / len(self.df)) * 100,
                        'pattern': self._detect_missing_pattern(col)
                    }
            except Exception as e:
                st.warning(f"Error detecting missing values in column '{col}': {str(e)}")
                continue
        
        self.preprocessing_report['missing_values'] = missing_info
        return missing_info
    
    def _detect_missing_pattern(self, column: str) -> str:
        """Missing value pattern detection"""
        try:
            is_missing = self.df[column].isna()
            missing_ratio = is_missing.sum() / len(is_missing)
            
            if missing_ratio > 0.9:
                return "Mostly missing (90%+)"
            
            correlations = []
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            for other_col in numeric_cols:
                if other_col != column and len(self.df[other_col].dropna()) > 10:
                    try:
                        other_notna = self.df[other_col].notna().astype(int)
                        missing_pattern = is_missing.astype(int)
                        
                        corr = missing_pattern.corr(other_notna)
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except:
                        continue
            
            if correlations:
                max_corr = max(correlations)
                if max_corr < 0.1:
                    return "MCAR (Missing Completely at Random)"
                elif max_corr > 0.3:
                    return "MAR (Missing at Random)"
                else:
                    return "MNAR (Missing Not at Random)"
            else:
                return "Pattern analysis unavailable"
                
        except Exception:
            return "Pattern analysis error"
    
    def handle_missing_values(self, method: str = 'drop', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Missing value handling"""
        if not columns:
            missing_info = self.detect_missing_values()
            columns = list(missing_info.keys())
        
        if not columns:
            return self.df
        
        original_shape = self.df.shape
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            try:
                missing_count_before = self.df[col].isna().sum()
                
                if missing_count_before == 0:
                    continue
                
                if method == 'drop':
                    self.df = self.df.dropna(subset=[col])
                elif method == 'forward_fill':
                    self.df[col] = self.df[col].ffill()
                elif method == 'backward_fill':
                    self.df[col] = self.df[col].bfill()
                elif method == 'mean':
                    if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        mean_val = self.df[col].mean()
                        if not np.isnan(mean_val):
                            self.df[col] = self.df[col].fillna(mean_val)
                    else:
                        st.warning(f"Column '{col}' is not numeric and cannot be filled with mean.")
                elif method == 'median':
                    if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        median_val = self.df[col].median()
                        if not np.isnan(median_val):
                            self.df[col] = self.df[col].fillna(median_val)
                    else:
                        st.warning(f"Column '{col}' is not numeric and cannot be filled with median.")
                elif method == 'mode':
                    mode_values = self.df[col].mode()
                    if len(mode_values) > 0:
                        mode_val = mode_values.iloc[0]
                        self.df[col] = self.df[col].fillna(mode_val)
                    else:
                        st.warning(f"Cannot find mode for column '{col}'.")
                elif method == 'interpolate':
                    if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        self.df[col] = self.df[col].interpolate()
                    else:
                        st.warning(f"Column '{col}' is not numeric and cannot be interpolated.")
                
                missing_count_after = self.df[col].isna().sum()
                processed_count = missing_count_before - missing_count_after
                
                if processed_count > 0:
                    st.info(f"Column '{col}': {processed_count} missing values were handled.")
                    
            except Exception as e:
                st.error(f"Error handling missing values in column '{col}': {str(e)}")
                continue
        
        if original_shape != self.df.shape:
            st.info(f"Data shape changed: {original_shape} → {self.df.shape}")
        
        return self.df
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Return preprocessing summary information"""
        current_missing = self.df.isnull().sum().sum()
        original_missing = self.original_df.isnull().sum().sum()
        
        return {
            'original_shape': self.original_df.shape,
            'processed_shape': self.df.shape,
            'original_missing': original_missing,
            'current_missing': current_missing,
            'rows_removed': self.original_df.shape[0] - self.df.shape[0],
            'preprocessing_report': self.preprocessing_report
        }


def render_preprocessing_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Data preprocessing UI"""
    st.header("🔧 Data Preprocessing")
    
    if df is None or df.empty:
        st.error("No data to process.")
        return df
    
    try:
        preprocessor = DataPreprocessor(df)
    except Exception as e:
        st.error(f"Preprocessor initialization failed: {str(e)}")
        return df
    
    # Display current data status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", f"{len(df.columns)}")
    with col3:
        missing_total = df.isnull().sum().sum()
        st.metric("Total Missing Values", f"{missing_total:,}")
    with col4:
        duplicates = len(df[df.duplicated()])
        st.metric("Duplicate Rows", f"{duplicates:,}")
    
    # Save preprocessor to session state
    if 'preprocessor' not in st.session_state:
        st.session_state['preprocessor'] = preprocessor
    
    # Step-by-step preprocessing execution
    preprocessing_tabs = st.tabs(["1️⃣ Type Conversion", "2️⃣ Remove Duplicates", "3️⃣ Handle Missing Values", "4️⃣ Handle Outliers", "5️⃣ Summary"])
    
    with preprocessing_tabs[0]:
        st.subheader("📋 Data Type Conversion")
        
        conversion_options = st.session_state['preprocessor'].analyze_type_conversion_options()
        
        st.write("**📊 Current Data Type Information**")
        type_info = []
        for col, info in conversion_options.items():
            type_info.append({
                'Column': col,
                'Current Type': info['current_type'],
                'Unique Count': info['unique_count'],
                'Missing Count': info['null_count'],
                'Sample Values': str(info['sample_values'][:3])
            })
        
        type_df = pd.DataFrame(type_info)
        st.dataframe(type_df, use_container_width=True)
        
        st.divider()
        
        st.write("**🔧 Individual Column Type Conversion**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_column = st.selectbox(
                "Select Column to Convert",
                list(conversion_options.keys()),
                help="Select the column you want to convert"
            )
        
        if selected_column:
            col_info = conversion_options[selected_column]
            
            with col2:
                available_types = ['Keep Current Type']
                type_descriptions = {'Keep Current Type': f"Current: {col_info['current_type']}"}
                
                for conv in col_info['possible_conversions']:
                    available_types.append(conv['target_type'])
                    type_descriptions[conv['target_type']] = f"{conv['description']} (Success Rate: {conv['success_rate']})"
                
                force_types = ['numeric', 'datetime', 'category', 'string']
                for ft in force_types:
                    if ft not in available_types:
                        available_types.append(f"{ft} (force)")
                        type_descriptions[f"{ft} (force)"] = f"Force convert to {ft} type (may cause errors)"
                
                selected_type = st.selectbox(
                    "Target Type",
                    available_types,
                    help="Select the data type you want to convert to"
                )
            
            if selected_type in type_descriptions:
                st.info(f"💡 {type_descriptions[selected_type]}")
            
            with st.expander(f"📋 '{selected_column}' Column Details"):
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.write(f"**Current Type**: {col_info['current_type']}")
                    st.write(f"**Unique Count**: {col_info['unique_count']:,}")
                    st.write(f"**Missing Count**: {col_info['null_count']:,}")
                
                with col_detail2:
                    st.write(f"**Sample Values**:")
                    for i, val in enumerate(col_info['sample_values'][:5]):
                        st.write(f"  {i+1}. {val}")
                
                if col_info['unique_count'] <= 20:
                    unique_vals = st.session_state['preprocessor'].df[selected_column].unique()
                    st.write(f"**All Unique Values**: {list(unique_vals)}")
            
            if selected_type != 'Keep Current Type':
                if st.button(f"🔄 Convert '{selected_column}' Type", key=f"convert_{selected_column}"):
                    target_type = selected_type.replace(' (force)', '')
                    
                    with st.spinner(f"Converting '{selected_column}' column to {target_type} type..."):
                        success = st.session_state['preprocessor'].convert_column_type(selected_column, target_type)
                        
                        if success:
                            st.success(f"✅ '{selected_column}' column was converted to {target_type} type!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to convert '{selected_column}' column.")
        
        st.divider()
        
        if 'type_conversions' in st.session_state['preprocessor'].preprocessing_report:
            conversions = st.session_state['preprocessor'].preprocessing_report['type_conversions']
            if conversions:
                st.write("**📜 Conversion History**")
                for col, conversion in conversions.items():
                    st.write(f"• {col}: {conversion}")
            else:
                st.info("No type conversions performed yet.")
        else:
            st.info("No type conversions performed yet.")
    
    with preprocessing_tabs[1]:
        st.subheader("Duplicate Data Handling")
        
        duplicates = st.session_state['preprocessor'].detect_duplicates()
        st.metric("Duplicate Rows", len(duplicates))
        
        if len(duplicates) > 0:
            st.write("**Duplicate Data Preview:**")
            st.dataframe(duplicates.head())
            
            keep_option = st.selectbox(
                "Select Row to Keep",
                ['first', 'last'],
                help="first: keep first occurrence, last: keep last occurrence"
            )
            
            if st.button("🗑️ Remove Duplicates", key="remove_duplicates"):
                with st.spinner("Removing duplicates..."):
                    st.session_state['preprocessor'].handle_duplicates(keep=keep_option)
        else:
            st.info("No duplicate rows found.")
    
    with preprocessing_tabs[2]:
        st.subheader("Missing Value Handling")
        
        missing_info = st.session_state['preprocessor'].detect_missing_values()
        
        if missing_info:
            missing_df = pd.DataFrame([
                {
                    'Column': col, 
                    'Missing Count': info['count'], 
                    'Percentage': f"{info['percentage']:.1f}%", 
                    'Pattern': info['pattern']
                }
                for col, info in missing_info.items()
            ])
            st.dataframe(missing_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_cols = st.multiselect(
                    "Select Columns to Process", 
                    list(missing_info.keys()),
                    default=list(missing_info.keys()),
                    help="Select columns to handle missing values"
                )
            
            with col2:
                missing_method = st.selectbox(
                    "Handling Method",
                    ['drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'interpolate'],
                    help="drop: remove rows, mean: fill with mean, median: fill with median, mode: fill with mode, forward_fill: fill with previous value, backward_fill: fill with next value, interpolate: interpolation"
                )
            
            if st.button("🔧 Handle Missing Values", key="handle_missing"):
                if selected_cols:
                    with st.spinner("Handling missing values..."):
                        st.session_state['preprocessor'].handle_missing_values(
                            method=missing_method, 
                            columns=selected_cols
                        )
                        st.success("✅ Missing value handling completed.")
                else:
                    st.warning("Please select columns to process.")
        else:
            st.info("✅ No missing values found.")
    
    with preprocessing_tabs[3]:
        st.subheader("Outlier Detection and Handling")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            outlier_method = st.selectbox(
                "Detection Method", 
                ['IQR', 'Z-Score'],
                help="IQR: Interquartile Range based, Z-Score: Standard deviation based"
            )
        
        with col2:
            threshold = st.number_input(
                "Threshold",
                min_value=0.5,
                max_value=5.0,
                value=1.5 if outlier_method == 'IQR' else 3.0,
                step=0.1,
                help="IQR: 1.5 recommended, Z-Score: 3.0 recommended"
            )
        
        with col3:
            if st.button("🔍 Detect Outliers", key="detect_outliers"):
                with st.spinner("Detecting outliers..."):
                    outliers = st.session_state['preprocessor'].detect_outliers(
                        method=outlier_method, 
                        threshold=threshold
                    )
                    
                    if outliers:
                        outlier_summary = pd.DataFrame([
                            {'Column': col, 'Outlier Count': len(indices)}
                            for col, indices in outliers.items()
                        ])
                        st.dataframe(outlier_summary)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            outlier_cols = st.multiselect(
                                "Select Columns to Process",
                                list(outliers.keys()),
                                default=list(outliers.keys()),
                                key="outlier_cols_select"
                            )
                        
                        with col2:
                            outlier_handle_method = st.selectbox(
                                "Handling Method",
                                ['remove', 'cap', 'mean', 'median'],
                                help="remove: delete rows, cap: limit to threshold, mean: replace with mean, median: replace with median"
                            )
                        
                        if st.button("⚡ Handle Outliers", key="handle_outliers"):
                            if outlier_cols:
                                with st.spinner("Handling outliers..."):
                                    st.session_state['preprocessor'].handle_outliers(
                                        method=outlier_handle_method, 
                                        columns=outlier_cols
                                    )
                                    st.success("✅ Outlier handling completed.")
                                    st.rerun()
                            else:
                                st.warning("Please select columns to process.")
                    else:
                        st.info("No outliers detected.")
    
    with preprocessing_tabs[4]:
        st.subheader("Preprocessing Summary")
        
        if st.button("📊 View Preprocessing Summary", key="summary"):
            summary = st.session_state['preprocessor'].get_preprocessing_summary()
            
            # Overall summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Original Data", 
                    f"{summary['original_shape'][0]} × {summary['original_shape'][1]}"
                )
            
            with col2:
                st.metric(
                    "Processed Data", 
                    f"{summary['processed_shape'][0]} × {summary['processed_shape'][1]}"
                )
            
            with col3:
                st.metric(
                    "Rows Removed", 
                    f"{summary['rows_removed']:,}"
                )
            
            # Missing value changes
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Missing Values Before", f"{summary['original_missing']:,}")
            with col2:
                st.metric("Missing Values After", f"{summary['current_missing']:,}")
            
            # Detailed report
            if summary['preprocessing_report']:
                with st.expander("Detailed Processing History"):
                    st.json(summary['preprocessing_report'])
    
    # Return processed data
    return st.session_state['preprocessor'].df

"""
Performance and Scalability Enhancement Module
- Large dataset processing optimization
- Memory usage monitoring
- Parallel processing support
- Caching mechanisms
"""

import streamlit as st
import pandas as pd
import numpy as np
import psutil
import gc
from functools import lru_cache, wraps
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from typing import Any, Callable, Optional, List, Dict
import hashlib
import pickle
import os
import tempfile


class MemoryMonitor:
    """Memory usage monitoring class"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Return current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    @staticmethod
    def display_memory_status():
        """Display memory status"""
        memory = MemoryMonitor.get_memory_usage()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Memory in Use", f"{memory['rss_mb']:.1f} MB")
        with col2:
            st.metric("Memory Usage", f"{memory['percent']:.1f}%")
        with col3:
            st.metric("Available Memory", f"{memory['available_mb']:.1f} MB")
        
        if memory['percent'] > 80:
            st.warning("⚠️ Memory usage is high. Consider reducing data size.")
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory"""
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert to categorical
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        st.info(f"💾 Memory optimization completed: {initial_memory:.1f}MB → {final_memory:.1f}MB ({reduction:.1f}% reduction)")
        
        return df


class DataChunker:
    """Large dataset chunk processing class"""
    
    @staticmethod
    def process_in_chunks(df: pd.DataFrame, chunk_size: int = 10000, 
                         func: Callable = None, *args, **kwargs) -> List[Any]:
        """Process data in chunks"""
        total_rows = len(df)
        chunks = []
        results = []
        
        # Create chunks
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunks.append(df.iloc[start:end])
        
        # Display progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            # Process chunk
            if func:
                result = func(chunk, *args, **kwargs)
                results.append(result)
            
            # Update progress
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {i+1}/{len(chunks)} chunks")
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    @staticmethod
    def read_csv_in_chunks(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
        """Read CSV file in chunks"""
        chunks = []
        
        with st.spinner("Reading large file..."):
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Optimize each chunk
                optimized_chunk = MemoryMonitor.optimize_dataframe(chunk)
                chunks.append(optimized_chunk)
        
        return pd.concat(chunks, ignore_index=True)


class ParallelProcessor:
    """Parallel processing class"""
    
    @staticmethod
    def parallel_apply(df: pd.DataFrame, func: Callable, 
                      n_workers: Optional[int] = None) -> pd.Series:
        """Apply function to DataFrame in parallel"""
        if n_workers is None:
            n_workers = multiprocessing.cpu_count() - 1
        
        # Split data by number of workers
        df_split = np.array_split(df, n_workers)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Parallel processing
            results = list(executor.map(func, df_split))
        
        # Combine results
        return pd.concat(results)
    
    @staticmethod
    def parallel_groupby(df: pd.DataFrame, group_col: str, 
                        agg_func: Dict[str, str], 
                        n_workers: Optional[int] = None) -> pd.DataFrame:
        """Perform GroupBy operations in parallel"""
        if n_workers is None:
            n_workers = multiprocessing.cpu_count() - 1
        
        def process_group(group_df):
            return group_df.groupby(group_col).agg(agg_func)
        
        # Split data
        df_split = np.array_split(df, n_workers)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Parallel processing
            results = list(executor.map(process_group, df_split))
        
        # Combine and re-aggregate results
        combined = pd.concat(results)
        return combined.groupby(level=0).agg(agg_func)


class CacheManager:
    """Cache management class"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="eda_cache_")
        else:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key"""
        key_data = f"{func_name}_{str(args)}_{str(kwargs)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Return cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def cache_dataframe(self, key: str, df: pd.DataFrame):
        """Cache DataFrame"""
        cache_path = self._get_cache_path(key)
        df.to_pickle(cache_path)
    
    def load_cached_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Load cached DataFrame"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)
        return None
    
    def cache_result(self, func: Callable) -> Callable:
        """Function result caching decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self._get_cache_key(func.__name__, *args, **kwargs)
            cache_path = self._get_cache_path(cache_key)
            
            # Check cache
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    
    def clear_cache(self):
        """Clear cache"""
        import shutil
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)


# Streamlit caching helper functions
@st.cache_data
def cached_read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Read CSV file (cached)"""
    return pd.read_csv(file_path, **kwargs)


@st.cache_data
def cached_read_excel(file_path: str, **kwargs) -> pd.DataFrame:
    """Read Excel file (cached)"""
    return pd.read_excel(file_path, **kwargs)


@st.cache_data
def cached_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation (cached)"""
    return df.corr()


@st.cache_data
def cached_groupby(df: pd.DataFrame, group_col: str, agg_dict: dict) -> pd.DataFrame:
    """GroupBy operation (cached)"""
    return df.groupby(group_col).agg(agg_dict)


def render_performance_settings():
    """Performance settings UI"""
    with st.expander("⚙️ Performance Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="Chunk size to use for large dataset processing"
            )
        
        with col2:
            n_workers = st.number_input(
                "Number of Parallel Workers",
                min_value=1,
                max_value=multiprocessing.cpu_count(),
                value=multiprocessing.cpu_count() - 1,
                help="Number of workers to use for parallel processing"
            )
        
        with col3:
            enable_caching = st.checkbox(
                "Enable Caching",
                value=True,
                help="Cache analysis results for performance improvement"
            )
        
        # Memory optimization
        if st.button("🚀 Run Memory Optimization"):
            st.session_state['optimize_memory'] = True
        
        # Current memory status
        st.subheader("💾 Memory Status")
        MemoryMonitor.display_memory_status()
        
        return {
            'chunk_size': chunk_size,
            'n_workers': n_workers,
            'enable_caching': enable_caching
        }


def optimize_large_dataset(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    """Optimize large dataset"""
    st.info("🔄 Starting large dataset optimization...")
    
    # 1. Memory optimization
    if settings.get('optimize_memory', True):
        df = MemoryMonitor.optimize_dataframe(df)
    
    # 2. Index optimization
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        st.info("✅ Index sorting completed")
    
    # 3. Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        st.info(f"✅ {initial_rows - len(df)} duplicate rows removed")
    
    # 4. Sparse data optimization
    for col in df.select_dtypes(include=['float', 'int']).columns:
        if (df[col] == 0).sum() / len(df) > 0.5:  # More than 50% zeros
            df[col] = pd.arrays.SparseArray(df[col], fill_value=0)
    
    # Garbage collection
    gc.collect()
    
    st.success("✅ Dataset optimization completed!")
    return df

import streamlit as st
import pandas as pd
import numpy as np


def render_overview(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:

    df_filtered = df.copy()


    st.title("Data Overview")
    # Apply date filter
    for col in date_cols:
        min_date = df[col].min().date()
        max_date = df[col].max().date()
        sel_range = st.date_input(
            f"Select {col} range", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )
        if isinstance(sel_range, (list, tuple)) and len(sel_range) == 2:
            start_date, end_date = sel_range
        else:
            start_date = end_date = sel_range
        df_filtered = df_filtered[
            (df_filtered[col] >= pd.to_datetime(start_date)) &
            (df_filtered[col] <= pd.to_datetime(end_date))
        ]

    # Identify numeric and categorical columns
    num_columns, cat_columns = [], []
    for col in df.columns:
        if len(df[col].unique()) <= 30 or df[col].dtype == np.object_:
            cat_columns.append(col.strip())
        else:
            num_columns.append(col.strip())

    st.subheader("Dataset Information")
    st.write(f"Dataset size: {df_filtered.shape[0]} rows, {df_filtered.shape[1]} columns")
    st.write(f"Columns: {', '.join(df_filtered.columns)}")
    st.write(f"Date columns: {', '.join(date_cols) if date_cols else 'None'}")
    st.write(f"Numeric columns: {len(num_columns)} columns")
    st.write(num_columns)
    st.write(f"Categorical columns: {len(cat_columns)} columns")
    st.write(cat_columns)

    # Display data overview
    rows_to_show = st.slider(
        "Number of rows to display", min_value=1, max_value=len(df_filtered), value=30, step=1, 
        help="Select the number of rows to display."
    )
    display_df = df_filtered.copy()
    for col in date_cols:
        display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.write(display_df.head(rows_to_show))
    
    if st.checkbox("Show statistical summary and missing values"):
        st.write(display_df.describe())
        # Display missing value counts
        missing_counts = df_filtered.isnull().sum().sort_values(ascending=False)
        missing_counts = missing_counts[missing_counts > 0]
        if not missing_counts.empty:
            st.subheader("Missing Value Counts")
            st.table(missing_counts.astype(int))
        else:
            st.write("No missing values found.")

    return df_filtered

# Auto EDA & Insight Platform 📊

> **Automated Exploratory Data Analysis (EDA) and Insight Generation Platform**  
> Comprehensive data analysis tool based on Python and Streamlit

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🎯 Project Overview

Auto EDA & Insight Platform is a **one-click data analysis solution** for data analysts and business users. You can perform professional-level exploratory data analysis (EDA) and generate automated insights without complex coding.

[Use at streamlit server](https://dsyuk-datool-main-facrjq.streamlit.app/)

### 🚀 Core Values
- **Accessibility**: Professional data analysis without coding knowledge
- **Efficiency**: Reduce manual EDA work from dozens of hours to minutes
- **Integration**: One-stop solution from data preprocessing to report generation
- **Scalability**: Support for large datasets and parallel processing

## ✨ Key Features

### 📋 1. Smart Data Preprocessing
- **Automatic data type detection and conversion**
  - Automatic conversion of numeric strings → numeric types
  - Automatic date format recognition and parsing
  - Categorical data optimization (50% memory reduction)

- **Outlier detection**
  - Support for IQR/Z-Score methodologies
  - Visual outlier distribution display
  - Various handling options (removal/capping/replacement)

- **Missing data pattern analysis**
  - Automatic classification of MCAR/MAR/MNAR patterns
  - 12 different missing data handling methods
  - Data quality comparison before and after processing

### 📊 2. Multi-level Visualization System

#### Basic Visualization
- Distribution histograms (with box plots)
- Correlation heatmaps
- Time series trend analysis

#### Advanced Visualization
- **3D interactive scatter plots**
- **Pair plots** (up to 5 variables)
- **Q-Q plots** (with normality tests)
- **Residual analysis plots**
- **Parallel coordinate plots**

#### Categorical Visualization
- Dynamic bar charts (frequency/proportion)
- Interactive pie charts
- Hierarchical treemaps

### 🔬 3. Advanced Statistical Analysis
- **Hypothesis testing suite**
  - t-test, ANOVA, chi-square test
  - Effect size calculation (Cohen's d, Cramér's V)
  - Automatic assumption verification (normality, homoscedasticity)

- **Predictive modeling**
  - Automatic task type detection (regression/classification)
  - Model performance comparison and recommendations
  - Feature importance analysis

### 🤖 4. Automated Insights
- **Automatic pattern detection**
  - Strong correlation discovery (adjustable thresholds)
  - Time series trend and seasonality analysis
  - Category imbalance detection

- **Business insight generation**
  - Business interpretation of statistical findings
  - Action item recommendations
  - Risk and opportunity identification

### 📄 5. Report Generation
- **Comprehensive HTML reports**
  - Responsive web design
  - Interactive charts included
  - Download and sharing capabilities

### ⚡ 6. Performance Optimization
- **Large dataset processing**
  - Chunk-based processing (memory efficiency)
  - Parallel processing support (multi-core utilization)
  - Smart caching system

- **Memory optimization**
  - Data type downcasting
  - Sparse data structure utilization
  - Real-time memory monitoring

## 🛠 Tech Stack

### **Backend**
- **Python 3.8+**: Core logic and data processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computation and array operations
- **Scipy**: Statistical analysis and scientific computing
- **Scikit-learn**: Machine learning and data mining

### **Frontend & Visualization**
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization
- **Seaborn/Matplotlib**: Statistical visualization
- **HTML/CSS**: Report styling

### **Data Processing**
- **Statsmodels**: Advanced statistical modeling
- **Openpyxl**: Excel file processing
- **Psutil**: System resource monitoring

## 📁 Project Structure

```
datool/
├── main.py                     # Main application
├── data_preprocessing.py       # Data preprocessing module
├── visualization.py           # Integrated visualization module
├── statistical_analysis.py   # Statistical analysis engine
├── advanced_analysis.py       # Advanced analysis (time series, text, ML)
├── advanced_insights.py       # AI insight generator
├── report.py                  # Report generation system
├── performance_optimization.py # Performance optimization tools
├── user_experience.py         # UX/UI helpers
├── overview.py                # Data overview analysis
├── data/                      # Sample datasets
└── screenshots/               # Screenshots
```

## 🎨 Main Screens

### 1. Main Dashboard
![homepage screenshot](/screenshots/mainpage.png)
- Intuitive tab-based navigation
- Real-time data quality check
- Progress indicators and memory monitoring

### 2. Visualization
- Histogram, Correlation Heatmap
- 3D scatter plot interaction
- Real-time parameter adjustment
- Multi-variable relationship analysis

### 3. Automated Insights
- AI-generated business insights
- Statistical significance verification
- Action item recommendations

### 4. Report Generation
- Data summary, visualization, statistical analysis, insights HTML file generation and download

## 🚀 Installation and Execution

### Requirements
```bash
Python 3.8+
pip install -r requirements.txt
```

### Execution
```bash
streamlit run main.py
```

## 💼 Business Impact

### 📈 Efficiency Improvement
- **95% reduction in analysis time**: Manual EDA work from dozens of hours → 5 minutes
- **No coding knowledge required**: Non-developers can perform professional analysis
- **Consistent analysis quality**: Standardized analysis process guarantee

### 🎯 Decision Support
- **Accelerated data-driven decision making**
- **Automatic discovery of hidden patterns**
- **Immediate business insight generation**

### 💰 Cost Reduction
- **Optimization of data analyst resources**
- **Automation of repetitive tasks**
- **Maximized reusability of analysis results**

## 🤝 Contributions
- 🌟 **Star**: Give a star if this helped you
- 🐛 **Bug Report**: Report any issues you find  
- 💡 **Feature Request**: Suggest new ideas
- 🔧 **Pull Request**: Contribute directly to help project

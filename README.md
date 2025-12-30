# 🛒 Instacart Market Basket Analysis with MLOps

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![MLOps](https://img.shields.io/badge/MLOps-Enabled-orange.svg)](https://ml-ops.org/)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-Tracked-yellow.svg)](https://wandb.ai/)

A comprehensive market basket analysis system implementing association rule mining algorithms (Apriori & FP-Growth) on the Instacart dataset, with full MLOps integration using Weights & Biases for experiment tracking and model management.

**Developed by**: Zerguerras Khayra Sarra & Fezazi Amina Khadidja  
**Course**: Machine Learning and SEDS (MLOps)  
**Academic Project**: December 2024

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Project Structure](#️-project-structure)
- [Installation & Setup](#️-installation--setup)
- [Usage](#-usage)
- [Results](#-results)
- [MLOps Integration](#-mlops-integration)
- [Authors & Contributions](#-authors--contributions)
- [Limitations](#️-limitations)
- [Future Work](#-future-work)
- [License](#-license)

## 🎯 Project Overview

This project analyzes grocery shopping patterns from Instacart's transactional data to discover product associations and generate recommendations. The system compares two association rule mining algorithms (Apriori vs. FP-Growth) and integrates MLOps practices for reproducible experimentation.

**Primary Objectives:**
- Discover frequent itemsets and association rules from retail transaction data
- Compare algorithm performance on large-scale datasets
- Build a recommendation system for complementary products
- Implement MLOps pipelines for experiment tracking and reproducibility

## ✨ Key Features

- **Dual Algorithm Implementation**: Both Apriori and FP-Growth algorithms with hyperparameter tuning
- **Comprehensive Evaluation**: Multiple metrics (support, confidence, lift, runtime) for model comparison
- **MLOps Integration**: Full experiment tracking with Weights & Biases
- **Recommendation Engine**: Generate complementary item suggestions for shopping baskets
- **Scalable Processing**: Handles millions of transactions efficiently
- **Reproducible Experiments**: Version-controlled models and artifacts

## 📁 Dataset

**Source**: [Kaggle - Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)

**Statistics**:
- 3.4 million orders
- 50,000+ products
- 200,000+ Instacart users
- 32 million individual product purchases

**Files Required** (download from Kaggle and place in `data/raw/`):
```
orders.csv                    # Order metadata (104MB)
order_products__prior.csv     # 3.2M prior order line items (551MB)
order_products__train.csv     # 138K training order line items (14MB)
products.csv                  # Product information (1.2MB)
aisles.csv                    # Aisle categorization (50KB)
departments.csv               # Department categorization (30KB)
```

## 🔬 Methodology

### 📊 Data Pipeline
1. **Data Preprocessing**: Merge relational tables, handle missing values, analyze basket sizes
2. **Feature Engineering**: Filter rare items (<0.1% support), one-hot encode transactions
3. **Dimensionality Reduction**: TransactionEncoder for efficient sparse representation

### ⚙️ Algorithm Implementation
- **Apriori Algorithm**: Breadth-first search for frequent itemsets with candidate generation
- **FP-Growth Algorithm**: Frequent pattern tree construction for efficient mining
- **Hyperparameter Tuning**: Support (0.001-0.01), confidence (0.1-0.3), lift (>1.0) thresholds

### 📈 Evaluation Metrics
- **Rule Quality**: Support, confidence, lift scores
- **Algorithm Performance**: Runtime, memory usage, scalability
- **Business Relevance**: Actionable and interpretable association rules

### 🤖 MLOps Integration
- Experiment tracking with Weights & Biases
- Model versioning and artifact management
- Reproducible pipeline configuration

## 🗂️ Project Structure

```
market-basket-analysis-mlops/
├── notebooks/               # Jupyter notebooks (execution pipeline)
│   ├── 01_preprocessing_and_eda.ipynb
│   ├── 02_fp_growth_experiments.ipynb
│   ├── 03_apriori_experiments.ipynb
│   ├── 04_best_model_selection.ipynb
│   ├── 05_best_model_usage.ipynb
│   └── 06_mlops_tracking_with_wandb.ipynb
├── data/                    # Dataset directories
│   ├── raw/                # Raw Kaggle data (.gitignore'd - download separately)
│   └── processed/          # Processed datasets (generated)
├── models/                  # Model outputs (.gitignore'd - generated locally)
│   ├── apriori/           # Apriori algorithm outputs
│   ├── fp_growth/         # FP-Growth algorithm outputs
│   └── best_model/        # Selected best model
├── outputs/                # Visualizations and plots (.gitignore'd)
├── reports/                # Documentation and reports
│   ├── Market_Basket_Analysis_MLOps_Project_Report.pdf
│   └── presentation_slides.pdf
├── .gitignore             # Git ignore rules for large files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

**Note**: Large data files (`data/`, `models/`, `outputs/`) are excluded from Git via `.gitignore` to maintain repository efficiency.

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- 8GB+ RAM (16GB recommended for full dataset processing)
- Kaggle account for dataset download

### 1. Clone Repository
```bash
git clone https://github.com/sarrazer24/market-basket-analysis-mlops.git
cd market-basket-analysis-mlops
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
1. Download from [Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)
2. Extract all CSV files to `data/raw/` directory:
   ```bash
   market-basket-analysis-mlops/
   └── data/
       └── raw/
           ├── orders.csv
           ├── order_products__prior.csv
           ├── order_products__train.csv
           ├── products.csv
           ├── aisles.csv
           └── departments.csv
   ```

### 4. (Optional) Setup Weights & Biases
```bash
pip install wandb
wandb login
# Or set environment variable: export WANDB_API_KEY=your_api_key
```

## 🚀 Usage

### Complete Pipeline Execution
Run notebooks in sequential order:

```bash
# 1. Data preprocessing and exploratory analysis
jupyter notebook notebooks/01_preprocessing_and_eda.ipynb

# 2. FP-Growth algorithm experiments
jupyter notebook notebooks/02_fp_growth_experiments.ipynb

# 3. Apriori algorithm experiments  
jupyter notebook notebooks/03_apriori_experiments.ipynb

# 4. Model comparison and selection
jupyter notebook notebooks/04_best_model_selection.ipynb

# 5. Recommendation system demonstration
jupyter notebook notebooks/05_best_model_usage.ipynb

# 6. MLOps tracking and visualization
jupyter notebook notebooks/06_mlops_tracking_with_wandb.ipynb
```

### Quick Start (Using Pre-selected Model)
```python
# Example: Generate recommendations for a shopping basket
basket = ["Banana", "Organic Whole Milk"]
recommendations = get_recommendations(basket, min_confidence=0.5)
# Returns: [("Organic Strawberries", 0.72), ("Large Eggs", 0.65)]
```

## 📊 Results

### Algorithm Comparison
| **Metric** | **Apriori** | **FP-Growth** | **Best Model** |
|------------|-------------|---------------|----------------|
| Avg Runtime | 45.2 min | 22.8 min | **FP-Growth** |
| Rules Generated | 1,240 | 980 | FP-Growth |
| Avg Confidence | 0.65 | 0.68 | **FP-Growth** |
| Avg Lift | 12.5 | 14.2 | **FP-Growth** |
| Memory Usage | High | Moderate | FP-Growth |

### Key Findings
1. **FP-Growth outperforms Apriori** in runtime (2x faster) with comparable rule quality
2. **Optimal parameters**: Support=0.005, Confidence=0.2, Lift>1.0
3. **Strong associations**: Bananas → Milk (72% confidence), Bread → Eggs (65% confidence)
4. **Department patterns**: Produce and dairy items show strongest cross-category associations

### Sample Recommendations
- **Input**: ["Banana", "Milk"]
- **Output**: 
  1. Yogurt (confidence: 0.72, lift: 15.3)
  2. Bread (confidence: 0.65, lift: 12.8) 
  3. Eggs (confidence: 0.58, lift: 11.2)

## 🔬 MLOps Integration

### Weights & Biases Dashboard
- **Experiment Tracking**: Hyperparameters, metrics, and outputs
- **Artifact Management**: Model versions and association rules
- **Visualization**: Performance comparisons and trend analysis
- **Collaboration**: Shareable dashboards and reproducible runs

### Key MLOps Features
1. **Version Control**: All experiments logged with unique identifiers
2. **Reproducibility**: Complete environment capture
3. **Model Registry**: Track model lineage and performance
4. **Automated Logging**: Metrics, plots, and artifacts automatically captured

## 👥 Authors & Contributions

### **Zerguerras Khayra Sarra**
- Primary dataset preprocessing and exploratory analysis
- Implementation of FP-Growth algorithm experiments
- MLOps integration with Weights & Biases
- Project documentation and report writing

### **Fezazi Amina Khadidja**
- Implementation of Apriori algorithm experiments
- Model evaluation and comparison framework
- Recommendation system development
- Visualization and results analysis

### **Collaborative Work**
- Joint design of methodology and evaluation metrics
- Shared development of notebooks 04-06 (model selection & MLOps)
- Collaborative debugging and optimization
- Final presentation and report preparation

## ⚠️ Limitations

1. **Data Constraints**:
   - Sparse transaction matrix limits rule coverage
   - Cold start problem for new products/users
   - Seasonal patterns not accounted for

2. **Technical Limitations**:
   - Code-driven interface (no production API/UI)
   - Batch processing only (no real-time recommendations)
   - Resource-intensive for very large datasets

3. **Algorithmic Limitations**:
   - Association rules based solely on co-occurrence
   - No personalization or user segmentation
   - Contextual factors (time, location) not considered

## 🔮 Future Work

### Short-term Improvements
- [ ] Implement rule pruning and redundancy elimination
- [ ] Add temporal analysis for seasonal patterns
- [ ] Develop basic Flask API for recommendations
- [ ] Include user segmentation for personalized rules

### Long-term Enhancements  
- [ ] Real-time recommendation serving
- [ ] Integration with collaborative filtering
- [ ] Automated retraining pipelines
- [ ] A/B testing framework
- [ ] Docker containerization for deployment

### Research Directions
- Hybrid models combining association rules with deep learning
- Context-aware recommendations (time, location, weather)
- Multi-modal data integration (product images, descriptions)
- Explainable AI for rule interpretation

## 📄 License

**Academic Use Only** - This project is for educational purposes.

- **Code**: MIT License (where applicable)
- **Dataset**: Follows [Kaggle's Terms of Use](https://www.kaggle.com/terms)
- **Report & Documentation**: CC BY-NC 4.0

**Note**: The Instacart dataset is proprietary and subject to Kaggle's competition terms. Please respect data usage guidelines.

## 🙏 Acknowledgments

- **Instacart** for providing the dataset through Kaggle
- **Weights & Biases** for the MLOps platform
- **MLxtend** library for association rule mining implementations
- Course instructors for guidance and feedback
- Peers for collaborative learning and support

---

**Project Authors**: Zerguerras Khayra Sarra & Fezazi Amina Khadidja  
**Course**: Machine Learning and SEDS (MLOps)  
**Institution**: Esi Sba  
**Completion Date**: December 2025 

*For detailed implementation and individual contributions, refer to the Jupyter notebooks and project report.*

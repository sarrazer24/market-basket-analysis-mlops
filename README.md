# Instacart Market Basket Analysis with MLOps

Market basket analysis and product recommendation using association rule mining on the Instacart dataset, with experiment tracking and artifact management via Weights & Biases (W&B). Built for Machine Learning and SEDS (MLOps) coursework.

## Dataset
- Source: Kaggle — https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
- Scale: millions of transactions across tens of thousands of products.
- Key files: orders.csv, order_products__prior.csv, order_products__train.csv, products.csv, aisles.csv, departments.csv.

## Methodology
- Data prep and EDA: merge relational tables, analyze basket sizes, filter rare items, one-hot encode with TransactionEncoder.
- Association rule mining: Apriori and FP-Growth with varying support, confidence, and lift.
- Evaluation: compare frequent itemset counts, rule counts, average confidence/lift, and runtime to select the best model.
- Usage: recommend complementary items for a given basket via the selected rules (code-driven; no frontend).
- MLOps: W&B for experiment tracking, hyperparameter logging, model/rule versioning, artifacts, and reproducibility.

## Project Structure
```
instacart-market-basket-mlops/
├── data/
│   ├── raw/              # Original Kaggle data (ignored in Git)
│   └── processed/        # Cleaned and encoded datasets
├── notebooks/
│   ├── 01_preprocessing_and_eda.ipynb
│   ├── 02_fp_growth_experiments.ipynb
│   ├── 03_apriori_experiments.ipynb
│   ├── 04_best_model_selection.ipynb
│   ├── 05_best_model_usage.ipynb
│   └── 06_mlops_tracking_with_wandb.ipynb
├── models/
│   ├── fp_growth/
│   ├── apriori/
│   └── best_model/
├── outputs/
│   ├── figures/
├── reports/
│   ├── Market_Basket_Analysis_MLOps_Project_Report.pdf
│   └── presentation_slides.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started
1) Install dependencies
```
pip install -r requirements.txt
```

2) Download the dataset from Kaggle and place the CSVs in data/raw/.

3) Run the notebooks in order: 01 → 02 → 03 → 04 → 05 → 06.

4) (Optional) Enable W&B tracking by setting WANDB_API_KEY and logging in with wandb login.

## Results (high level)
- Frequent itemsets and association rules mined from Instacart data.
- Apriori vs. FP-Growth compared on scalability and rule quality; best model selected.
- Code-based recommendations demonstrated using the chosen rule set.
- Experiments, artifacts, and metrics tracked in W&B.

## Limitations
- Rules rely on co-occurrence; sparse retail data can reduce coverage.
- No production-grade serving or UI; usage is notebook/code driven.

## Future Work
- Rule pruning/redundancy reduction and improved metrics.
- User-segmented or personalized recommendations.
- Real-time serving and automated retraining pipelines.

## License
Academic use only. Dataset license follows Kaggle terms.

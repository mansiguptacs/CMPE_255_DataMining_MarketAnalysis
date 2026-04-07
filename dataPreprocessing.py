import pandas as pd
from dataLoading import load_all_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def handle_missing_values(orders, products, order_products, order_train, departments, aisles):
    print("\n--- Phase 1: Missing Value Resolution ---")
    
    # Checking initial missing values in orders table
    print("Initial missing values in orders table:\n", orders.isnull().sum())
    
    # In instacart dataset, days_since_prior_order is NaN for the first order. We will impute these with 0 (since 0 days have passed, it's their first order).
    if 'days_since_prior_order' in orders.columns:
        orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)
        print("Imputed missing values in 'days_since_prior_order' with 0.")
    
    print("Remaining missing values in orders table:\n", orders.isnull().sum())

    # Checking initial missing values in products table
    print("Initial missing values in products table:\n", products.isnull().sum())
    
    # Checking initial missing values in departments table
    print("Initial missing values in departments table:\n", departments.isnull().sum())
    
    # Checking initial missing values in aisles table
    print("Initial missing values in aisles table:\n", aisles.isnull().sum())
    
    # Checking initial missing values in order_products table
    print("Initial missing values in order_products table:\n", order_products.isnull().sum())
    
    # Checking initial missing values in order_train table
    print("Initial missing values in order_train table:\n", order_train.isnull().sum())

    return orders, products, order_products, order_train, departments, aisles

def outlier_analysis(orders):
    print("\n--- Phase 1: Outlier Analysis (DBSCAN) ---")
    # using DBScan for outlier analysis
    
    numerical_cols = ['order_dow', 'order_hour_of_day', 'days_since_prior_order']
    df_numeric = orders[numerical_cols].fillna(0)
    
    # Standardize explicitly for DBSCAN
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # IMPORTANT: DBSCAN memory complexity is O(N^2). It will crash on 3.4M records.
    # We randomly sample 50,000 rows to efficiently identify general outliers structure.
    print("Performing outlier analysis on random sample of 50,000 records...")
    sample_size = min(50000, len(scaled_data))
    print(f"Sampling {sample_size} records to perform DBSCAN outlier detection without crashing memory...")
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(scaled_data), sample_size, replace=False)
    sample_data = scaled_data[sample_indices]
    
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    labels = dbscan.fit_predict(sample_data)
    
    # Outliers fall into the -1 cluster
    outlier_count = sum(labels == -1)
    print(f"DBSCAN flagged {outlier_count} outliers out of {sample_size} sampled records ({(outlier_count/sample_size)*100:.2f}%).")
    
    # Generate visualization for EDA
    os.makedirs('visualizations', exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    sample_orig = df_numeric.iloc[sample_indices]
    sns.scatterplot(
        x=sample_orig['order_hour_of_day'], 
        y=sample_orig['days_since_prior_order'],
        hue=['Outlier' if label == -1 else 'Inlier' for label in labels],
        palette={'Outlier': 'red', 'Inlier': 'blue'},
        alpha=0.6,
        s=15
    )
    plt.title('DBSCAN Outlier Detection (Sample)')
    plt.xlabel('Order Hour of Day')
    plt.ylabel('Days Since Prior Order')
    plt.savefig('visualizations/dbscan_outliers.png')
    plt.close()
    print("Saved outlier visualization to 'visualizations/dbscan_outliers.png'.")
    
    if outlier_count > 0:
        print("Note: To safely remove outliers across the *entire* 3.4 million row dataset without OOM issues, a method like Isolation Forest is highly recommended instead of DBSCAN.")
    
    # We return the intact dataset for now since dropping sample metrics from the full set is incomplete
    return orders

def standardize_data(orders, order_products):
    print("\n--- Phase 2: Data Standardization ---")
    scaler = StandardScaler()
    
    # Standardize numerical features in 'orders' table
    numerical_cols_orders = ['order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']
    
    print(f"Standardizing column subset in orders: {numerical_cols_orders}")
    
    print(orders[numerical_cols_orders].describe())

    # Fit and transform the data, then assign it back to the respective columns
    orders[numerical_cols_orders] = scaler.fit_transform(orders[numerical_cols_orders])
    
    return orders, order_products

def preprocess(orders, products, order_products, order_train, departments, aisles):
    """
    function to execute preprocessing pipeline.
    """
    orders, products, order_products, order_train, departments, aisles = handle_missing_values(orders, products, order_products, order_train, departments, aisles)
    
    # outlier analysis by dbscan
    orders = outlier_analysis(orders)
    
    orders, order_products = standardize_data(orders, order_products)
    
    print("Preprocessing complete.")
    return orders, products, order_products, order_train, departments, aisles

if __name__ == "__main__":
    from dataLoading import load_all_data
    # Load data when run as a standalone script
    orders, products, order_products, order_train, departments, aisles = load_all_data()
    
    # Run the preprocessing pipeline
    preprocess(orders, products, order_products, order_train, departments, aisles)

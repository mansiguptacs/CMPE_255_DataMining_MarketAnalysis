import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(orders, products, order_products, departments, aisles):
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
    

    return orders, products, order_products, departments, aisles

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

def preprocess(orders, products, order_products, departments, aisles):
    """
    function to execute preprocessing pipeline.
    """
    orders, products, order_products, departments, aisles = handle_missing_values(orders, products, order_products, departments, aisles)
    orders, order_products = standardize_data(orders, order_products)
    
    print("Preprocessing complete.")
    return orders, products, order_products, departments, aisles

if __name__ == "__main__":
    from dataLoading import load_all_data
    
    # Load data when run as a standalone script
    orders, products, order_products, departments, aisles = load_all_data()
    
    # Run the preprocessing pipeline
    preprocess(orders, products, order_products, departments, aisles)

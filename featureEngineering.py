import pandas as pd
import numpy as np

def engineer_features(orders, order_products, min_users=50000):
    print("\n--- Phase 2.5: Feature Engineering (User level) ---")
    
    # We create user-level features.
    
    # 1. Feature: Basket size & Reorder rates from order_products
    # Merge order_products with orders to get user_id for each product sold
    print("Merging order_products with orders to calculate basket-level features...")
    orders_slim = orders[['order_id', 'user_id']]
    op_merged = order_products.merge(orders_slim, on='order_id', how='inner')
    
    # Calculate basket size per order
    basket_sizes = op_merged.groupby('order_id').size().reset_index(name='basket_size')
    
    # Merge basket_size back to orders to average by user
    orders_with_basket = orders.merge(basket_sizes, on='order_id', how='left')
    orders_with_basket['basket_size'] = orders_with_basket['basket_size'].fillna(0)
    
    # Reorder rate per user
    user_reorder_rate = op_merged.groupby('user_id')['reordered'].mean().reset_index(name='user_reorder_rate')
    
    print("Grouping orders by user_id to compute behavioral metrics...")
    # Calculate user-level metrics from orders
    # Note: 'orders' numerical features were standardized in Data Preprocessing. 
    # Aggregating standardized features is perfectly viable for distance-based clustering.
    user_features = orders_with_basket.groupby('user_id').agg(
        purchase_frequency=('order_number', 'max'),
        avg_basket_size=('basket_size', 'mean'),
        avg_days_between_orders=('days_since_prior_order', 'mean'),
        customer_tenure=('days_since_prior_order', 'sum'),
        preferred_order_dow=('order_dow', 'mean'),
        preferred_order_hour=('order_hour_of_day', 'mean')
    ).reset_index()
    
    # Combine with reorder rate
    user_features = user_features.merge(user_reorder_rate, on='user_id', how='left')
    
    # Fill any NaNs that might have occurred from users with no products in the sample
    user_features = user_features.fillna(0)
    
    print(f"Generated engineered features for {len(user_features)} users.")
    
    # Sample down to `min_users` to prevent memory blowups in algorithms like Hierarchical/DBSCAN
    #if len(user_features) > min_users:
    #    print(f"Sampling dataset to {min_users} users for efficient clustering computations...")
    #    user_features = user_features.sample(n=min_users, random_state=42)
    
    # Finally, we standardize the *new* engineered features so they hold equal weight during clustering.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Drop user_id for purely statistical operations
    feature_cols = [col for col in user_features.columns if col != 'user_id']
    user_features[feature_cols] = scaler.fit_transform(user_features[feature_cols])
    
    print("Feature Engineering complete.")
    return user_features

if __name__ == "__main__":
    from dataLoading import load_all_data
    from dataPreprocessing import preprocess
    
    orders, products, order_products, departments, aisles = load_all_data()
    orders, products, order_products, departments, aisles = preprocess(orders, products, order_products, departments, aisles)
    
    user_features_df = engineer_features(orders, order_products)
    print(user_features_df.head())

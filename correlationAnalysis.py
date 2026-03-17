import os
import matplotlib.pyplot as plt
import seaborn as sns

VISUALIZATIONS_PATH = "./visualizations"

def perform_correlation_analysis(orders, order_products):
    print("\n--- Phase 3: Correlation Analysis ---")
    os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
    
    # Merge orders with order_products to get a combined view of order features and product addition order
    merged_data = order_products.merge(orders, on='order_id', how='inner')
    
    # Select numeric columns for correlation. 
    # 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order' were standardized.
    # 'add_to_cart_order' is numeric from order_products.
    numeric_cols = ['add_to_cart_order', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']
    
    correlation_matrix = merged_data[numeric_cols].corr()
    
    print("Correlation Matrix:\n", correlation_matrix)
    
    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of Order Factors')
    plt.tight_layout()
    
    heatmap_path = os.path.join(VISUALIZATIONS_PATH, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    
    print(f"Saved correlation heatmap to '{heatmap_path}'")
    
if __name__ == "__main__":
    from dataLoading import load_all_data
    from dataPreprocessing import preprocess
    
    # Load and preprocess data when run as a standalone script
    orders, products, order_products, departments, aisles = load_all_data()
    orders, products, order_products, departments, aisles = preprocess(orders, products, order_products, departments, aisles)
    
    # Run the correlation analysis
    perform_correlation_analysis(orders, order_products)

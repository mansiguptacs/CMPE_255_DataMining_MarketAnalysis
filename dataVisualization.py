import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

VISUALIZATIONS_PATH = "./visualizations"

def generate_eda_visualizations(orders, products, order_products):
    """
    Performs Exploratory Data Analysis perfectly matched to the project requirements:
    - Customer purchase frequency patterns
    - Reorder behavior
    - Basket size analysis
    - Temporal purchasing trends
    """
    print("\n--- EDA: Generating Market & Purchasing Visualizations ---")
    os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
    
    # 1. Temporal purchasing trends (Orders by Day & Hour)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='order_dow', data=orders, color='skyblue')
    plt.title('Orders by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='order_hour_of_day', data=orders, color='salmon')
    plt.title('Orders by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'eda_temporal_trends.png'))
    plt.close()
    print("Saved 'eda_temporal_trends.png'")
    
    # 2. Customer Purchase Frequency Patterns
    user_frequency = orders.groupby('user_id')['order_number'].max()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_frequency, bins=50, kde=True, color='purple')
    plt.title('Customer Purchase Frequency (Total Orders per User)')
    plt.xlabel('Total Orders Placed')
    plt.ylabel('Number of Customers')
    plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'eda_purchase_frequency.png'))
    plt.close()
    print("Saved 'eda_purchase_frequency.png'")
    
    # 3. Basket Size Analysis
    basket_sizes = order_products.groupby('order_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(basket_sizes, bins=50, kde=True, color='green')
    plt.xlim(0, 50) # Cap at 50 for readability
    plt.title('Basket Size Analysis (Items per Order)')
    plt.xlabel('Basket Size (Number of Items)')
    plt.ylabel('Frequency of Orders')
    plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'eda_basket_size.png'))
    plt.close()
    print("Saved 'eda_basket_size.png'")
    
    # 4. Reorder Behavior Analysis
    reorder_ratio = order_products['reordered'].value_counts(normalize=True)
    plt.figure(figsize=(6, 6))
    plt.pie(reorder_ratio, labels=['First Time Order', 'Reordered'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
    plt.title('Global Reorder Behavior')
    plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'eda_reorder_behavior.png'))
    plt.close()
    print("Saved 'eda_reorder_behavior.png'")

def generate_visualizations(orders, products, order_products):
    # Call the new EDA function natively
    generate_eda_visualizations(orders, products, order_products)
    
if __name__ == "__main__":
    from dataLoading import load_all_data
    
    # Load data when run as a standalone script
    orders, products, order_products, order_train, departments, aisles = load_all_data()
    
    # Run the visualization EDA directly on unscaled data
    generate_visualizations(orders, products, order_products)

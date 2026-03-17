import os
import matplotlib.pyplot as plt
import seaborn as sns

VISUALIZATIONS_PATH = "./visualizations"

def generate_visualizations(orders, products, order_products):
    print("\n--- Phase 4: General Market Visualizations ---")
    os.makedirs(VISUALIZATIONS_PATH, exist_ok=True)
    
    # Visualization 1: Orders by Day of the Week
    plt.figure(figsize=(10, 6))
    sns.countplot(x='order_dow', data=orders, color='skyblue')
    plt.title('Total Orders by Day of the Week (Standardized space)')
    plt.xlabel('Standardized Day of Week')
    plt.ylabel('Count of Orders')
    plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'orders_by_day_of_week.png'))
    plt.close()
    print("Saved 'orders_by_day_of_week.png'")

    # Visualization 2: Orders by Hour of the Day
    plt.figure(figsize=(10, 6))
    sns.countplot(x='order_hour_of_day', data=orders, color='salmon')
    plt.title('Total Orders by Hour of the Day (Standardized space)')
    plt.xlabel('Standardized Hour of the Day')
    plt.ylabel('Count of Orders')
    plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'orders_by_hour_of_day.png'))
    plt.close()
    print("Saved 'orders_by_hour_of_day.png'")

    # Visualization 3: Top 10 Best-Selling Products
    # Merge order_products with products to get product names
    merged_prod = order_products.merge(products[['product_id', 'product_name']], on='product_id', how='left')
    top_products = merged_prod['product_name'].value_counts().head(10)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(y=top_products.index, x=top_products.values, color='teal') # Using color instead of palette to avoid warning
    plt.title('Top 10 Best-Selling Products')
    plt.xlabel('Total Quantity Sold')
    plt.ylabel('Product Name')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_PATH, 'top_10_products.png'))
    plt.close()
    print("Saved 'top_10_products.png'")

if __name__ == "__main__":
    #print("This script is meant to be run via main.py")
    from dataLoading import load_all_data
    
    # Load data when run as a standalone script
    orders, products, order_products, departments, aisles = load_all_data()
    
    # Run the preprocessing pipeline
    generate_visualizations(orders, products, order_products, departments, aisles)


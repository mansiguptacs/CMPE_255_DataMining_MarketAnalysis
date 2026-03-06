import os
import shutil
import pandas as pd
import kagglehub

DATA_PATH = "./data"

REQUIRED_FILES = [
    "aisles.csv",
    "departments.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "orders.csv",
    "products.csv"
]

def ensure_data():
    os.makedirs(DATA_PATH, exist_ok=True)
    
    missing_files = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(DATA_PATH, f))]
    
    if missing_files:
        print(f"Missing files: {missing_files}. Downloading dataset...")
        path = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")
        print(f"Downloaded dataset to: {path}")
        
        for file in missing_files:
            src = os.path.join(path, file)
            dst = os.path.join(DATA_PATH, file)
            if os.path.exists(src):
                print(f"Copying {file} to {DATA_PATH}...")
                shutil.copy2(src, dst)
            else:
                print(f"Warning: {file} not found in {src}.")
    else:
        print("All required data files are present.")

def load_all_data():
    ensure_data()
    print("Loading raw data from CSV files...")
    orders = pd.read_csv(os.path.join(DATA_PATH, "orders.csv"))
    products = pd.read_csv(os.path.join(DATA_PATH, "products.csv"))
    order_products = pd.read_csv(os.path.join(DATA_PATH, "order_products__prior.csv"))
    departments = pd.read_csv(os.path.join(DATA_PATH, "departments.csv"))
    
    print(f"Loaded: {len(orders)} orders, {len(products)} products, {len(order_products)} order items.")
    return orders, products, order_products, departments

if __name__ == "__main__":
    orders, products, order_products, departments = load_all_data()
    print("Data Loading script executed successfully.")

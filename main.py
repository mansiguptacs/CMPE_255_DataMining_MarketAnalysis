from dataLoading import load_all_data
from dataPreprocessing import preprocess
from featureEngineering import engineer_features
from pca import pca
from clustering import run_clustering
from correlationAnalysis import perform_correlation_analysis
from dataVisualization import generate_visualizations

def run_pipeline():
    print("Starting Market Analysis & Customer Segmentation Pipeline...")
    
    # Step 1: Data Loading
    orders, products, order_products, order_train, departments, aisles = load_all_data()
    
    # Step 2: Data Preprocessing (Standardization & Missing Values)
    orders, products, order_products, order_train, departments, aisles = preprocess(orders, products, order_products, order_train, departments, aisles)
    
    # Step 3: Feature Engineering (User-level behavioral metrics)
    user_features_df = engineer_features(orders, order_products)
    
    # Step 4: PCA Evaluation (EDA to check dimensionality reduction potential)
    # We pass the full features through untouched
    user_features_df = pca(user_features_df)
    
    # Step 5: Clustering Models & Validation
    clustered_users = run_clustering(user_features_df)
    
    # Step 6: Market Level Analysis & Visualization
    perform_correlation_analysis(orders, order_products)
    generate_visualizations(orders, products, order_products)
    
    print("\nPipeline execution complete. All visualizations are saved in the './visualizations' directory.")

if __name__ == "__main__":
    run_pipeline()

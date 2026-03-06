from dataLoading import load_all_data
from dataPreprocessing import preprocess
from correlationAnalysis import perform_correlation_analysis
from dataVisualization import generate_visualizations

def run_pipeline():
    print("Starting Market Analysis Pipeline...")
    
    # Step 1: Data Loading
    orders, products, order_products, departments = load_all_data()
    
    # Step 2: Data Preprocessing
    orders, products, order_products, departments = preprocess(orders, products, order_products, departments)
    
    # Step 3: Correlation Analysis
    perform_correlation_analysis(orders, order_products)
    
    # Step 4: General Visualizations
    generate_visualizations(orders, products, order_products)
    
    print("\nPipeline execution complete. All visualizations are saved in the './visualizations' directory.")

if __name__ == "__main__":
    run_pipeline()

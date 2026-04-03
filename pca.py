import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataLoading import load_all_data
from dataPreprocessing import preprocess
from featureEngineering import engineer_features

def check_pca_necessity(df, feature_cols):
    """
    Evaluates whether PCA is necessary by computing the principal components
    and analyzing the cumulative explained variance.
    Returns boolean indicating if PCA is recommended, and the number of components needed.
    """
    print(f"\nEvaluating PCA on {len(feature_cols)} features...")
    
    # Drop NaNs just in case
    data_for_pca = df[feature_cols].copy().dropna()
    
    pca = PCA()
    pca.fit(data_for_pca)
    
    explained_var = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)
    
    # Print variance details
    print("\nExplained Variance Ratio per Principal Component:")
    for i, var in enumerate(explained_var):
        print(f"PC{i+1}: {var:.4f} (Cumulative: {cum_explained_var[i]:.4f})")
    
    # Check how many components explain 90% variance
    components_90 = np.argmax(cum_explained_var >= 0.90) + 1
    
    print(f"\nNumber of components needed to explain 90% of variance: {components_90}")
    
    # Generate visualization
    os.makedirs('visualizations', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum_explained_var) + 1), cum_explained_var, marker='o', linestyle='-', color='b')
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Threshold')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(range(1, len(cum_explained_var) + 1))
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join('visualizations', 'pca_variance_analysis.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved PCA scree plot to {save_path}")
    
    # Interpretation
    threshold_ratio = components_90 / len(feature_cols)
    print("\n--- PCA Recommendation ---")
    
    is_recommended = threshold_ratio < 0.7
    if is_recommended:
        print("Recommendation: PCA is recommended. A significant dimensionality reduction "
              "is possible while retaining 90% of the variance.")
    else:
        print("Recommendation: PCA is NOT strictly necessary for dimensionality reduction. "
              "You need most of the components to explain the variance, implying features are largely independent. "
              "Proceeding with original engineered features for clustering.")
              
    return is_recommended, components_90

def pca(user_features_df):
    print("\n--- Phase 3: PCA Evaluation ---")
    
    # Evaluate PCA on the available engineered customer features
    feature_cols = [col for col in user_features_df.columns if col != 'user_id']
    
    # Proceed with checking PCA necessity
    is_recommended, components_90 = check_pca_necessity(user_features_df, feature_cols)
    
    # If PCA is considered useful, perform reduction and save to CSV for optional downstream modeling
    if is_recommended:
        print(f"\nApplying PCA dimensionality reduction to {components_90} components...")
        
        # Fit and transform
        pca_model = PCA(n_components=components_90)
        reduced_features = pca_model.fit_transform(user_features_df[feature_cols].fillna(0))
        
        # Reconstruct as DataFrame with PC column names
        pca_cols = [f'PC{i+1}' for i in range(components_90)]
        reduced_df = pd.DataFrame(reduced_features, columns=pca_cols)
        
        # Append the user_id back for processing linkage
        reduced_df['user_id'] = user_features_df['user_id'].values
        
        # Reorder so user_id is first
        reduced_df = reduced_df[['user_id'] + pca_cols]
        
        # Save to memory/disk
        os.makedirs('data', exist_ok=True)
        reduced_path = os.path.join('data', 'user_features_pca_reduced.csv')
        reduced_df.to_csv(reduced_path, index=False)
        print(f"-> Successfully saved PCA-reduced dataset to '{reduced_path}'")
    
    # Return unchanged feature dataset for the standard clustering pipeline
    return user_features_df

if __name__ == "__main__":
    from dataVisualization import generate_visualizations
    
    # Load data when run as a standalone script
    orders, products, order_products, departments, aisles = load_all_data()
    
    # Run the preprocessing pipeline
    orders, products, order_products, departments, aisles = preprocess(orders, products, order_products, departments, aisles)
    
    # Feature Engineering
    user_features_df = engineer_features(orders, order_products)

    # PCA Check
    user_features_df = pca(user_features_df)

    # Data Visualization (Generates Market Visualizations on raw data)
    generate_visualizations(orders, products, order_products)

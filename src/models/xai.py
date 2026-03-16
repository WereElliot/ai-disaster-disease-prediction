import shap
import lime
import lime.lime_tabular
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from src.utils.config import load_config
from src.models.dl_models import create_lstm_dataset

# Suppress SHAP/NumPy related FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_trained_assets():
    """Load configuration and trained models."""
    config = load_config()
    model_dir = Path(config['data']['paths']['models'])
    
    with open(model_dir / "dl_feature_extractor.pkl", "rb") as f:
        dl_model = pickle.load(f)
    with open(model_dir / "xgb_classifier.pkl", "rb") as f:
        xgb_model = pickle.load(f)
        
    return config, dl_model, xgb_model

def generate_global_explanations(xgb_model, X_train, feature_names):
    """Generate SHAP global feature importance plots."""
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_train)
    
    output_dir = Path("docs/xai/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.savefig(output_dir / "shap_summary.png", bbox_inches='tight')
    plt.close()
    
    # SHAP Bar Plot with symmetric log scale to ensure visibility of all features
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)
    plt.gca().set_xscale('symlog')
    plt.title("Mean Absolute SHAP Values (Symlog Scale)")
    plt.savefig(output_dir / "shap_bar.png", bbox_inches='tight')
    plt.close()
    
    print(f"Global explanations saved to {output_dir}")

def explain_prediction(instance, xgb_model, X_train, feature_names):
    """
    Generate local explanation for a single prediction using LIME.
    
    instance: 1D array of features for a single sample.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['No Outbreak', 'Outbreak'],
        mode='classification',
        random_state=42
    )
    
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=xgb_model.predict_proba,
        num_features=10
    )
    
    output_dir = Path("docs/xai/")
    output_dir.mkdir(parents=True, exist_ok=True)
    exp.save_to_file(output_dir / "lime_local_explanation.html")
    
    return exp

def run_xai_pipeline():
    """Execute Phase 4: XAI Layer."""
    from src.models.train import add_lagged_features
    np.random.seed(42)
    config, dl_model, xgb_model = load_trained_assets()

    # Load data
    processed_path = Path(config['data']['paths']['processed']) / "final_dataset.parquet"
    df = pd.read_parquet(processed_path)

    target_col = 'cases'
    exclude_cols = [target_col, 'date', 'location', 'malaria_cases', 'cholera_cases', 'dengue_cases', 'diarrheal_cases']
    base_features = [c for c in df.columns if c not in exclude_cols]

    # Apply Lags
    df_lagged = add_lagged_features(df, target_col, base_features)
    features = [c for c in df_lagged.columns if c not in exclude_cols]
    lookback = config['models']['lstm']['lookback']

    # Prepare data (Hybrid features: Original + DL Prediction)
    X_dl, _ = create_lstm_dataset(df_lagged[features + [target_col]].values, lookback)
    dl_feats = dl_model.predict(X_dl).reshape(-1, 1)

    # Feature Names for XAI
    hybrid_feature_names = features + ["DL_Proxy_Feature"]
    X_hybrid = np.hstack([df_lagged[features].values[lookback:], dl_feats])

    # Generate Global Explanations
    X_sample = X_hybrid[:500] if len(X_hybrid) > 500 else X_hybrid
    generate_global_explanations(xgb_model, X_sample, hybrid_feature_names)

    # Generate Local Explanation for a high-risk sample
    high_risk_idx = np.argmax(xgb_model.predict_proba(X_hybrid)[:, 1])
    sample_instance = X_hybrid[high_risk_idx]
    explain_prediction(sample_instance, xgb_model, X_hybrid[:100], hybrid_feature_names)

    print("XAI Pipeline completed successfully. Plots saved to docs/xai/")
    print("XAI Pipeline completed successfully.")

if __name__ == "__main__":
    run_xai_pipeline()

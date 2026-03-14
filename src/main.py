import time
import argparse
from src.data.ingest import run_pipeline as run_ingest
from src.data.eda import run_eda
from src.models.train import train_hybrid_pipeline
from src.models.xai import run_xai_pipeline
from src.evaluation.evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="AI Disaster and Disease Prediction Platform")
    parser.add_argument('--all', action='store_true', help="Run all phases (1-5) sequentially")
    parser.add_argument('--mode', type=str, choices=['ingest', 'eda', 'train', 'explain', 'evaluate'],
                        help="Mode: ingest (data pipeline), eda (Phase 2), train (model training), explain (XAI layer), evaluate (model evaluation)")
    
    args = parser.parse_args()
    start_all = time.time()
    
    if args.all:
        print("=== [Full Pipeline Execution] ===")
        print("\n[Phase 1] Data Ingestion & Alignment...")
        run_ingest()
        
        print("\n[Phase 2] Exploratory Data Analysis...")
        run_eda()
        
        print("\n[Phase 3] Hybrid Model Training (LSTM + XGBoost)...")
        train_hybrid_pipeline()
        
        print("\n[Phase 4] Generating Explainability Artifacts (SHAP/LIME)...")
        run_xai_pipeline()
        
        print("\n[Phase 5] Final Model Evaluation...")
        run_evaluation()
        
        end_all = time.time()
        print(f"\nTotal Platform Runtime: {end_all - start_all:.2f} seconds")
        print("Verification Summary: All phases completed successfully. Reports saved in docs/.")
    
    elif args.mode == 'ingest':
        run_ingest()
    elif args.mode == 'eda':
        run_eda()
    elif args.mode == 'train':
        train_hybrid_pipeline()
    elif args.mode == 'explain':
        run_xai_pipeline()
    elif args.mode == 'evaluate':
        run_evaluation()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

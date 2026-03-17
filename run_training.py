#!/usr/bin/env python
"""Simple script to run the training pipeline."""
import sys
sys.path.insert(0, '.')

from src.models.train import train_hybrid_pipeline

if __name__ == "__main__":
    print("Starting hybrid model training...")
    try:
        train_hybrid_pipeline()
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

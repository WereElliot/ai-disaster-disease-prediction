#!/usr/bin/env python
"""Simple script to run the evaluation pipeline."""
import sys
sys.path.insert(0, '.')

from src.evaluation.evaluate import run_evaluation

if __name__ == "__main__":
    print("Starting model evaluation...")
    try:
        run_evaluation()
        print("\n✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

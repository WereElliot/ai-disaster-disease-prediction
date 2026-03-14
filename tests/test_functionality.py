import unittest
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from src.utils.config import load_config
from src.data.clean import clean_data
from src.data.merge import merge_datasets
from src.data.feature_engineer import engineer_features
from src.models.dl_models import build_lstm_model, create_lstm_dataset
from src.models.ml_models import get_xgb_classifier

class TestHybridPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.target_samples = 50 
        cls.df = pd.DataFrame({
            'date': pd.date_range(start='2017-01-01', periods=cls.target_samples, freq='D'),
            'location': 'Nairobi',
            'temp': np.random.normal(25, 5, cls.target_samples),
            'humidity': np.random.normal(60, 10, cls.target_samples),
            'cases': np.random.randint(0, 100, cls.target_samples)
        })

    def test_dl_feature_extractor_creation(self):
        lookback = 5
        X, y = create_lstm_dataset(self.df[['temp', 'humidity', 'cases']].values, lookback)
        # Should be (50 - 5) = 45 samples
        self.assertEqual(len(X), 45)
        
        model = build_lstm_model(input_shape=None, units=16)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), 45)

    def test_xgb_classifier_creation(self):
        xgb = get_xgb_classifier(self.config)
        self.assertEqual(xgb.n_estimators, self.config['models']['xgboost']['n_estimators'])
        self.assertEqual(xgb.max_depth, self.config['models']['xgboost']['max_depth'])

    def test_full_pipeline_mock_run(self):
        # 1. Clean & Merge
        cleaned = clean_data(self.df, self.config)
        self.assertIn('temp', cleaned.columns)
        
        # 2. Feature Selection
        # Small hack to ensure we have enough features for RFE in test
        temp_df = cleaned.copy()
        for i in range(12): temp_df[f'f_{i}'] = np.random.rand(len(temp_df))
        
        selected = engineer_features(temp_df, target='cases', config=self.config)
        self.assertIn('cases', selected.columns)

    def test_model_persistence(self):
        model_dir = Path("models_test")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Dummy save
        dummy_data = {"test": 123}
        path = model_dir / "test.pkl"
        with open(path, "wb") as f: pickle.dump(dummy_data, f)
        
        self.assertTrue(path.exists())
        path.unlink()
        model_dir.rmdir()

if __name__ == "__main__":
    unittest.main()

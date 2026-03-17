#!/usr/bin/env python
"""Generate synthetic training data for demo purposes."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_synthetic_data():
    """Generate synthetic health and climate data suitable for model training."""
    print("Generating synthetic training data...")
    
    # Create date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    # Locations
    locations = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']
    
    # Generate synthetic data
    data = []
    np.random.seed(42)
    for date in dates:
        for loc in locations:
            # Simulate seasonal patterns for disease cases
            day_of_year = date.timetuple().tm_yday
            seasonal_component = 50 * np.sin(2 * np.pi * day_of_year / 365)
            cases = max(0, int(100 + seasonal_component + np.random.randn() * 30))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'location': loc,
                'cases': cases,
                'hospitalizations': max(0, int(cases * 0.2 + np.random.randn() * 5)),
                'deaths': max(0, int(cases * 0.02 + np.random.randn() * 2)),
                'malaria_cases': max(0, int(cases * 0.3 + np.random.randn() * 10)),
                'cholera_cases': max(0, int(cases * 0.1 + np.random.randn() * 5)),
                'dengue_cases': max(0, int(cases * 0.2 + np.random.randn() * 8)),
                'diarrheal_cases': max(0, int(cases * 0.1 + np.random.randn() * 5)),
            })
    
    df_health = pd.DataFrame(data)
    
    # Generate climate data
    climate_data = []
    for date in dates:
        for loc in locations:
            day_of_year = date.timetuple().tm_yday
            # Simulate seasonal temperature
            temperature = 25 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.randn() * 2
            humidity = 70 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.randn() * 5
            precipitation = max(0, 5 + 10 * np.cos(2 * np.pi * day_of_year / 365) + np.random.randn() * 3)
            
            climate_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'location': loc,
                'temperature': round(temperature, 2),
                'humidity': round(max(0, min(100, humidity)), 2),
                'precipitation': round(precipitation, 2),
                'wind_speed': round(max(0, 5 + np.random.randn() * 2), 2),
            })
    
    df_climate = pd.DataFrame(climate_data)
    
    # Generate disaster data
    disaster_data = []
    disaster_types = ['Flood', 'Drought', 'Heatwave', 'Landslide']
    for i in range(50):  # 50 disaster events
        date_idx = np.random.randint(0, len(dates))
        date = dates[date_idx]
        
        disaster_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'location': np.random.choice(locations),
            'disaster_type': np.random.choice(disaster_types),
            'severity_score': np.random.uniform(1, 5),
            'affected_population': np.random.randint(1000, 100000),
        })
    
    df_disasters = pd.DataFrame(disaster_data)
    
    # Save to fixtures
    fixtures_dir = Path('tests/fixtures')
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    df_health.to_csv(fixtures_dir / 'health_sample.csv', index=False)
    df_climate.to_csv(fixtures_dir / 'climate_sample.csv', index=False)
    df_disasters.to_csv(fixtures_dir / 'disasters_sample.csv', index=False)
    
    print(f"✓ Generated health data: {df_health.shape}")
    print(f"✓ Generated climate data: {df_climate.shape}")
    print(f"✓ Generated disaster data: {df_disasters.shape}")
    print(f"✓ Saved to tests/fixtures/")
    
    return df_health, df_climate, df_disasters

if __name__ == "__main__":
    generate_synthetic_data()

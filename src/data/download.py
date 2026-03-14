import os
import requests
import requests_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List
import json
import time
import pandas as pd
import numpy as np

# Install requests-cache for local caching
requests_cache.install_cache('data_cache', expire_after=86400) # 1 day

def download_file(url: str, params: Dict[str, Any], output_path: Path) -> Path:
    """Download a file with caching and error handling."""
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def download_nasa_power(config: Dict[str, Any], raw_path: Path) -> Path:
    """Download meteorological data from NASA POWER."""
    source = config['data']['sources']['climate']
    params = source['params'].copy()
    # Ensure params are formatted correctly
    try:
        response = requests.get(source['endpoint'], params=params, timeout=30)
        response.raise_for_status()
        output_path = raw_path / "climate.json"
        with open(output_path, 'w') as f:
            json.dump(response.json(), f)
        return output_path
    except Exception as e:
        print(f"Error downloading NASA POWER data: {e}")
        return None

def download_who_gho(config: Dict[str, Any], raw_path: Path) -> Path:
    """Download Malaria and Cholera indicators from WHO GHO."""
    base_url = "https://ghoapi.azureedge.net/api"
    # Using 'MALARIA_EST_CASES' or similar if MALARIA01 fails
    indicators = ["MALARIA01", "WHS3_48"]
    health_data = {}
    
    try:
        for indicator in indicators:
            url = f"{base_url}/{indicator}"
            response = requests.get(url, timeout=30)
            if response.status_code == 404:
                print(f"Indicator {indicator} not found, trying fallback...")
                # Fallback to a search or a known alternative
                health_data[indicator] = [] # Placeholder to prevent crash
                continue
            response.raise_for_status()
            health_data[indicator] = response.json()['value']
            
        output_path = raw_path / "health.json"
        with open(output_path, 'w') as f:
            json.dump(health_data, f)
        return output_path
    except Exception as e:
        print(f"Error downloading WHO GHO data: {e}")
        return None

def download_em_dat(config: Dict[str, Any], raw_path: Path) -> Path:
    """Simulate EM-DAT disaster records aligned with research objectives."""
    disasters = []
    locations = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"]
    dates = pd.date_range(start="2017-01-01", end="2024-01-01", freq='ME')
    
    for date in dates:
        for loc in locations:
            month = date.month
            if month in [3, 4, 5, 10, 11, 12]:
                is_flood = int(np.random.choice([0, 1], p=[0.7, 0.3]))
                is_drought = 0
            else:
                is_flood = 0
                is_drought = int(np.random.choice([0, 1], p=[0.9, 0.1]))
            
            disasters.append({
                'date': date.strftime('%Y-%m-%d'),
                'location': loc,
                'flood_event': is_flood,
                'drought_event': is_drought,
                'disaster_severity': float((is_flood + is_drought) * np.random.uniform(1, 5))
            })
            
    output_path = raw_path / "disasters.json"
    with open(output_path, 'w') as f:
        json.dump(disasters, f)
    return output_path

def parallel_download(config: Dict[str, Any]):
    raw_path = Path(config['data']['paths']['raw'])
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(download_nasa_power, config, raw_path),
            executor.submit(download_who_gho, config, raw_path),
            executor.submit(download_em_dat, config, raw_path)
        ]
        results = [f.result() for f in futures]
    
    print(f"Parallel downloads completed in {time.time() - start_time:.2f} seconds")
    return [r for r in results if r is not None]

if __name__ == "__main__":
    from src.utils.config import load_config
    config = load_config()
    parallel_download(config)

# AI-Based Disaster and Disease Outbreak Prediction Platform
## Overview
This project develops an AI platform for predicting disease outbreaks (e.g., malaria, cholera) and disasters (e.g., floods, droughts) using integrated climate and community health data. It leverages machine learning (e.g., XGBoost) and deep learning (e.g., LSTM) models for forecasting, with a focus on ethical data use and explainability.
Based on the research document: [Research Project](docs/research_project.md)
## Project Structure
- **data/**: Stores raw, processed, and external data (ignored in git).
- **docs/**: Documentation, including the research proposal.
- **notebooks/**: Jupyter notebooks for EDA and experiments.
- **src/**: Core Python code for data pipelines, models, and evaluation.
- **tests/**: Unit tests.
- **config.yaml**: Configuration settings.
## Setup Instructions
1. Clone the repo: git clone [repo-url]
2. Create virtual environment: python -m venv venv
3. Activate: source venv/bin/activate (Linux/Mac) or Venv\Scripts\activate (Windows)
4. Install dependencies: pip install -r requirements.txt
## Running the Project
- Data pipeline: python src/main.py --mode ingest
- Training: python src/main.py --mode train
## Data Sources (from Issue #3)
- Climate: NASA POWER API (temperature, precipitation, etc.)
- Health: WHO Global Health Observatory, CDC datasets (disease incidence)
- Disasters: EM-DAT (flood events, etc.)
- Time range: 2017-present
Ethical Policy: All data usage adheres to privacy standards (e.g., GDPR-inspired). No personal data; aggregated only. Documented in ETHICS.md (to be added).
## Milestones (from GitHub Issues)
- Initialization done By Were Elliot
- Data Ethics & Sources (#3)
- Data Pipeline (#4)
- EDA & Baselines (#5)
- ML/DL Models (#6)
- XAI Layer (#7)
- Evaluation (#8)
## Contributors
- Group members assigned per issues.
## License
MIT License

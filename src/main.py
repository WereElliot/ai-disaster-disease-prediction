import argparse
from src.utils.config import load_config
from src.data.ingest import ingest_data
 # Placeholder imports
# Add other imports as pipelines are built
def main(args):
    config = load_config()
    if args.mode == 'ingest':
        ingest_data(config)
    # Add other modes
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='ingest', help='Mode: ingest, train, evaluate')
    args = parser.parse_args()
    main(args)

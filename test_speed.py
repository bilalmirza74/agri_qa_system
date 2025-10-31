#!/usr/bin/env python3
import time
import os
import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data_loader import DataGovINLoader
from query_processor import QueryProcessor

def main():
    # Try to load env from .env file manually
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    print("Testing query speed...")
    loader = DataGovINLoader()
    processor = QueryProcessor(loader)
    
    test_query = "Compare rice production between Punjab and Haryana"
    
    start_time = time.time()
    
    result = processor.process_query(test_query)
    
    elapsed = time.time() - start_time
    
    print(f"\nQuery completed in {elapsed:.2f} seconds")
    print("\nRESPONSE:")
    print(result.answer)

if __name__ == "__main__":
    main()


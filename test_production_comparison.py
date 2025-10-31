#!/usr/bin/env python3
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
    
    print("Initializing data loader and query processor...")
    data_loader = DataGovINLoader()
    query_processor = QueryProcessor(data_loader)
    
    print("\nTesting production comparison query...")
    test_query = "Compare rice production between Punjab and Haryana"
    
    print(f"\n{'='*80}")
    print(f"QUERY: {test_query}")
    print(f"{'='*80}")
        
    result = query_processor.process_query(test_query)
    
    print("\nRESPONSE:")
    print(result.answer)
    
    if result.data is not None and not result.data.empty:
        print("\nDATA:")
        print(result.data)
    
    if result.sources:
        print("\nSOURCES:")
        for source in result.sources:
            print(f"- {source.get('name')}: {source.get('description', '')}")

if __name__ == "__main__":
    main()

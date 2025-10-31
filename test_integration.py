import os
from dotenv import load_dotenv
from agri_qa_system.src.data_loader import DataLoader, DataSource
from agri_qa_system.src.query_processor import QueryProcessor
import pandas as pd

load_dotenv()

def test_data_loader():
    print("Testing DataLoader...")
    
    loader = DataLoader()
    
    print("\nTest 1: Loading crop production data...")
    crop_data = loader.fetch_data(
        data_source=DataSource.CROP_PRODUCTION,
        filters={"state_name": ["Punjab", "Haryana"], "crop_year": [2020, 2021]},
        limit=100
    )
    
    if not crop_data.empty:
        print(f"Successfully loaded {len(crop_data)} rows of crop production data")
        print("Sample data:")
        print(crop_data.head())
        print("\nAvailable columns:", crop_data.columns.tolist())
    else:
        print("No crop production data found. Falling back to sample data.")
    
    print("\nTest 2: Loading rainfall data...")
    rainfall_data = loader.fetch_data(
        data_source=DataSource.RAINFALL,
        filters={"State": ["Punjab", "Haryana"], "Year": [2020, 2021]},
        limit=100
    )
    
    if not rainfall_data.empty:
        print(f"Successfully loaded {len(rainfall_data)} rows of rainfall data")
        print("Sample data:")
        print(rainfall_data.head())
        print("\nAvailable columns:", rainfall_data.columns.tolist())
    else:
        print("No rainfall data found. Falling back to sample data.")

def test_query_processor():
    print("\nTesting QueryProcessor...")
    
    loader = DataLoader()
    processor = QueryProcessor(loader)
    
    test_queries = [
        "Show me wheat production in Punjab and Haryana for 2020-2022",
        "What was the rainfall in Punjab during 2021?",
        "Show me the correlation between rainfall and wheat yield in Punjab"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: Processing query - '{query}'")
        try:
            result = processor.process_query(query)
            if result:
                print("Answer:", result.answer)
                if result.data is not None and not result.data.empty:
                    print("\nData preview:")
                    print(result.data.head())
            else:
                print("No results returned")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    if not os.getenv('DATA_GOV_API_KEY'):
        print("WARNING: DATA_GOV_API_KEY environment variable not set. Using sample data only.")
    
    test_data_loader()
    test_query_processor()

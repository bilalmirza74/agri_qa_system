import os
import sys
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agri_qa_system.src.data_loader import DataLoader, DataSource

def print_data_sample(df: pd.DataFrame, source_name: str, num_rows: int = 5) -> None:
    """Print a sample of the data and its shape."""
    print(f"\n=== {source_name} Data (First {min(num_rows, len(df))} of {len(df)} rows) ===")
    if not df.empty:
        print(df.head(num_rows).to_string())
    else:
        print("No data returned")
    print("\n" + "="*80 + "\n")

def test_crop_production(loader: DataLoader) -> None:
    """Test fetching crop production data."""
    print("\n" + "="*80)
    print("TESTING CROP PRODUCTION DATA")
    print("="*80)
    
    # Test 1: Basic fetch with limit
    print("\nTest 1: Fetching latest crop production records...")
    crop_data = loader.fetch_data(
        data_source=DataSource.CROP_PRODUCTION,
        limit=5  # Just get a few records for testing
    )
    print_data_sample(crop_data, "Crop Production")
    
    # Test 2: Fetch with filters
    if not crop_data.empty:
        state = crop_data['state'].iloc[0]
        # Extract year from arrival_date (format: DD/MM/YYYY)
        year = crop_data['arrival_date'].iloc[0].split('/')[-1]
        print(f"\nTest 2: Fetching crop data for {state}...")
        filtered_data = loader.fetch_data(
            data_source=DataSource.CROP_PRODUCTION,
            filters={
                'state': state
                # Note: The API might not support filtering by arrival_date
                # If needed, we can add date filtering later
            },
            limit=3
        )
        print_data_sample(filtered_data, f"Filtered Crop Data for {state}, {year}")

def test_rainfall(loader: DataLoader) -> None:
    """Test fetching rainfall data."""
    print("\n" + "="*80)
    print("TESTING RAINFALL DATA")
    print("="*80)
    
    # Test 1: Basic fetch with limit
    print("\nTest 1: Fetching latest rainfall records...")
    rainfall_data = loader.fetch_data(
        data_source=DataSource.RAINFALL,
        limit=5
    )
    print_data_sample(rainfall_data, "Rainfall Data")
    
    # Test 2: Fetch with state filter
    if not rainfall_data.empty:
        state = rainfall_data['state'].iloc[0]
        print(f"\nTest 2: Fetching rainfall data for {state}...")
        filtered_data = loader.fetch_data(
            data_source=DataSource.RAINFALL,
            filters={
                'state': state,
                'year': '2023'  # Adjust year as needed
            },
            limit=3
        )
        print_data_sample(filtered_data, f"Filtered Rainfall Data for {state}")

def test_gis(loader: DataLoader) -> None:
    """Test fetching GIS data."""
    print("\n" + "="*80)
    print("TESTING GIS DATA")
    print("="*80)
    
    # Test 1: Basic fetch
    print("\nTest 1: Fetching GIS data...")
    gis_data = loader.fetch_data(
        data_source=DataSource.GIS,
        limit=5
    )
    print_data_sample(gis_data, "GIS Data")

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize the data loader
    api_key = os.getenv('DATA_GOV_API_KEY')
    if not api_key:
        print("Error: DATA_GOV_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key or set it in your environment.")
        return
    
    try:
        # DataLoader reads the API key from environment variables
        print("Initializing DataLoader...")
        loader = DataLoader()
        
        # Run tests for each data source
        test_crop_production(loader)
        test_rainfall(loader)
        test_gis(loader)
        
        print("\nAll tests completed successfully!")
        
        # After running tests, fetch some sample data
        print("\nFetching sample data...")
        data_source = DataSource.CROP_PRODUCTION  # This corresponds to the crop production data
        
        # Example filters - using the exact field names expected by the API
        filters = {
            "state_name": ["Punjab", "Haryana"],
            "crop_year": [2020, 2021, 2022]  # Using integers for years as expected by the API
        }
        
        # Try to fetch some data
        print(f"Fetching data for: {data_source.value}")
        data = loader.fetch_data(data_source, filters=filters, limit=10)
        
        if not data.empty:
            print("\nSample data retrieved successfully:")
            print(data.head())
            
            # Save the data to a CSV file for inspection
            output_file = os.path.join('data', 'sample_data.csv')
            data.to_csv(output_file, index=False)
            print(f"\nData saved to: {output_file}")
        else:
            print("\nNo data returned from the API.")
            
        print("\nAll operations completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing or data fetching: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure your API key is valid")
        print("2. Check that the resource ID is correct")
        print("3. Verify that your filters match the expected format")
        print("4. Check your internet connection")

if __name__ == "__main__":
    main()

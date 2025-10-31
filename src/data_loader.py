                                                                                                                                                                                                                                                                                                                                                                                                                                        import os
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from enum import Enum
import json
import glob

class DataSource(Enum):
    CROP_PRODUCTION = "crop_production"
    RAINFALL = "rainfall"
    MARKET_PRICES = "market_prices"

class APIConfig:
    """Configuration for data.gov.in APIs"""
    BASE_URL = "https://api.data.gov.in/resource"
    
    DEFAULT_RESOURCE_IDS = {
        DataSource.CROP_PRODUCTION: "35be999b-0208-4354-b557-f6ca9a5355de",  # District-wise crop production
        DataSource.RAINFALL: "9ef84268-d588-465a-a308-a864a43d0070",         # Daily rainfall data
        DataSource.MARKET_PRICES: "9ef84268-d588-465a-a308-a864a43d0070"      # Market prices (temporary same as rainfall)
    }
    
    def __init__(self):
        self.RESOURCE_IDS = {}
        for source, default_id in self.DEFAULT_RESOURCE_IDS.items():
            env_var = f"{source.value.upper()}_RESOURCE_ID"
            self.RESOURCE_IDS[source] = os.getenv(env_var, default_id)
            
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using resource IDs: {self.RESOURCE_IDS}")
    
    FIELD_MAPPINGS = {
        DataSource.CROP_PRODUCTION: {
            'state': 'state_name',
            'district': 'district_name',
            'year': 'crop_year',
            'season': 'season',
            'crop': 'crop',
            'area': 'area_',                                                                                                                                                                                                                                                                                        
            'production': 'production_',
            'yield_value': 'yield_'
        },
        DataSource.RAINFALL: {
            'state': 'State',
            'district': 'District',
            'date': 'Date',
            'year': 'Year',
            'month': 'Month',
            'rainfall': 'Avg_rainfall',
            'agency': 'Agency_name'
        },
        DataSource.MARKET_PRICES: {
            'crop': 'commercial_crop',
            'state': 'state',
            'district': 'district',
            'msp_2020': '2020',
            'msp_2021': '2021',
            'msp_2022': '2022',
            'msp_2023': '2023',
            'msp_2024': '2024',
            'commodity': 'commodity',
            'min_price': 'min_price',
            'max_price': 'max_price',
            'modal_price': 'modal_price',
            'date': 'arrival_date'
        }
    }

class DataGovINLoader:
    """Loader for fetching data from data.gov.in API"""
    
    def __init__(self, api_key: str = None):
        """Initialize the data loader
        
        Args:
            api_key: API key for data.gov.in. If not provided, will try to get from environment variable DATA_GOV_API_KEY
        """
        self.api_key = api_key or os.getenv('DATA_GOV_IN_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Please provide it or set DATA_GOV_IN_API_KEY environment variable.")
            
        self.config = APIConfig()
        self.headers = {
            'accept': 'application/json',
            'X-API-KEY': self.api_key
        }
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # Simple in-memory cache for API responses
    
    def _make_api_request(self, data_source: DataSource, params: dict = None) -> dict:
        """Make a request to the data.gov.in API
        
        Args:
            data_source: The data source to query
            params: Additional query parameters
            
        Returns:
            dict: The JSON response with 'records' key containing the data
            
        Raises:
            requests.HTTPError: If the API request fails
            ValueError: If the response cannot be parsed as JSON
        """
        if not hasattr(self.config, 'RESOURCE_IDS') or not self.config.RESOURCE_IDS:
            self.config = APIConfig()  # Reinitialize config if needed
            
        if data_source not in self.config.RESOURCE_IDS:
            raise ValueError(f"Unsupported data source: {data_source}")
            
        resource_id = self.config.RESOURCE_IDS[data_source]
        if not resource_id:
            raise ValueError(f"No resource ID configured for data source: {data_source}")
            
        url = f"{self.config.BASE_URL}/{resource_id}"
        
        # Build request params correctly
        request_params = params.copy() if params else {}
        request_params.update({
            'format': 'json',
            'api-key': self.api_key
        })
        # Set default limit if not already set
        if 'limit' not in request_params:
            request_params['limit'] = 1000
        
        # Handle filters properly - convert to filters[field_name] format
        if 'filters' in request_params and request_params['filters']:
            filters_dict = request_params.pop('filters')
            # Convert filters dict to filters[field_name] format
            for key, value in filters_dict.items():
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        value = ','.join(map(str, value))
                    request_params[f'filters[{key}]'] = str(value)
        
        # Check cache first
        cache_key = str((data_source, tuple(sorted(request_params.items()))))
        if cache_key in self.cache:
            self.logger.info(f"Cache hit for {cache_key[:50]}...")
            return self.cache[cache_key]
        
        self.logger.info(f"Making API request to: {url}")
        self.logger.info(f"Params: {request_params}")
        
        try:
            response = requests.get(
                url, 
                headers={
                    'accept': 'application/json',
                    'X-API-KEY': self.api_key
                },
                params=request_params,
                timeout=60  # Increased timeout
            )
            response.raise_for_status()
            
            data = response.json()
            self.logger.info(f"API response received with keys: {list(data.keys())}")
            
            if 'records' not in data:
                self.logger.warning("No 'records' key in API response")
                data['records'] = []
            elif not data['records']:
                self.logger.warning("Empty records list in API response")
            
            # Cache the result
            if len(self.cache) < 100:  # Limit cache size
                self.cache[cache_key] = data
                
            return data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {e}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nStatus Code: {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f"\nError: {error_data.get('message', 'No error message')}"
                except (ValueError, AttributeError, KeyError):
                    error_msg += f"\nResponse: {e.response.text[:500] if hasattr(e.response, 'text') else 'No response text'}"
            
            self.logger.error(error_msg)
            return {'records': []}
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return {'records': []}
            
        except Exception as e:
            self.logger.error(f"Unexpected error in _make_api_request: {str(e)}", exc_info=True)
            return {'records': []}
    
    def get_agriculture_data(self, state: str = None, year: int = None, 
                           crop: str = None, season: str = None, limit: int = 1000) -> pd.DataFrame:
        """Get agriculture production data from the data.gov.in API
        
        Args:
            state: Filter by state name (e.g., 'Punjab', 'Maharashtra')
            year: Filter by crop year (e.g., 2022, 2023)
            crop: Filter by crop name (e.g., 'Rice', 'Wheat')
            season: Filter by season ('Kharif'/'Rabi'/'Whole Year')
            limit: Maximum number of records to return (default: 1000, max: 10000)
            
        Returns:
            DataFrame containing the agriculture data with standardized column names
        """
        filters = {}
        if state:
            filters['state_name'] = state
        if year:
            filters['crop_year'] = str(year)
        if crop:
            filters['crop'] = crop
        if season:
            filters['season'] = season
            
        # Fetch data with pagination
        all_records = []
        page_size = 10  # API appears to limit to 10 records per request
        max_pages = min(limit // page_size + 1, 1000) if limit else 1000  # Safety limit
        
        self.logger.info(f"Fetching agriculture data with filters: {filters}, limit: {limit}")
        
        try:
            for offset in range(0, max_pages * page_size, page_size):
                params = {
                    'limit': page_size,
                    'offset': offset,
                    'filters': filters
                }
                
                response = self._make_api_request(DataSource.CROP_PRODUCTION, params)
                
                if not response or 'records' not in response or not response['records']:
                    break
                
                page_records = response['records']
                all_records.extend(page_records)
                
                self.logger.info(f"Received {len(page_records)} records at offset {offset}")
                
                # If we got fewer than page_size, we've reached the end
                if len(page_records) < page_size:
                    break
                
                # Check if we've reached the requested limit
                if limit and len(all_records) >= limit:
                    all_records = all_records[:limit]
                    break
            
            self.logger.info(f"Total records fetched: {len(all_records)}")
            
            if not all_records:
                self.logger.warning("No records found in API response")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_records)
            
            self.logger.info(f"Columns in response: {df.columns.tolist()}")
            
            column_mapping = {
                'state_name': 'state',
                'state': 'state',
                'district_name': 'district',
                'district': 'district',
                'crop_year': 'year',
                'year': 'year',
                'crop': 'crop',
                'crop_name': 'crop',
                'season': 'season',
                'production_': 'production',
                'production': 'production',
                'area_': 'area',
                'area': 'area',
                'yield_': 'yield_value',
                'yield': 'yield_value'
            }
            
            rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
            if rename_cols:
                df = df.rename(columns=rename_cols)
            
            numeric_cols = ['production', 'area', 'yield_value']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            
            required_cols = ['state', 'district', 'year', 'crop', 'season']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
                    self.logger.warning(f"Required column '{col}' not found in API response")
            
            output_cols = required_cols + [c for c in numeric_cols if c in df.columns]
            df = df[output_cols].copy()
            
            if not df.empty:
                self.logger.info(f"Sample data (first row): {df.iloc[0].to_dict()}")
                self.logger.info(f"Sample data (first 2 rows): {df.head(2).to_dict(orient='records')}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching agriculture data: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def get_rainfall_data(self, state: str = None, year: int = None, 
                         district: str = None) -> pd.DataFrame:
        """Get rainfall data
        
        Args:
            state: Filter by state
            year: Filter by year
            district: Filter by district
            
        Returns:
            DataFrame containing the rainfall data
        """
        filters = {}
        field_map = self.config.FIELD_MAPPINGS[DataSource.RAINFALL]
        
        if state:
            filters[f'filters[{field_map["state"]}]'] = state
        if year:
            filters[f'filters[{field_map["year"]}]'] = str(year)
        if district:
            filters[f'filters[{field_map["district"]}]'] = district
            
        data = self._make_api_request(DataSource.RAINFALL, filters)
        
        if not data.get('records'):
            return pd.DataFrame()
            
        df = pd.DataFrame(data['records'])
        
        reverse_map = {v: k for k, v in field_map.items()}
        df = df.rename(columns=reverse_map)
        
        if 'rainfall' in df.columns:
            df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
        
        return df
    
    def get_market_prices(self, commodity: str = None, state: str = None, 
                         start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get market price data
        
        Args:
            commodity: Filter by commodity
            state: Filter by state
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing the market price data
        """
        filters = {}
        field_map = self.config.FIELD_MAPPINGS[DataSource.MARKET_PRICES]
        
        if commodity:
            filters[f'filters[{field_map["commodity"]}]'] = commodity
        if state:
            filters[f'filters[{field_map["state"]}]'] = state
        if start_date and end_date:
            filters[f'filters[{field_map["date"]}]'] = f"{start_date},{end_date}"
            
        data = self._make_api_request(DataSource.MARKET_PRICES, filters)
        
        if not data.get('records'):
            return pd.DataFrame()
            
        df = pd.DataFrame(data['records'])
        
        reverse_map = {v: k for k, v in field_map.items()}
        df = df.rename(columns=reverse_map)
        
        price_cols = ['min_price', 'max_price', 'modal_price']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
    
    FIELD_MAPPINGS = {
        DataSource.CROP_PRODUCTION: {
            'state': 'state_name',
            'district': 'district_name',
            'year': 'crop_year',
            'season': 'season',
            'crop': 'crop',
            'area': 'area_',
            'production': 'production_',
            'yield_value': 'yield_'
        },
        DataSource.RAINFALL: {
            'state': 'State',
            'district': 'District',
            'year': 'Year',
            'month': 'Month',
            'rainfall': 'Rainfall',
            'agency': 'Agency_name'
        },
        DataSource.MARKET_PRICES: {
            'state': 'state',
            'district': 'district',
            'commodity': 'commodity',
            'min_price': 'min_price',
            'max_price': 'max_price',
            'modal_price': 'modal_price',
            'date': 'arrival_date'
        }
    }
    
    DEFAULT_PARAMS = {
        DataSource.CROP_PRODUCTION: {
            "format": "json",
            "limit": 1000,
            "offset": 0,
            "api-key": ""
        },
        DataSource.RAINFALL: {
            "format": "json",
            "limit": 1000,
            "offset": 0,
            "api-key": ""
        },
        DataSource.MARKET_PRICES: {
            "format": "json",
            "limit": 1000,
            "offset": 0,
            "api-key": ""
        }
    }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of agricultural and climate data."""
    
    def __init__(self, data_source: str = None):
        """Initialize the DataLoader.
        
        Args:
            data_source: Either 'api' to use data.gov.in API or 'local' to use local CSV files
            
        Raises:
            ValueError: If data_source is 'api' but no API key is provided
        """
        self.data_source = data_source or os.getenv('DATA_SOURCE', 'api')
        self.api_key = os.getenv('DATA_GOV_API_KEY')
        
        if self.data_source == 'api' and not self.api_key:
            raise ValueError(
                "API key is required when using API data source. "
                "Please set the DATA_GOV_API_KEY environment variable."
            )
            
        self.config = APIConfig()
        self.headers = {
            'accept': 'application/json',
            'X-API-KEY': self.api_key or ''
        }
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache = {}
        
    def _make_api_request(self, data_source: DataSource, params: dict[str, str] | None = None) -> dict[str, Any]:
        """Make an API request to data.gov.in with error handling and retries.
        
        Args:
            data_source: The data source to fetch data from
            params: Additional query parameters for filtering
            
        Returns:
            dict: The JSON response as a dictionary
            
        Raises:
            requests.HTTPError: If the API request fails
            ValueError: If the response cannot be parsed as JSON
        """
        if params is None:
            params = {}
            
        request_params = self.config.DEFAULT_PARAMS[data_source].copy()
        request_params['api-key'] = self.api_key
        
        if 'filters' in params and params['filters']:
            for key, value in params['filters'].items():
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        value = ','.join(map(str, value))
                    request_params[f'filters[{key}]'] = str(value)
        
        for key in ['limit', 'offset', 'format']:
            if key in params:
                request_params[key] = params[key]
        resource_id = self.config.RESOURCE_IDS[data_source]
        url = f"{self.config.BASE_URL}/{resource_id}"
        
        try:
            logger.info(f"Making API request to {url} with params: {request_params}")
            response = requests.get(
                url, 
                params=request_params, 
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError(f"Unexpected response format: {data}")
                return data
            except ValueError as e:
                logger.error(f"Failed to parse JSON response: {response.text}")
                raise ValueError(f"Invalid JSON response: {e}")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {e}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nStatus Code: {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg += f"\nError: {error_data.get('message', 'No error message')}"
                except (ValueError, AttributeError, KeyError):
                    error_msg += f"\nResponse: {e.response.text[:500] if hasattr(e.response, 'text') else 'No response text'}"  # Truncate long responses
            
            logger.error(error_msg)
            raise requests.HTTPError(error_msg) from e
    
    def _get_cached_data(self, filename: str, max_age_days: int = 7) -> Optional[dict]:
        """Get cached data if it exists and is not too old."""
        cache_path = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(cache_path):
            file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))).days
            if file_age <= max_age_days:
                with open(cache_path, 'r') as f:
                    return json.load(f)
        return None
    
    def _save_to_cache(self, data: dict, filename: str) -> None:
        """Save data to cache."""
        cache_path = os.path.join(self.cache_dir, filename)
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def _load_local_csv(self, file_pattern: str = '*.csv') -> pd.DataFrame:
        """Load data from local CSV files matching the given pattern.
        
        Args:
            file_pattern: Glob pattern to match CSV files in the data directory
            
        Returns:
            Combined DataFrame containing data from all matching CSV files
        """
        try:
            cache_key = f"local_csv_{file_pattern}"
            if cache_key in self.cache:
                return self.cache[cache_key].copy()
                
            csv_files = glob.glob(os.path.join(self.data_dir, file_pattern))
            
            if not csv_files and 'market_prices' in file_pattern:
                cleaned_file = os.path.join(self.data_dir, 'market_prices_cleaned.csv')
                if os.path.exists(cleaned_file):
                    csv_files = [cleaned_file]
                else:
                    original_file = os.path.join(self.data_dir, '35985678-0d79-46b4-9ed6-6f13308a1d24_2b8d7ca6b279f25493cd9fc72b1d8a69.csv')
                    if os.path.exists(original_file):
                        import subprocess
                        subprocess.run(['python', 'clean_csv.py'], cwd=os.path.dirname(self.data_dir))
                        if os.path.exists(cleaned_file):
                            csv_files = [cleaned_file]
            
            if not csv_files:
                logger.warning(f"No CSV files found matching pattern: {file_pattern}")
                return pd.DataFrame()
                
            dfs = []
            for file in csv_files:
                try:
                    if 'market_prices' in file.lower() or '35985678' in file:
                        df = pd.read_csv(
                            file, 
                            parse_dates=['Arrival_Date'],
                            dayfirst=True,
                            dtype={'Min_Price': float, 'Max_Price': float, 'Modal_Price': float}
                        )
                        for col in df.select_dtypes(include=['object']).columns:
                            df[col] = df[col].str.strip()
                    else:
                        df = pd.read_csv(file)
                    
                    df['_source_file'] = os.path.basename(file)
                    dfs.append(df)
                    logger.info(f"Successfully loaded {len(df)} rows from {file}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
            
            if not dfs:
                return pd.DataFrame()
                
            result = pd.concat(dfs, ignore_index=True)
            
            self.cache[cache_key] = result
            
            return result.copy()
            
        except Exception as e:
            logger.error(f"Error loading CSV files: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def fetch_data(
        self, 
        data_source: DataSource, 
        filters: dict = None, 
        limit: int = 1000,
        offset: int = 0, 
        file_pattern: str = None,
        max_retries: int = 3,
        max_records: int = 10000
    ) -> pd.DataFrame:
        """Fetch data from the specified data source with pagination and retries.
        
        Args:
            data_source: The data source to fetch data from
            filters: Dictionary of filters to apply (mapped to API field names)
            limit: Maximum number of records to fetch per request (max 1000)
            offset: Starting record number for pagination
            file_pattern: Pattern for local CSV files (used if data_source is 'local')
            max_retries: Maximum number of retry attempts for failed requests
            max_records: Maximum total records to fetch (to prevent excessive API usage)
            
        Returns:
            pd.DataFrame: The fetched data as a DataFrame with standardized column names
            
        Raises:
            ValueError: If data_source is invalid or no data is found
        """
        if data_source not in DataSource:
            raise ValueError(f"Invalid data source: {data_source}")
            
        if self.data_source == 'local':
            if file_pattern is None:
                if data_source == DataSource.CROP_PRODUCTION:
                    file_pattern = '*crop*.csv'
                elif data_source == DataSource.RAINFALL:
                    file_pattern = '*rainfall*.csv'
                else:
                    file_pattern = '*.csv'
            return self._load_local_csv(file_pattern)
        
        all_records = []
        current_offset = offset
        remaining = min(limit, max_records) if limit else max_records
        
        try:
            while remaining > 0:
                fetch_size = min(remaining, 1000)
                
                params = {
                    'filters': {},
                    'limit': fetch_size,
                    'offset': current_offset,
                    'format': 'json'
                }
                
                if filters:
                    field_mapping = self.config.FIELD_MAPPINGS[data_source]
                    for key, value in filters.items():
                        if key in field_mapping:
                            api_field = field_mapping[key]
                            if isinstance(value, (list, tuple)):
                                params['filters'][api_field] = ",".join(str(v) for v in value)
                            else:
                                params['filters'][api_field] = str(value)
                
                data = self._make_api_request(data_source, params)
                if not data or 'records' not in data:
                    break
                    
                df = pd.DataFrame(data['records'])
                
                if len(df) > 0:
                    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
                
                numeric_columns = []
                if data_source == DataSource.CROP_PRODUCTION:
                    numeric_columns = ['area_', 'production_', 'yield_', 'crop_year']
                elif data_source == DataSource.RAINFALL:
                    numeric_columns = ['rainfall', 'year', 'month']
                    
                for col in numeric_columns:
                    if col in df.columns and not df[col].empty:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Error converting column '{col}' to numeric: {e}")
                            
                all_records.append(df)
                
                current_offset += fetch_size
                remaining -= fetch_size
                
                if len(df) < fetch_size:
                    break
            
            if all_records:
                result = pd.concat(all_records, ignore_index=True)
                cache_key = f"{data_source.value}_{str(filters)}_{limit}_{offset}"
                self.cache[cache_key] = result
                return result.copy()
            
            return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def load_agriculture_data(self, states: List[str] = None, years: List[int] = None) -> Dict[str, pd.DataFrame]:
        """Load agriculture and climate data from data.gov.in API.
        
        Args:
            states: List of state names to filter by
            years: List of years to filter by
            
        Returns:
            Dictionary containing DataFrames for each data type
        """
        data = {}
        
        try:
            crop_filters = {}
            if states:
                crop_filters['state'] = states
            if years:
                crop_filters['year'] = years
                
            crop_data = self.fetch_data(
                data_source=DataSource.CROP_PRODUCTION,
                filters=crop_filters,
                limit=10000
            )
            
            if not crop_data.empty:
                crop_data = crop_data.rename(columns={
                    'state_name': 'state',
                    'district_name': 'district',
                    'crop_year': 'year',
                    'production_': 'production',
                    'area_': 'area'
                })
                
                numeric_cols = ['production', 'area', 'yield_']
                for col in numeric_cols:
                    if col in crop_data.columns:
                        crop_data[col] = pd.to_numeric(crop_data[col], errors='coerce')
                
                data['crop_production'] = crop_data

            rainfall_filters = {}
            if states:
                rainfall_filters['State'] = states
            if years:
                rainfall_filters['Year'] = years
                
            rainfall_data = self.fetch_data(
                data_source=DataSource.RAINFALL,
                filters=rainfall_filters,
                limit=10000
            )
            
            if not rainfall_data.empty:
                rainfall_data = rainfall_data.rename(columns={
                    'State': 'state',
                    'District': 'district',
                    'Year': 'year',
                    'Rainfall': 'rainfall'
                })
                
                numeric_cols = ['rainfall', 'year', 'month']
                for col in numeric_cols:
                    if col in rainfall_data.columns:
                        rainfall_data[col] = pd.to_numeric(rainfall_data[col], errors='coerce')
                
                data['rainfall'] = rainfall_data
                
        except Exception as e:
            logger.error(f"Error loading agriculture data: {e}")
            if 'crop_production' not in data:
                data['crop_production'] = pd.DataFrame()
            if 'rainfall' not in data:
                data['rainfall'] = pd.DataFrame()
        
        return data
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    loader = DataLoader()
    
    print("Fetching crop production data...")
    crop_data = loader.load_agriculture_data(
        states=["Punjab", "Haryana"],
        years=[2020, 2021, 2022]
    )
    
    if 'crop_production' in crop_data and not crop_data['crop_production'].empty:
        print("\nCrop Production Data:")
        print(crop_data['crop_production'].head())
    
    if 'rainfall' in crop_data and not crop_data['rainfall'].empty:
        print("\nRainfall Data:")
        print(crop_data['rainfall'].head())
    
    print("\nAvailable columns in crop data:", crop_data['crop_production'].columns.tolist() if 'crop_production' in crop_data else "No data")
    print("Available columns in rainfall data:", crop_data['rainfall'].columns.tolist() if 'rainfall' in crop_data else "No data")

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re
import logging
import pandas as pd

from data_loader import DataGovINLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Container for query results with metadata."""
    answer: str
    data: Optional[pd.DataFrame] = None
    sources: List[Dict[str, str]] = None
    visualization: Optional[Dict[str, Any]] = None

    visualizations: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

class QueryProcessor:
    """Processes natural language queries and generates responses using data from DataGovINLoader."""
    
    def __init__(self, data_loader: DataGovINLoader):
        """Initialize the query processor with a data loader.
        
        Args:
            data_loader: An instance of DataGovINLoader for fetching data
        """
        self.data_loader = data_loader
        self.entity_extractors = {
            'state': self._extract_states,
            'year': self._extract_years,
            'crop': self._extract_crops,
            'district': self._extract_districts,
            'commodity': self._extract_commodities
        }

    def _extract_states(self, query: str) -> List[str]:
        """Extract state names from the query."""
        indian_states = [
            'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
            'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
            'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
            'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
            'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal',
            'andaman and nicobar islands', 'chandigarh', 'dadra and nagar haveli and daman and diu',
            'delhi', 'jammu and kashmir', 'ladakh', 'lakshadweep', 'puducherry'
        ]
        
        return [state for state in indian_states if state.lower() in query.lower()]
    
    def _extract_years(self, query: str) -> List[int]:
        """Extract years from the query."""
        year_matches = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', query)
        return [int(year) for year in year_matches]
    
    def _extract_crops(self, query: str) -> List[str]:
        """Extract crop names from the query."""
        crops = [
            'rice', 'wheat', 'maize', 'sugarcane', 'cotton', 'soybean', 'pulses',
            'millets', 'oilseeds', 'jute', 'coffee', 'tea', 'spices', 'vegetables',
            'fruits', 'potato', 'onion', 'tomato', 'mango', 'banana', 'grapes',
            'apple', 'orange', 'groundnut', 'mustard', 'sunflower', 'safflower'
        ]
        return [crop for crop in crops if crop.lower() in query.lower()]
    
    def _extract_districts(self, query: str) -> List[str]:
        """Extract district names from the query."""
        districts = [
            'mumbai', 'bengaluru', 'hyderabad', 'ahmedabad', 'chennai', 'kolkata',
            'surat', 'pune', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore',
            'thane', 'bhopal', 'visakhapatnam', 'patna', 'vadodara', 'ghaziabad',
            'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'varanasi'
        ]
        return [district for district in districts if district.lower() in query.lower()]
    
    def _extract_commodities(self, query: str) -> List[str]:
        """Extract commodity names from the query."""
        commodities = [
            'rice', 'wheat', 'sugar', 'potato', 'onion', 'tomato', 'pulses',
            'oilseeds', 'mustard', 'soybean', 'groundnut', 'maize', 'jowar',
            'bajra', 'ragi', 'barley', 'turmeric', 'chili', 'coriander',
            'cumin', 'pepper', 'cardamom', 'ginger', 'garlic', 'tamarind'
        ]
        return [commodity for commodity in commodities if commodity.lower() in query.lower()]
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities like states, years, crops from the query."""
        entities = {
            'states': self._extract_states(query),
            'districts': self._extract_districts(query),
            'crops': self._extract_crops(query),
            'commodities': self._extract_commodities(query),
            'years': self._extract_years(query),
            'price_types': [],
            'comparison': 'compare' in query.lower() or 'vs' in query.lower() or 'versus' in query.lower(),
            'time_range': None,
            'limit': 5
        }
        
        indian_states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
                        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
                        'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
                        'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 
                        'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
        
        for state in indian_states:
            if state.lower() in query.lower() and state not in entities['states']:
                entities['states'].append(state)
        
        if 'district' in query.lower():
            district_match = re.search(r'in (\w+)(?:\s+district)?', query.lower())
            if district_match:
                district = district_match.group(1).title()
                if district not in entities['districts']:
                    entities['districts'].append(district)
        
        common_crops = ['rice', 'wheat', 'maize', 'sugarcane', 'cotton', 'tomato', 
                       'potato', 'onion', 'apple', 'banana', 'mango', 'grape']
        for crop in common_crops:
            crop_name = crop.capitalize()
            if crop in query.lower() and crop_name not in entities['crops']:
                entities['crops'].append(crop_name)
        
        price_terms = {
            'min': ['minimum', 'lowest', 'min'],
            'max': ['maximum', 'highest', 'max'],
            'modal': ['average', 'modal', 'price', 'normal']
        }
        
        for price_type, terms in price_terms.items():
            if any(term in query.lower() for term in terms):
                if price_type not in entities['price_types']:
                    entities['price_types'].append(price_type)
        
        if not entities['price_types'] and any(term in query.lower() for term in ['price', 'cost']):
            entities['price_types'] = ['modal']
            
        comparison_terms = ['compare', 'versus', 'vs', 'difference', 'between', 'vs.']
        if any(term in query.lower() for term in comparison_terms):
            entities['comparison'] = True
            
        if 'last' in query.lower() and ('year' in query.lower() or 'month' in query.lower()):
            match = re.search(r'last (\d+) (year|month)', query.lower())
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                if unit == 'year':
                    entities['time_range'] = f"last {value} years"
                else:
                    entities['time_range'] = f"last {value} months"
        
        limit_match = re.search(r'top (\d+)', query.lower())
        if limit_match:
            entities['limit'] = int(limit_match.group(1))
            
        return entities
    
    def _generate_sql_query(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL-like query from extracted entities."""
        query = {
            'select': ['*'],  
            'from': ['market_prices'],
            'where': [],
            'group_by': [],
            'order_by': [],
            'limit': entities.get('limit', 10),
            'state': None,
            'district': None,
            'commodity': None
        }
        
        try:
            if entities.get('states'):
                query['state'] = entities['states'][0]  # Take first state if multiple
                states = [f"state = '{state}'" for state in entities['states']]
                query['where'].append(f"({' OR '.join(states)})")
                
            if entities.get('districts'):
                query['district'] = entities['districts'][0]  # Take first district if multiple
                districts = [f"district = '{district}'" for district in entities['districts']]
                query['where'].append(f"({' OR '.join(districts)})")
                
            if entities.get('crops'):
                query['commodity'] = entities['crops'][0]  # Take first crop if multiple
                crops = [f"commodity = '{crop}'" for crop in entities['crops']]
                query['where'].append(f"({' OR '.join(crops)})")
            
            if entities.get('comparison') and len(entities.get('states', [])) >= 2:
                query['order_by'].append('state')
                
        except Exception as e:
            self.logger.error(f"Error generating SQL query: {str(e)}")
            raise ValueError(f"Error processing your query. Please try again with different parameters.")
            
        return query
    
    def _execute_query(self, query: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Execute the query against the data sources."""
        results = {}
        
        try:
            if 'market_prices' in query.get('from', []):
                df = self.data_loader.get_market_prices(
                    commodity=query.get('commodity'),
                    state=query.get('state'),
                    start_date=query.get('start_date'),
                    end_date=query.get('end_date')
                )
                
                if query.get('where'):
                    for where_clause in query['where']:
                        if 'State' in where_clause:
                            states = [s.strip() for s in where_clause.split('=')[1].strip("()").split('OR')]
                            states = [s.strip(" '\"") for s in states]
                            if not df.empty and 'State' in df.columns:
                                df = df[df['State'].isin(states)]
                        elif 'District' in where_clause:
                            districts = [d.strip() for d in where_clause.split('=')[1].strip("()").split('OR')]
                            districts = [d.strip(" '\"") for d in districts]
                            if not df.empty and 'District' in df.columns:
                                df = df[df['District'].isin(districts)]
                        elif 'Commodity' in where_clause:
                            crops = [c.strip() for c in where_clause.split('=')[1].strip("()").split('OR')]
                            crops = [c.strip(" '\"") for c in crops]
                            if not df.empty and 'Commodity' in df.columns:
                                df = df[df['Commodity'].str.lower().isin([c.lower() for c in crops])]
                
                if query.get('order_by') and not df.empty:
                    order_by = query['order_by']
                    if isinstance(order_by, list):
                        order_by = [col for col in order_by if col in df.columns]
                        if order_by:
                            df = df.sort_values(by=order_by)
                    elif order_by in df.columns:
                        df = df.sort_values(by=order_by)
                
                if query.get('limit') and not df.empty:
                    df = df.head(query['limit'])
                
                results['market_prices'] = df
                
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.exception(e)
            raise ValueError(f"Error processing your query: {str(e)}")
            
        return results
    
    def _generate_response(self, query: str, entities: Dict[str, Any], 
                         data: Dict[str, pd.DataFrame]) -> QueryResult:
        """Generate a natural language response from the query and data."""
        answer_parts = []
        sources = []
        visualizations = []
        
        if not data or all(df.empty for df in data.values()):
            return QueryResult(
                answer="I couldn't find any data matching your query. Please try a different query or check your data files.",
                sources=sources
            )
            
        for source_name, df in data.items():
            if not df.empty:
                answer_parts.append(f"Found {len(df)} records of market prices.")
                sources.append({
                    'name': 'data.gov.in - Agricultural Market Prices',
                    'url': 'https://data.gov.in/',
                    'description': 'Government dataset: Agricultural Market Prices'
                })
                
                if entities['comparison'] and len(entities['states']) >= 2:
                    state1, state2 = entities['states'][:2]
                    answer_parts.append(f"\nComparison between {state1} and {state2}:")
                    
                    state1_data = df[df['State'] == state1]
                    state2_data = df[df['State'] == state2]
                    
                    commodities = df['Commodity'].unique()
                    
                    for commodity in entities['crops'] or commodities[:3]:
                        comm_data = df[df['Commodity'].str.lower() == commodity.lower()]
                        if comm_data.empty:
                            continue
                            
                        answer_parts.append(f"\n{commodity}:")
                        
                        for price_type in entities['price_types']:
                            if price_type == 'min':
                                price_col = 'Min_Price'
                                desc = 'minimum price'
                            elif price_type == 'max':
                                price_col = 'Max_Price'
                                desc = 'maximum price'
                            else: 
                                price_col = 'Modal_Price'
                                desc = 'modal price'
                            
                            try:
                                price1 = state1_data[price_col].mean()
                                price2 = state2_data[price_col].mean()
                                
                                if not pd.isna(price1) and not pd.isna(price2):
                                    answer_parts.append(
                                        f"- Average {desc}: {state1} = ₹{price1:.2f}, {state2} = ₹{price2:.2f}"
                                    )
                            except Exception as e:
                                logger.warning(f"Error calculating {desc}: {e}")
                
                elif not entities['comparison']:
                    answer_parts.append("\nSummary of market prices:")
                    
                    for price_type in entities['price_types']:
                        if price_type == 'min':
                            price_col = 'Min_Price'
                            desc = 'Minimum price'
                        elif price_type == 'max':
                            price_col = 'Max_Price'
                            desc = 'Maximum price'
                        else: 
                            price_col = 'Modal_Price'
                            desc = 'Average price'
                        
                        try:
                            if not df.empty and price_col in df.columns:
                                avg_price = df[price_col].mean()
                                if not pd.isna(avg_price):
                                    answer_parts.append(f"- {desc}: ₹{avg_price:.2f}")
                        except Exception as e:
                            logger.warning(f"Error calculating {desc}: {e}")
                    
                    try:
                        if not df.empty and 'Modal_Price' in df.columns:
                            top_commodities = df.groupby('Commodity')['Modal_Price'].mean().nlargest(3)
                            if not top_commodities.empty:
                                answer_parts.append("\nTop commodities by average price:")
                                for commodity, price in top_commodities.items():
                                    answer_parts.append(f"- {commodity}: ₹{price:.2f}")
                    except Exception as e:
                        logger.warning(f"Error finding top commodities: {e}")
            
        if len(answer_parts) <= 1:
            answer_parts.append("\nTry being more specific with your query, for example:")
            answer_parts.append("- 'Show me tomato prices in Maharashtra'")
            answer_parts.append("- 'Compare rice prices between Punjab and Haryana'")
            answer_parts.append("- 'What are the maximum prices for onions?'")
        
        if len(entities.get('states', [])) == 2 and 'metric' in entities:
            state1, state2 = entities['states']
            metric = entities['metric']
        elif len(entities['years']) > 1 and 'crop_production' in data:
            df = data['crop_production']
            if not df.empty and 'year' in df.columns:
                answer_parts.append("\nTrend over time:")
                for metric in entities['metrics']:
                    if metric in df.columns:
                        trend = df.groupby('year')[metric].mean()
                        answer_parts.append(f"- {metric.capitalize()}: {trend.idxmin()} to {trend.idxmax()}")
                        
                        visualizations.append({
                            'type': 'line',
                            'title': f'{metric.capitalize()} Over Time',
                            'data': {
                                'labels': trend.index.tolist(),
                                'datasets': [{
                                    'label': metric.capitalize(),
                                    'data': trend.values.tolist(),
                                    'fill': False
                                }]
                            }
                        })
                        
        if 'top' in query.lower() and 'crop_production' in data:
            df = data['crop_production']
            if not df.empty and 'crop' in df.columns and 'production' in df.columns:
                top_n = min(entities.get('limit', 5), len(df))
                top_crops = df.nlargest(top_n, 'production')[['crop', 'production']]
                answer_parts.append(f"\nTop {top_n} crops by production:")
                for _, row in top_crops.iterrows():
                    answer_parts.append(f"- {row['crop']}: {row['production']:,.2f}")
                    
        return QueryResult(
            answer='\n'.join(answer_parts),
            data=pd.concat(data.values()) if data else None,
            sources=sources,
            visualizations=visualizations,
            metadata={
                'entities': entities,
                'query': query
            }
        )
    
    def process_query(self, query: str) -> QueryResult:
        """Process a natural language query and return a response.
        
        Args:
            query: The user's natural language query
            
        Returns:
            QueryResult containing the response and any relevant data
        """
        try:
            entities = self._extract_entities(query)
            
            agri_data = self.data_loader.get_agriculture_data(
                state=None,
                year=None,
                crop=None,
                season=None,
                limit=1000
            )
            
            if not any(entities.values()):
                unique_crops = agri_data['crop'].unique().tolist() if not agri_data.empty else []
                unique_states = agri_data['state'].unique().tolist() if not agri_data.empty else []
                unique_years = agri_data['year'].unique().tolist() if not agri_data.empty else []
                
                help_message = """
Please be more specific in your query. Here's what data is available:

**Available Crops:** {crops}

**Available States:** {states}

**Available Years:** {years}

**Example Queries:**
- Show rice production in Maharashtra
- Compare wheat and rice production in Punjab and Haryana
- What was the production of sugarcane in 2020?
""".format(
                    crops=", ".join(unique_crops[:10]) + ("..." if len(unique_crops) > 10 else ""),
                    states=", ".join(unique_states[:5]) + ("..." if len(unique_states) > 5 else ""),
                    years=", ".join(map(str, sorted(unique_years)[-5:])) + ("..." if len(unique_years) > 5 else "")
                )
                
                sample_data = agri_data.head(5) if not agri_data.empty else None
                
                return QueryResult(
                    answer=help_message,
                    data=sample_data,
                    sources=[{"name": "Ministry of Agriculture & Farmers Welfare", "url": "https://data.gov.in"}]
                )
            
            sql_query = self._generate_sql_query(entities)
            
            results = self._execute_query(sql_query)
            
            return self._generate_response(query, entities, results)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            
            error_message = f"""
Sorry, I encountered an error processing your query: {str(e)}

**Troubleshooting Tips:**
- Try being more specific with your query
- Check your spelling of crop, state, or district names
- Try a different time period
- Example: "Show rice production in Maharashtra in 2020"
"""
            
            return QueryResult(
                answer=error_message,
                data=None,
                sources=[]
            )
        years = entities.get('year', [])
        states = entities.get('state', [])
        crops = entities.get('crop', [])
        
        if not crops:
            return QueryResult(
                answer="Please specify a crop to get production data.",
                sources=[]
            )
        
        crop = crops[0]
        
        try:
            df = self.data_loader.get_agriculture_data(
                state=states[0] if states else None,
                crop=crop
            )
            
            if df.empty:
                return QueryResult(
                    answer=f"I couldn't find any production data for {crop}.",
                    sources=["data.gov.in - Crop Production Data"]
                )
            
            if years:
                df = df[df['year'].astype(str).isin([str(y) for y in years])]
            
            prod_by_year = df.groupby('year')['production'].sum().reset_index()
            prod_by_year = prod_by_year.sort_values('year')
            
            if len(prod_by_year) < 2:
                return QueryResult(
                    answer=f"Not enough data points to show trend for {crop}.",
                    sources=["data.gov.in - Crop Production Data"]
                )
            
            first_year = prod_by_year['year'].iloc[0]
            last_year = prod_by_year['year'].iloc[-1]
            first_prod = prod_by_year['production'].iloc[0]
            last_prod = prod_by_year['production'].iloc[-1]
            
            trend = "increased" if last_prod > first_prod else "decreased"
            pct_change = ((last_prod - first_prod) / first_prod * 100) if first_prod > 0 else 0
            
            response = (
                f"Production of {crop} {trend} by {abs(pct_change):.1f}% "
                f"from {first_year} to {last_year}."
            )
            
            return QueryResult(
                answer=response,
                data=prod_by_year,
                sources=["data.gov.in - Crop Production Data"]
            )
            
        except Exception as e:
            logger.error(f"Error in _handle_production_trend: {str(e)}", exc_info=True)
            return QueryResult(
                answer=f"Error processing your query: {str(e)}",
                sources=["data.gov.in - Crop Production Data"]
            )
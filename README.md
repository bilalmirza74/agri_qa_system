# Agri-QA System

An intelligent Q&A system for analyzing Indian agricultural and climate data from data.gov.in.

## Features

- Natural language query interface for agricultural and climate data
- Integration with multiple data sources from data.gov.in
- Data visualization and analysis
- Source citation for all data points

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DATA_GOV_API_KEY=your_data_gov_api_key
   ```
4. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```

## Project Structure

- `data/`: Contains sample datasets and cached data
- `notebooks/`: Jupyter notebooks for data exploration
- `src/`: Source code for the application
  - `data_loader.py`: Handles data loading and preprocessing
  - `query_processor.py`: Processes natural language queries
  - `app.py`: Streamlit web application

## Data Sources

- [data.gov.in](https://data.gov.in)
- India Meteorological Department (IMD)
- Ministry of Agriculture & Farmers Welfare

#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader import DataLoader, DataGovINLoader, DataSource
from query_processor import QueryProcessor

load_dotenv()

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def get_data_loader():
    try:
        loader = DataGovINLoader()
        test_data = loader.get_agriculture_data(limit=1)
        if test_data is None or test_data.empty:
            st.warning("API connection successful but no data returned. Check your query parameters.")
        return loader
    except Exception as e:
        st.error(f"Failed to initialize data loader: {str(e)}")
        try:
            local_loader = DataLoader(data_source='local')
            return local_loader
        except Exception as e2:
            st.error(f"Also failed to initialize local data loader: {str(e2)}")
            return None

@st.cache_resource
def get_query_processor():
    try:
        data_loader = DataGovINLoader()
        test_data = data_loader.get_agriculture_data()
        if test_data is None or test_data.empty:
            st.warning("API connection successful but no data returned. Check your API key and parameters.")
        return QueryProcessor(data_loader)
    except Exception as e:
        st.error(f"Failed to initialize query processor: {str(e)}")
        try:
            local_loader = DataLoader(data_source='local')
            return QueryProcessor(local_loader)
        except Exception as e2:
            st.error(f"Also failed to initialize local data loader: {str(e2)}")
            return None

def main():
    st.set_page_config(
        page_title="Agri-QA System",
        page_icon="🌾",
        layout="wide"
    )
    
    query_processor = get_query_processor()
    
    if query_processor is None:
        st.error("Failed to initialize the query processor. Please check the logs for more details.")
        st.stop()
    
    st.title("🌾 Agri-QA System")
    st.write("Ask questions about Indian agricultural and climate data.")
    
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""
    
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    query_input = st.text_input("Ask a question:", 
                              value=st.session_state.query_input if 'query_input' in st.session_state else "",
                              key="query_input")
    
    process_query = False
    
    if st.button("Ask"):
        if query_input:
            st.session_state.query = query_input
            process_query = True
    elif 'last_query' in st.session_state and st.session_state.last_query:
        st.session_state.query = st.session_state.last_query
        st.session_state.query_input = st.session_state.last_query
        st.session_state.last_query = None
        process_query = True
        st.rerun()
    
    query = st.session_state.query if st.session_state.query else query_input
    
    if process_query and query:
        with st.spinner("Processing your query..."):
            try:
                result = query_processor.process_query(query)
                
                st.session_state.history.insert(0, {
                    'query': query,
                    'answer': result.answer if hasattr(result, 'answer') else str(result),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.session_state.query = ""
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.session_state.query = ""
            
            if len(st.session_state.history) > 10:
                st.session_state.history = st.session_state.history[:10]
            
            st.subheader("Results")
            
            if hasattr(result, 'answer') and result.answer:
                st.write("### Answer")
                st.write(result.answer)
                st.write("---")
                
            if hasattr(result, 'data') and result.data is not None and not result.data.empty:
                st.write("### Data")
                st.dataframe(result.data)
                
                csv = result.data.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='agriculture_data.csv',
                    mime='text/csv',
                )
                st.write("---")
                
            if hasattr(result, 'visualization') and result.visualization:
                st.write("### Visualization")
                if result.visualization.get('type') == 'table':
                    st.table(result.visualization.get('data'))
                elif result.visualization.get('type') == 'bar':
                    st.bar_chart(result.visualization.get('data'))
                elif result.visualization.get('type') == 'line':
                    st.line_chart(result.visualization.get('data'))
                else:
                    st.write("Unsupported visualization type")
                st.write("---")
                
                if hasattr(result, 'sources') and result.sources:
                    st.subheader("Sources")
                    for source in result.sources:
                        if isinstance(source, dict):
                            name = source.get('name', 'Source')
                            url = source.get('url', '#')
                            description = source.get('description', '')
                            st.markdown(f"- [{name}]({url}): {description}" if url else f"- {name}: {description}")
                        else:
                            st.markdown(f"- {source}")
                
                if hasattr(result, 'visualizations') and result.visualizations:
                    st.subheader("Visualizations")
                    for viz in result.visualizations:
                        if viz['type'] == 'table':
                            st.dataframe(viz['data'])
                        elif viz['type'] == 'bar':
                            fig = px.bar(
                                x=viz['data']['labels'],
                                y=viz['data']['datasets'][0]['data'],
                                title=viz['title'],
                                labels={'x': 'Category', 'y': 'Value'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif viz['type'] == 'line':
                            fig = px.line(
                                x=viz['data']['labels'],
                                y=viz['data']['datasets'][0]['data'],
                                title=viz['title'],
                                labels={'x': 'Year', 'y': viz['data']['datasets'][0]['label']}
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(result)
    
    with st.sidebar:
        if st.session_state.history:
            st.subheader("History")
            for i, item in enumerate(st.session_state.history[:10]):
                if st.button(f"{item['query'][:50]}...", key=f"history_{i}"):
                    st.session_state.last_query = item['query']
        
        st.markdown("""
        ### About
        This is an intelligent Q&A system for Indian agricultural and climate data.

        ### Examples
        - Show me the top 5 crops by production in Maharashtra
        - What was the rainfall in Karnataka in 2020?

        ### Data Sources
        - Ministry of Agriculture & Farmers Welfare
        - India Meteorological Department
        - Data.gov.in
        """)
        
        st.subheader("Sample Queries")
        sample_queries = [
            "Show me the top 5 crops by production in Maharashtra",
            "What was the rainfall in Karnataka in 2020?",
            "Show wheat production trend in Uttar Pradesh for last 5 years",
            "Compare average rainfall between Kerala and Tamil Nadu"
        ]
        
        for sample in sample_queries:
            if st.button(sample, key=f"sample_{sample[:10]}"):
                st.session_state.last_query = sample
                st.session_state.query_input = sample
                st.rerun()

if __name__ == "__main__":
    main()
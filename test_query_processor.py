from dotenv import load_dotenv
from agri_qa_system.src.data_loader import DataLoader
from agri_qa_system.src.query_processor import QueryProcessor

def main():
    load_dotenv()
    
    data_loader = DataLoader()
    query_processor = QueryProcessor(data_loader)
    
    test_queries = [
        "Show me wheat production in Punjab and Haryana for 2021-2022",
        "Compare rainfall between Punjab and Haryana in 2022",
        "What is the trend of rice production in Punjab?",
        "Correlate rainfall with wheat yield in Haryana"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
            
        result = query_processor.process_query(query)
        
        print("\nRESPONSE:")
        print(result.answer)
        
        if result.data is not None and not result.data.empty:
            print("\nDATA PREVIEW:")
            print(result.data.head())
        
        if result.sources:
            print("\nSOURCES:")
            for source in result.sources:
                print(f"- {source.get('name')}: {source.get('description')}")

if __name__ == "__main__":
    main()

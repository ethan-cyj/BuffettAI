import argparse
import os
from data_utils import load_documents, load_all_documents, inspect_file
from rag.rag_pipeline import RAGPipeline
from rag.retrieval import initialize_faiss_databases
from rag.retrieval import main_intialise_retrievers

def main():
    print(f"Script running from: {os.getcwd()}")

    parser = argparse.ArgumentParser(description="BuffettAI RAG Pipeline")
    
    
    # Other arguments
    parser.add_argument("--inspect", action="store_true",
                       help="Show file structure before loading")
    parser.add_argument("--query", default="What are NVIDIA's recent initiatives?",
                       help="Query to process")
    parser.add_argument("--company", default="NVIDIA",
                       help="Company name for report")

    
    args = parser.parse_args()

    # Inspection mode
    if args.inspect:
        if args.load_all:
            print("Inspecting all standard files in data/ directory:")
            for filename in ["buffet_qna.xlsx", "brka_trades.xlsx", 
                           "news.pkl", "shareholder_letters.pkl"]:
                inspect_file(f"data/{filename}")
        else:
            inspect_file(args.data_path)
        return

    
    # Initialize and run pipeline
    rag_pipeline = RAGPipeline(llm_type="ollama")
    response = rag_pipeline.process_query(args.query, args.company)
    
    # Output results
    print("Generated Response: ", response)

if __name__ == "__main__":
    main()
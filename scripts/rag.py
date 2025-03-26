import argparse
import os
from data_utils import load_documents, load_all_documents, inspect_file
from rag.rag_pipeline import RAGPipeline
from rag.retrieval import initialize_faiss_databases

def main():
    print(f"Script running from: {os.getcwd()}")

    parser = argparse.ArgumentParser(description="BuffettAI RAG Pipeline")
    
    # Mutually exclusive group for loading options
    load_group = parser.add_mutually_exclusive_group(required=True)
    load_group.add_argument("--data_path", 
                          help="Path to individual data file (XLSX or PKL)")
    load_group.add_argument("--load_all", action="store_true",
                          help="Load all standard files from data/ directory")
    
    # Other arguments
    parser.add_argument("--inspect", action="store_true",
                       help="Show file structure before loading")
    parser.add_argument("--query", default="What are NVIDIA's recent initiatives?",
                       help="Query to process")
    parser.add_argument("--company", default="NVIDIA",
                       help="Company name for report")
    parser.add_argument("--method", default="ensemble", 
                       choices=["bm25", "faiss", "ensemble"],
                       help="Retrieval method")
    
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

    # Document loading
    if args.load_all:
        print("Loading all standard documents from data/ directory...")
        raw_docs = load_all_documents(base_path="../data")
        vectorstores = initialize_faiss_databases(raw_docs)
    else:
        doc_name = os.path.basename(args.data_path)
        raw_docs = load_documents(args.data_path)
        vectorstores = initialize_faiss_databases({doc_name: raw_docs})
    
    # Initialize and run pipeline
    pipeline = RAGPipeline(raw_docs)
    report, evaluation = pipeline.process_query(
        args.query, args.company, method=args.method
    )
    
    # Output results
    print("\nGenerated Report:")
    print(report)
    print("\nEvaluation:")
    print(evaluation)

if __name__ == "__main__":
    main()
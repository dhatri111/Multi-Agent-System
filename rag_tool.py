# rag_tool.py
"""
Fixed RAG Tool: Properly configured for ChromaDB.
IMPORTANT: Collection names must be alphanumeric with underscores/hyphens only.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from crewai.tools import tool
from dotenv import load_dotenv
from crewai import LLM


# config.py
"""
Configuration module: loads environment variables and initializes the LLM.
"""

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables")

# Set environment variables for CrewAI
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
os.environ["MISTRAL_MODEL_NAME"] = MISTRAL_MODEL_NAME

# Initialize LLM (Mistral) - shared across all agents
mistral_llm = LLM(
    api_key=MISTRAL_API_KEY,
    model=MISTRAL_MODEL_NAME
)
class RAGSystem:
    """Enhanced RAG system with proper ChromaDB configuration."""
    
    def __init__(self):
        self.vector_store = None # Vector store instance
        self.embeddings = None # Embeddings model
        self.db_initialized = False     # Flag to track initialization status
        self.initialize_rag() # Initialize RAG on creation
    
    # initialize RAG system
    def initialize_rag(self):
        """Initialize the RAG system with PDF documents."""
        print(" Initializing RAG System...")
        # Path to discrete math PDF
        pdf_path = "data/DiscreteMath.pdf"
        
        if not os.path.exists(pdf_path):
            print(f" ERROR: PDF not found at {pdf_path}")
            print("Please ensure the discrete_math.pdf file is in the data/ folder")
            self.db_initialized = False
            return
        
        try:
            # Load PDF
            print(f" Loading PDF from: {pdf_path}")
            loader = PyPDFLoader(pdf_path) # Load PDF using PyPDFLoader
            documents = loader.load() # Load all pages as documents
            print(f" Loaded {len(documents)} pages from PDF")
            
            if len(documents) == 0:
                print(" ERROR: PDF has no pages")
                self.db_initialized = False
                return
            
            # Split documents into chunks
            print("Splitting documents into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks from PDF")
            
            if len(chunks) == 0:
                print("ERROR: No chunks created")
                self.db_initialized = False
                return
            
            # Initialize embeddings
            print("Initializing embeddings model...")
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            print("Embeddings model loaded")
            
            # Create vector store
            print("Creating vector store in ChromaDB...")
            
            # CRITICAL: Collection name must be alphanumeric with underscores/hyphens only
            # NO slashes, NO special characters except underscore and hyphen
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name="discrete_math_kb",  # Valid collection name
                persist_directory="./chroma_db"
            )
            
            self.db_initialized = True
            print("=" * 60)
            print("RAG SYSTEM INITIALIZED SUCCESSFULLY")
            print(f"Vector Store: {len(chunks)} chunks indexed")
            print(f"Collection Name: discrete_math_kb")
            print(f"Persist Directory: ./chroma_db")
            print("=" * 60)
            
        except Exception as e:
            print(f"ERROR initializing RAG: {str(e)}")
            import traceback
            traceback.print_exc()
            self.db_initialized = False
    
    def retrieve_context(self, query: str, k: int = 4):
        """
        Retrieve relevant chunks from vector store.
        """
        print("\n" + "=" * 60)
        print(f"SEARCHING VECTOR STORE")
        print(f"Query: {query}")
        print("=" * 60)
        
        if not self.db_initialized or not self.vector_store:
            error_msg = "RAG system not initialized. Cannot retrieve from vector store."
            print(error_msg)
            return {
                "context": "",
                "sources": [],
                "rag_used": False,
                "message": error_msg,
                "chunks_found": 0
            }
        
        try:
            # Perform similarity search
            print(f"Searching for top {k} similar chunks...")
            docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k
            )
            
            print(f"Retrieved {len(docs_with_scores)} chunks from vector store")
            
            if not docs_with_scores or len(docs_with_scores) == 0:
                no_results_msg = "Vector store returned 0 results."
                print(no_results_msg)
                return {
                    "context": "",
                    "sources": [],
                    "rag_used": False,
                    "message": no_results_msg,
                    "chunks_found": 0
                }
            
            # Process retrieved chunks
            context_parts = []
            sources = []
            
            print(f"\nProcessing {len(docs_with_scores)} retrieved chunks:\n")
            
            for i, (doc, score) in enumerate(docs_with_scores):
                metadata = doc.metadata
                page = metadata.get('page', 'Unknown')
                source_name = metadata.get('source', 'discrete_math.pdf')
                
                print(f"  Chunk {i+1}: Page {page}, Relevance Score: {score:.3f}")
                print(f"  Preview: {doc.page_content[:100]}...\n")
                
                # Build context
                context_parts.append(
                    f"\n{'='*70}\n"
                    f"SOURCE {i+1} | Page {page} | Relevance: {score:.2f}\n"
                    f"{'='*70}\n"
                    f"{doc.page_content}\n"
                    f"{'='*70}\n"
                )
                
                # Store source metadata
                sources.append({
                    "number": i + 1,
                    "file_name": os.path.basename(source_name),
                    "page": page,
                    "relevance_score": round(score, 3),
                    "preview": doc.page_content[:150].replace('\n', ' ') + "..."
                })
            
            full_context = "\n".join(context_parts)
            
            success_msg = f"Successfully retrieved {len(docs_with_scores)} chunks from knowledge base"
            print(f"\n{success_msg}")
            print("=" * 60 + "\n")
            
            return {
                "context": full_context, # Full retrieved context
                "sources": sources, # List of source metadata
                "rag_used": True, # Indicate RAG retrieval was used
                "message": success_msg, # Status message
                "chunks_found": len(docs_with_scores)   # Number of chunks found
            }
            
        except Exception as e:
            error_msg = f"Error during vector store retrieval: {str(e)}"
            print(error_msg)
            import traceback # for detailed error logging
            traceback.print_exc() # print stack trace
            return {
                "context": "", # No context on error
                "sources": [], # No sources on error
                "rag_used": False, # Indicate RAG retrieval failed
                "message": error_msg,   # Error message
                "chunks_found": 0   # No chunks found
            }

# Global RAG instance
print("INITIALIZING GLOBAL RAG SYSTEM")
# Create a single RAG system instance for use in tools
rag_system = RAGSystem()

# RAG Tool for Discrete Math
@tool("query_discrete_math_rag")
def query_discrete_math_rag(query: str) -> str:
    """
    Search the discrete math vector store and return relevant context.
    This tool MUST be called before answering any discrete math question.
    
    Args:
        query: The discrete math question to search for
    
    Returns:
        String containing retrieved context and source information
    """
    result = rag_system.retrieve_context(query, k=4)
    
    # Format response as a clear string for the agent
    if result["rag_used"]:
        response = f"""
RAG RETRIEVAL SUCCESSFUL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Status: {result['message']}
Chunks Retrieved: {result['chunks_found']}

RETRIEVED CONTEXT FROM KNOWLEDGE BASE:
{result['context']}

SOURCES:
"""
        for src in result['sources']:
            response += f"\n[{src['number']}] {src['file_name']} - Page {src['page']} (Relevance: {src['relevance_score']})"
            response += f"\n    Preview: {src['preview']}\n"
        
        response += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: You MUST base your answer on the context above from the knowledge base.
Do NOT use general LLM knowledge. Cite the sources in your response.
"""
        return response
    else:
        return f"""
RAG RETRIEVAL FAILED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Status: {result['message']}
Chunks Retrieved: 0

Since no context was retrieved from the knowledge base, you may use 
general LLM knowledge to answer this question. However, clearly state that 
you are using LLM knowledge and not the knowledge base.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# RAG Tool for Calculus (No KB yet)
@tool("query_calculus_rag")
def query_calculus_rag(query: str) -> str:
    """
    Query calculus knowledge base (not yet implemented).
    
    Args:
        query: The calculus question
    
    Returns:
        Information about calculus KB availability
    """
    return """
CALCULUS KNOWLEDGE BASE STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Status: Calculus knowledge base is not yet implemented
Chunks Retrieved: 0

You may use general LLM knowledge to answer calculus questions.
Please state clearly that you are using LLM general knowledge.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
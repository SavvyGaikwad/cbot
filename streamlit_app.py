import streamlit as st
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import time

# Configure logging to be less verbose
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Set environment variables to fix protobuf and torch issues
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    # Import with proper error handling
    import torch
    # Force CPU usage to avoid device issues
    torch.set_default_tensor_type('torch.FloatTensor')
    
    from langchain_chroma import Chroma
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.error("Please install: pip install langchain-chroma sentence-transformers chromadb")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Document Search Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EmbeddingFunction:
    """Custom embedding function to handle Streamlit Cloud issues"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            try:
                # Force CPU and avoid device issues
                self._model = SentenceTransformer(
                    self.model_name,
                    device='cpu',
                    cache_folder=None
                )
                # Ensure model is on CPU
                self._model = self._model.cpu()
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                raise e
        return self._model
    
    def __call__(self, input_texts):
        """Encode texts to embeddings"""
        model = self._get_model()
        try:
            # Convert to list if single string
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            
            # Generate embeddings with CPU
            with torch.no_grad():
                embeddings = model.encode(
                    input_texts,
                    convert_to_tensor=False,  # Return numpy arrays
                    show_progress_bar=False,
                    device='cpu'
                )
            
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return dummy embeddings as fallback
            return [[0.0] * 384 for _ in input_texts]

class ChromaDBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = None
        self.embedding_function = None
        
    def initialize_embeddings(self):
        """Initialize embeddings with proper error handling"""
        if self.embedding_function is None:
            try:
                with st.spinner("Loading embeddings model (first time may take a while)..."):
                    self.embedding_function = EmbeddingFunction()
                    # Test the embedding function
                    test_embedding = self.embedding_function(["test"])
                    if test_embedding:
                        logger.info("Embeddings model loaded successfully")
                    else:
                        raise Exception("Test embedding failed")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                st.error(f"Embedding model error: {str(e)}")
                return None
        return self.embedding_function
    
    def load_database(self):
        """Load ChromaDB with proper error handling"""
        if self.db is None:
            try:
                if not os.path.exists(self.db_path):
                    st.error(f"Database path does not exist: {self.db_path}")
                    return None
                
                embedding_function = self.initialize_embeddings()
                if embedding_function is None:
                    return None
                
                with st.spinner("Loading vector database..."):
                    # Use direct ChromaDB client
                    client = chromadb.PersistentClient(
                        path=self.db_path,
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
                    
                    # Get the collection
                    collections = client.list_collections()
                    if not collections:
                        st.error("No collections found in database")
                        return None
                    
                    collection = collections[0]  # Use first collection
                    
                    # Create wrapper for queries
                    self.db = DatabaseWrapper(client, collection, embedding_function)
                
                logger.info("Vector database loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load vector database: {e}")
                st.error(f"Database loading error: {str(e)}")
                return None
                
        return self.db
    
    def query_database(self, query: str, document_type: str = None, 
                      document_title: str = None, k: int = 5) -> List[Dict[str, Any]]:
        """Query the database with optional filters"""
        db = self.load_database()
        if db is None:
            return []
        
        try:
            # Perform similarity search
            results = db.similarity_search(query, k=k*2)
            
            # Apply filters
            filtered_results = []
            for result in results:
                include = True
                
                if document_type and document_type != "All":
                    if result.get("document_type") != document_type.lower():
                        include = False
                
                if document_title and document_title != "All":
                    doc_title = result.get("document_title", "")
                    if document_title.lower() not in doc_title.lower():
                        include = False
                
                if include:
                    filtered_results.append(result)
                
                if len(filtered_results) >= k:
                    break
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            st.error(f"Query error: {str(e)}")
            return []

class DatabaseWrapper:
    """Wrapper to make ChromaDB work like LangChain Chroma"""
    
    def __init__(self, client, collection, embedding_function):
        self.client = client
        self.collection = collection
        self.embedding_function = embedding_function
    
    def similarity_search(self, query: str, k: int = 5):
        """Perform similarity search"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_function([query])[0]
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    result = {
                        "content": doc,
                        "rank": i + 1,
                        "distance": results['distances'][0][i] if results['distances'] else 0,
                        **metadata  # Unpack all metadata
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

@st.cache_resource
def get_db_manager(db_path: str):
    """Cached database manager initialization"""
    return ChromaDBManager(db_path)

def display_search_results(results: List[Dict[str, Any]]):
    """Display search results in a nice format"""
    if not results:
        st.warning("No results found for your query.")
        return
    
    st.success(f"Found {len(results)} relevant documents")
    
    for i, result in enumerate(results):
        doc_title = result.get('document_title', result.get('source', 'Unknown'))
        
        with st.expander(f"Result {result.get('rank', i+1)}: {doc_title}", expanded=i==0):
            
            # Metadata section
            col1, col2, col3 = st.columns(3)
            with col1:
                doc_type = result.get('document_type', 'unknown')
                st.badge(doc_type.upper(), type="secondary")
            with col2:
                section = result.get('section', '')
                if section:
                    st.write(f"**Section:** {section}")
            with col3:
                has_images = result.get('has_images', False)
                if has_images:
                    image_count = result.get('image_count', 0)
                    st.write(f"📷 {image_count} images")
            
            # Content section
            st.write("**Content:**")
            content = result.get('content', 'No content available')
            st.write(content)
            
            # Source information
            source = result.get('source', '')
            if source:
                st.caption(f"Source: {source}")
            
            # Distance/similarity score
            distance = result.get('distance')
            if distance is not None:
                similarity = max(0, 1 - distance)
                st.caption(f"Similarity: {similarity:.2%}")
            
            # Images section
            images_str = result.get('images', '')
            if images_str:
                images = images_str.split('|') if images_str else []
                if images:
                    with st.expander("📷 Associated Images"):
                        for img_path in images:
                            if img_path:
                                st.write(f"• {img_path}")

def main():
    st.title("🔍 Document Search Interface")
    st.markdown("Search through your processed documents using semantic similarity")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database path input
        db_path = st.text_input(
            "Vector Database Path", 
            value="comprehensive_vector_db",
            help="Path to your ChromaDB directory"
        )
        
        # Search filters
        st.subheader("🔧 Search Filters")
        
        document_type = st.selectbox(
            "Document Type",
            ["All", "user_guide", "glossary", "faq"],
            help="Filter by document type"
        )
        
        document_title = st.text_input(
            "Document Title (contains)",
            placeholder="Leave empty for all documents",
            help="Filter by document title"
        )
        
        max_results = st.slider(
            "Maximum Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of results to return"
        )
        
        # Database status
        st.subheader("📊 Database Status")
        
        if st.button("🔄 Refresh Database"):
            st.cache_resource.clear()
            st.rerun()
        
        # Test database connection
        if os.path.exists(db_path):
            st.success("✅ Database found")
            try:
                db_manager = get_db_manager(db_path)
                db = db_manager.load_database()
                if db:
                    st.success("✅ Database loaded")
                else:
                    st.error("❌ Database loading failed")
            except Exception as e:
                st.error(f"❌ Database error: {str(e)}")
        else:
            st.error("❌ Database path not found")
    
    # Main search interface
    st.header("🔍 Search Documents")
    
    # Search input
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., 'cycle count recount', 'saved view', 'column chooser'",
        key="search_query"
    )
    
    # Example queries
    with st.expander("💡 Example Queries"):
        example_queries = [
            "How to create a saved view?",
            "cycle count recount process",
            "column chooser functionality",
            "mobile app features",
            "API documentation",
            "user guide steps"
        ]
        
        st.write("Try these example queries:")
        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            col = cols[i % 2]
            with col:
                if st.button(f"🔸 {example}", key=f"example_{i}"):
                    st.session_state.search_query = example
                    st.rerun()
    
    # Search button and results
    if st.button("🔍 Search", type="primary", use_container_width=True) or query:
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching documents..."):
                try:
                    db_manager = get_db_manager(db_path)
                    
                    # Perform search
                    results = db_manager.query_database(
                        query=query,
                        document_type=document_type if document_type != "All" else None,
                        document_title=document_title if document_title else None,
                        k=max_results
                    )
                    
                    # Display results
                    if results:
                        st.header("📋 Search Results")
                        display_search_results(results)
                    else:
                        st.warning("No results found. Try adjusting your query or filters.")
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.error("Please check your database path and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Use specific keywords from your documents for better results. "
        "The search uses semantic similarity, so related terms will also match."
    )

if __name__ == "__main__":
    main()
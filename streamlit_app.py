import streamlit as st
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Document Search Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ChromaDBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = None
        self.embeddings = None
        
    def initialize_embeddings(self):
        """Initialize embeddings model with caching"""
        if self.embeddings is None:
            try:
                with st.spinner("Loading embeddings model..."):
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                logger.info("Embeddings model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                raise e
        return self.embeddings
    
    def load_database(self):
        """Load ChromaDB with proper error handling"""
        if self.db is None:
            try:
                if not os.path.exists(self.db_path):
                    st.error(f"Database path does not exist: {self.db_path}")
                    return None
                
                embeddings = self.initialize_embeddings()
                
                with st.spinner("Loading vector database..."):
                    # Try to load existing database
                    self.db = Chroma(
                        persist_directory=self.db_path,
                        embedding_function=embeddings
                    )
                
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
            results = db.similarity_search(query, k=k*2)  # Get extra to filter
            
            # Apply filters
            filtered_results = []
            for result in results:
                include = True
                
                if document_type and document_type != "All":
                    if result.metadata.get("document_type") != document_type.lower():
                        include = False
                
                if document_title and document_title != "All":
                    if document_title.lower() not in result.metadata.get("document_title", "").lower():
                        include = False
                
                if include:
                    filtered_results.append(result)
                
                if len(filtered_results) >= k:
                    break
            
            # Format results
            response_data = []
            for i, result in enumerate(filtered_results):
                images_str = result.metadata.get("images", "")
                images_list = images_str.split('|') if images_str else []
                
                item = {
                    "rank": i + 1,
                    "content": result.page_content,
                    "document_type": result.metadata.get("document_type", "unknown"),
                    "source": result.metadata.get("source", "unknown"),
                    "document_title": result.metadata.get("document_title", ""),
                    "section": result.metadata.get("section", ""),
                    "type": result.metadata.get("type", ""),
                    "images": images_list,
                    "has_images": result.metadata.get("has_images", False),
                    "metadata": result.metadata
                }
                response_data.append(item)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            st.error(f"Query error: {str(e)}")
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
        with st.expander(f"Result {result['rank']}: {result['document_title']}", expanded=i==0):
            
            # Metadata section
            col1, col2, col3 = st.columns(3)
            with col1:
                st.badge(result['document_type'].upper(), type="secondary")
            with col2:
                if result['section']:
                    st.write(f"**Section:** {result['section']}")
            with col3:
                if result['has_images']:
                    st.write(f"📷 {len(result['images'])} images")
            
            # Content section
            st.write("**Content:**")
            st.write(result['content'])
            
            # Source information
            if result['source']:
                st.caption(f"Source: {result['source']}")
            
            # Images section
            if result['images']:
                with st.expander("📷 Associated Images"):
                    for img_path in result['images']:
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
        db_manager = get_db_manager(db_path)
        
        if st.button("🔄 Refresh Database"):
            st.cache_resource.clear()
            st.rerun()
        
        # Test database connection
        if os.path.exists(db_path):
            st.success("✅ Database found")
            try:
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
        "Enter your search query:",
        placeholder="e.g., 'cycle count recount', 'saved view', 'column chooser'",
        label_visibility="collapsed"
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
        for example in example_queries:
            if st.button(f"🔸 {example}", key=f"example_{example}"):
                st.session_state.query = example
                st.rerun()
    
    # Use session state for query if set by example
    if 'query' in st.session_state:
        query = st.session_state.query
        del st.session_state.query
    
    # Search button and results
    if st.button("🔍 Search", type="primary", use_container_width=True) or query:
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching documents..."):
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
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Use specific keywords from your documents for better results. "
        "The search uses semantic similarity, so related terms will also match."
    )

if __name__ == "__main__":
    main()
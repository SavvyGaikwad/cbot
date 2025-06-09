import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import webbrowser
import os
import json
import glob
from typing import List, Dict, Any

# Configure Gemini
genai.configure(api_key="AIzaSyAvzloY_NyX-yjtZb8EE_RdXPs3rPmMEso")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class JSONDocumentProcessor:
    """Process JSON documents and extract content with metadata"""
    
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.documents = []
        
    def extract_images_from_content(self, content_items: List[Dict]) -> List[str]:
        """Extract image URLs from content items"""
        images = []
        for item in content_items:
            if item.get('type') == 'media' and item.get('path'):
                images.append(item['path'])
            elif item.get('type') == 'section' and 'content' in item:
                # Recursively extract images from nested sections
                nested_images = self.extract_images_from_content(item['content'])
                images.extend(nested_images)
        return images
    
    def process_content_items(self, content_items: List[Dict], parent_section: str = "") -> List[Dict]:
        """Process content items and create structured text chunks"""
        chunks = []
        current_chunk = []
        current_images = []
        
        for item in content_items:
            item_type = item.get('type', '')
            
            if item_type == 'section':
                # If we have accumulated content, save it as a chunk
                if current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'images': current_images.copy(),
                        'section': parent_section
                    })
                    current_chunk = []
                    current_images = []
                
                # Process the section
                section_title = item.get('title', '')
                section_content = item.get('content', [])
                
                # Create section header chunk
                section_text = f"Section: {section_title}"
                section_images = self.extract_images_from_content(section_content)
                
                # Recursively process section content
                nested_chunks = self.process_content_items(section_content, section_title)
                
                # Add section header
                chunks.append({
                    'text': section_text,
                    'images': section_images,
                    'section': section_title
                })
                
                # Add nested chunks
                chunks.extend(nested_chunks)
                
            elif item_type == 'media':
                # Add media reference to current chunk
                media_path = item.get('path', '')
                if media_path:
                    current_images.append(media_path)
                    current_chunk.append(f"[Image: {media_path.split('/')[-1]}]")
                    
            elif item_type in ['info', 'step', 'substep', 'text']:
                # Add text content to current chunk
                text = item.get('text', '')
                if text:
                    current_chunk.append(text)
                    
            elif item_type == 'list':
                # Handle list items
                list_items = item.get('items', [])
                for list_item in list_items:
                    current_chunk.append(f"• {list_item}")
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'images': current_images.copy(),
                'section': parent_section
            })
        
        return chunks
    
    def process_json_file(self, file_path: str) -> List[Document]:
        """Process a single JSON file and return Document objects"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = os.path.basename(file_path)
            document_title = data.get('document_title', filename)
            content = data.get('content', [])
            
            print(f"📄 Processing: {document_title}")
            
            # Extract all content chunks
            chunks = self.process_content_items(content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                # Create metadata
                metadata = {
                    'source': document_title,
                    'file_path': file_path,
                    'section': chunk['section'],
                    'chunk_id': i,
                    'document_type': 'instruction_manual',
                    'has_images': len(chunk['images']) > 0,
                    'images': '|'.join(chunk['images']) if chunk['images'] else '',
                    'image_count': len(chunk['images'])
                }
                
                # Create Document object
                doc = Document(
                    page_content=chunk['text'],
                    metadata=metadata
                )
                documents.append(doc)
            
            print(f"   ✅ Created {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {str(e)}")
            return []
    
    def process_all_files(self) -> List[Document]:
        """Process all JSON files in the data directory"""
        json_files = glob.glob(os.path.join(self.data_directory, "*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {self.data_directory}")
            return []
        
        print(f"🔍 Found {len(json_files)} JSON files to process")
        print("="*60)
        
        all_documents = []
        
        for file_path in json_files:
            documents = self.process_json_file(file_path)
            all_documents.extend(documents)
        
        print("="*60)
        print(f"📊 Processing Summary:")
        print(f"   • Files processed: {len(json_files)}")
        print(f"   • Total document chunks: {len(all_documents)}")
        print(f"   • Documents with images: {len([d for d in all_documents if d.metadata.get('has_images', False)])}")
        
        # Show breakdown by source
        source_counts = {}
        for doc in all_documents:
            source = doc.metadata.get('source', 'unknown')
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
        
        print(f"   • Breakdown by source:")
        for source, count in source_counts.items():
            print(f"     - {source}: {count} chunks")
        
        return all_documents

def create_vector_database(data_directory: str, persist_directory: str = "vector_db_unified"):
    """Create or update the vector database from JSON files"""
    
    # Initialize processor
    processor = JSONDocumentProcessor(data_directory)
    
    # Process all files
    documents = processor.process_all_files()
    
    if not documents:
        print("❌ No documents to process. Exiting.")
        return None
    
    # Split documents if they're too long
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    print(f"\n📝 Splitting long documents...")
    split_documents = text_splitter.split_documents(documents)
    
    print(f"   • Original chunks: {len(documents)}")
    print(f"   • After splitting: {len(split_documents)}")
    
    # Create vector database
    print(f"\n🔄 Creating vector database...")
    print(f"   • Persist directory: {persist_directory}")
    
    # Remove existing database if it exists
    if os.path.exists(persist_directory):
        print(f"   • Removing existing database...")
        import shutil
        shutil.rmtree(persist_directory)
    
    # Create new database
    vector_db = Chroma.from_documents(
        documents=split_documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print(f"   ✅ Vector database created successfully!")
    print(f"   • Total embeddings: {len(split_documents)}")
    
    return vector_db

def convert_github_url_to_raw(github_url):
    """Convert GitHub blob URL to raw URL for direct image access"""
    if "github.com" in github_url and "/blob/" in github_url:
        return github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return github_url

def display_image_from_url(image_url, max_width=600):
    """Display image from URL using matplotlib"""
    try:
        # Convert GitHub URL to raw URL
        raw_url = convert_github_url_to_raw(image_url)
        
        # Download and display image
        response = requests.get(raw_url, timeout=10)
        response.raise_for_status()
        
        # Load image
        img = Image.open(BytesIO(response.content))
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Reference Image")
        plt.tight_layout()
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Could not display image from {image_url}")
        print(f"   Error: {str(e)}")
        return False

def display_images_html(image_urls, max_width=400):
    """Create HTML file to display images in browser"""
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Related Images</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 15px; }}
                .image-card {{ 
                    border: 1px solid #ccc; 
                    padding: 10px; 
                    border-radius: 8px; 
                    max-width: {max_width}px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .image-card img {{ 
                    max-width: 100%; 
                    height: auto; 
                    border-radius: 4px;
                }}
                .image-title {{ 
                    margin-top: 8px; 
                    font-size: 12px; 
                    color: #666;
                    word-break: break-all;
                }}
                .error {{ color: red; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h2>Related Images</h2>
            <div class="container">
        """
        
        for i, url in enumerate(image_urls):
            raw_url = convert_github_url_to_raw(url)
            filename = url.split('/')[-1]
            html_content += f"""
            <div class="image-card">
                <img src='{raw_url}' alt='{filename}' 
                     onerror="this.style.display='none'; this.nextSibling.style.display='block';">
                <div class="error" style="display: none;">
                    Failed to load: {filename}
                </div>
                <div class="image-title">{filename}</div>
            </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML file and open in browser
        html_file = "temp_images.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Open in default browser
        webbrowser.open(f'file://{os.path.abspath(html_file)}')
        print(f"📸 Opening {len(image_urls)} images in your browser...")
        
        return True
            
    except Exception as e:
        print(f"Error creating HTML display: {e}")
        return False

def collect_images_from_docs_comprehensive(docs, k_limit=10):
    """Enhanced image collection that processes ALL retrieved documents"""
    all_images = []
    seen_images = set()
    doc_info = []
    
    print(f"\n[Image Collection] Processing {len(docs)} retrieved documents...")
    print("-" * 60)
    
    for doc_idx, doc in enumerate(docs):
        doc_images = []
        doc_source = "unknown"
        doc_section = "unknown"
        doc_type = "unknown"
        
        # Extract document information
        if hasattr(doc, 'metadata') and doc.metadata:
            doc_source = doc.metadata.get('source', 'unknown')
            doc_section = doc.metadata.get('section', 'unknown')
            doc_type = doc.metadata.get('document_type', 'unknown')
            has_images = doc.metadata.get('has_images', False)
            images_str = doc.metadata.get('images', '')
            
            # Process images if they exist
            if images_str and images_str.strip():
                # Handle pipe-separated format
                images = [img.strip() for img in images_str.split('|') if img.strip()]
                
                # Add unique images
                for img in images:
                    if img and img not in seen_images:
                        all_images.append(img)
                        seen_images.add(img)
                        doc_images.append(img)
            
            # Log document processing results
            status = f"✅ {len(doc_images)} images" if doc_images else ("⚠️  has_images=True but no images found" if has_images else "ℹ️  no images")
            print(f"Doc {doc_idx+1:2d}: [{doc_type:10s}] {doc_source:25s} | {doc_section:20s} | {status}")
            
            if doc_images:
                for img in doc_images:
                    filename = img.split('/')[-1] if '/' in img else img
                    print(f"      └─ {filename}")
        
        else:
            print(f"Doc {doc_idx+1:2d}: [no metadata] - Cannot extract images")
    
    # Summary
    print("-" * 60)
    print(f"📊 COLLECTION SUMMARY:")
    print(f"   • Documents processed: {len(docs)}")
    print(f"   • Documents with images: {len([d for d in docs if d.metadata.get('has_images', False)])}")
    print(f"   • Total unique images: {len(all_images)}")
    
    # Limit images if requested
    if k_limit and len(all_images) > k_limit:
        print(f"   • Limiting to first {k_limit} images")
        all_images = all_images[:k_limit]
    
    print()
    return all_images

def display_images_console(image_urls):
    """Display image information in console with clickable links"""
    if not image_urls:
        print("\n📸 No related images found.")
        return
    
    print(f"\n📸 Related Images ({len(image_urls)}):")
    print("-" * 50)
    
    for i, url in enumerate(image_urls, 1):
        # Convert to raw URL for direct viewing
        raw_url = convert_github_url_to_raw(url)
        filename = url.split('/')[-1]
        
        print(f"{i:2d}. {filename}")
        print(f"    View: {raw_url}")
        print(f"    GitHub: {url}")
        print()

def ask_question(vector_db, query, show_images=True, display_method='console', k_docs=5, k_images=10):
    """Enhanced question answering with comprehensive image collection"""
    try:
        print(f"\n🔍 Searching for: '{query}'")
        print(f"📄 Retrieving {k_docs} most relevant documents...")
        
        # Get relevant documents
        docs = vector_db.similarity_search(query, k=k_docs)
        
        # Combine context more effectively
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
            section = doc.metadata.get('section', '') if hasattr(doc, 'metadata') else ''
            context_parts.append(f"[Source {i+1}: {source} - {section}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Collect images from ALL documents with comprehensive logging
        image_urls = []
        if show_images:
            print(f"\n🖼️  Collecting images from retrieved documents...")
            image_urls = collect_images_from_docs_comprehensive(docs, k_limit=k_images)
        
        # Generate response
        prompt = f"""
You are a helpful assistant answering questions about user profile settings and software documentation.

Use the provided context to answer the question accurately and completely. If the context contains relevant information, provide a detailed answer. If you're not sure or the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {query}

Please provide a clear, helpful answer based on the context above:
"""
        
        print(f"\n🧠 Generating response...")
        response = model.generate_content(prompt)
        print("\n" + "="*60)
        print("🧠 CHATBOT ANSWER:")
        print("="*60)
        print(response.text)
        
        # Display images if available
        if image_urls and show_images:
            print(f"\n" + "="*60)
            print("📸 RELATED IMAGES:")
            print("="*60)
            
            if display_method == 'console':
                display_images_console(image_urls)
            
            elif display_method == 'matplotlib':
                print(f"\n📸 Displaying {len(image_urls)} related images using matplotlib...")
                for i, url in enumerate(image_urls):
                    print(f"\nImage {i+1}/{len(image_urls)}:")
                    if not display_image_from_url(url):
                        print(f"   Skipping {url.split('/')[-1]}")
            
            elif display_method == 'html':
                print(f"\n📸 Creating HTML page with {len(image_urls)} related images...")
                if not display_images_html(image_urls):
                    # Fallback to console display
                    print("HTML display failed, showing in console:")
                    display_images_console(image_urls)
        
        elif show_images and not image_urls:
            print(f"\n📸 No images found for this query.")
        
        # Final summary
        print(f"\n" + "="*60)
        print("📊 SEARCH SUMMARY:")
        print("="*60)
        print(f"   • Documents retrieved: {len(docs)}")
        print(f"   • Images found: {len(image_urls)}")
        if image_urls:
            print(f"   • Display method: {display_method}")
        
        # Show document sources
        sources = set()
        for doc in docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', 'unknown')
                sources.add(source)
        
        if sources:
            print(f"   • Sources consulted: {', '.join(sorted(sources))}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check your API key and internet connection.")

def main():
    """Main function to set up and run the chatbot"""
    
    # Configuration
    DATA_DIRECTORY = r"C:\Users\hp\Downloads\final\data"
    PERSIST_DIRECTORY = "vector_db_unified"
    
    print("🚀 Multi-File JSON Document Chatbot")
    print("="*70)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIRECTORY):
        print(f"❌ Data directory not found: {DATA_DIRECTORY}")
        print("Please check the path and try again.")
        return
    
    # Create or load vector database
    print(f"📁 Data directory: {DATA_DIRECTORY}")
    
    # Ask user if they want to rebuild the database
    rebuild = input("\nRebuild vector database? (y/n): ").lower() == 'y'
    
    if rebuild or not os.path.exists(PERSIST_DIRECTORY):
        print("\n🔄 Creating new vector database...")
        vector_db = create_vector_database(DATA_DIRECTORY, PERSIST_DIRECTORY)
        if not vector_db:
            return
    else:
        print(f"\n📚 Loading existing vector database from {PERSIST_DIRECTORY}...")
        try:
            vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
            print("   ✅ Vector database loaded successfully!")
        except Exception as e:
            print(f"   ❌ Error loading database: {e}")
            print("   Creating new database...")
            vector_db = create_vector_database(DATA_DIRECTORY, PERSIST_DIRECTORY)
            if not vector_db:
                return
    
    # Interactive chatbot
    print("\n" + "="*70)
    print("🤖 CHATBOT READY!")
    print("="*70)
    print("Available commands:")
    print("  • Type your question normally")
    print("  • 'quit' - Exit the program")
    print("  • 'rebuild' - Rebuild the vector database")
    print("  • 'stats' - Show database statistics")
    print("  • 'help' - Show this help message")
    print("="*70)
    
    # Default settings
    show_images = True
    display_method = 'console'
    k_docs = 5
    k_images = 10
    
    while True:
        question = input("\n💬 Ask a question: ")
        
        if question.lower() == 'quit':
            print("👋 Goodbye!")
            break
            
        elif question.lower() == 'help':
            print("\n📚 Available commands:")
            print("  • Regular questions - Just type naturally")
            print("  • rebuild - Recreate the vector database")
            print("  • stats - Show database statistics")
            print("  • quit - Exit the program")
            
        elif question.lower() == 'rebuild':
            print("\n🔄 Rebuilding vector database...")
            vector_db = create_vector_database(DATA_DIRECTORY, PERSIST_DIRECTORY)
            if vector_db:
                print("✅ Database rebuilt successfully!")
            else:
                print("❌ Failed to rebuild database")
                
        elif question.lower() == 'stats':
            try:
                # Get some statistics about the database
                test_docs = vector_db.similarity_search("test", k=100)  # Get many docs to see variety
                
                print(f"\n📊 Database Statistics:")
                print(f"   • Total documents available: {len(test_docs)}")
                
                # Count by source
                sources = {}
                images_count = 0
                for doc in test_docs:
                    if hasattr(doc, 'metadata') and doc.metadata:
                        source = doc.metadata.get('source', 'unknown')
                        sources[source] = sources.get(source, 0) + 1
                        if doc.metadata.get('has_images', False):
                            images_count += 1
                
                print(f"   • Documents with images: {images_count}")
                print(f"   • Sources in database:")
                for source, count in sources.items():
                    print(f"     - {source}: {count} chunks")
                    
            except Exception as e:
                print(f"❌ Error getting statistics: {e}")
            
        else:
            if question.strip():
                ask_question(vector_db, question, show_images, display_method, k_docs, k_images)
            else:
                print("❌ Please enter a question or command")

if __name__ == "__main__":
    main()
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def check_sqlite_version():
    """Check if SQLite version meets ChromaDB requirements"""
    sqlite_version = sqlite3.sqlite_version
    required_version = "3.35.0"
    
    print(f"Current SQLite version: {sqlite_version}")
    print(f"Required SQLite version: {required_version} or higher")
    
    # Parse version strings for comparison
    current_parts = [int(x) for x in sqlite_version.split('.')]
    required_parts = [int(x) for x in required_version.split('.')]
    
    # Pad shorter version with zeros
    max_len = max(len(current_parts), len(required_parts))
    current_parts.extend([0] * (max_len - len(current_parts)))
    required_parts.extend([0] * (max_len - len(required_parts)))
    
    if current_parts < required_parts:
        print(f"❌ SQLite version {sqlite_version} is below required {required_version}")
        print("\nTo fix this issue:")
        print("1. Update Python to latest version (3.9+ recommended)")
        print("2. Or install pysqlite3-binary: pip install pysqlite3-binary")
        print("3. For conda users: conda install sqlite")
        return False
    else:
        print(f"✅ SQLite version {sqlite_version} meets requirements")
        return True

def load_json(filepath):
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_glossary(data, source_file):
    """Process glossary JSON structure"""
    docs = []
    
    for section in data.get("content", []):
        section_title = section.get("title", "")
        
        for item in section.get("content", []):
            if item.get("type") == "term":
                title = item.get("title", "")
                definition = item.get("definition", "")
                
                # Create full content
                content = f"Term: {title}\nDefinition: {definition}"
                
                # Extract term name without quotes/abbreviations for better searchability
                clean_title = title.split('"')[0].strip() if '"' in title else title
                
                metadata = {
                    "source": source_file,
                    "document_type": "glossary",
                    "section": section_title,
                    "term": clean_title,
                    "full_term": title,
                    "type": "definition"
                }
                
                docs.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
                
                # Also create a searchable version with just the definition
                docs.append(Document(
                    page_content=f"{clean_title}: {definition}",
                    metadata={**metadata, "type": "term_definition"}
                ))
    
    return docs

def process_faqs(data, source_file):
    """Process FAQ JSON structure"""
    docs = []
    
    for section in data.get("content", []):
        section_title = section.get("title", "")
        current_question = ""
        current_answers = []
        current_links = []
        
        for item in section.get("content", []):
            if item.get("type") == "question":
                # Process previous Q&A if exists
                if current_question and current_answers:
                    content = f"Question: {current_question}\nAnswer: {' '.join(current_answers)}"
                    if current_links:
                        content += f"\nReferences: {', '.join(current_links)}"
                    
                    metadata = {
                        "source": source_file,
                        "document_type": "faq",
                        "section": section_title,
                        "question": current_question,
                        "type": "qa_pair",
                        "has_links": len(current_links) > 0
                    }
                    
                    docs.append(Document(page_content=content, metadata=metadata))
                
                # Start new Q&A
                current_question = item.get("text", "")
                current_answers = []
                current_links = []
                
            elif item.get("type") == "answer":
                answer_text = item.get("text", "")
                current_answers.append(answer_text)
                
                # Extract links if present
                links = item.get("links", [])
                current_links.extend(links)
        
        # Process final Q&A in section
        if current_question and current_answers:
            content = f"Question: {current_question}\nAnswer: {' '.join(current_answers)}"
            if current_links:
                content += f"\nReferences: {', '.join(current_links)}"
            
            metadata = {
                "source": source_file,
                "document_type": "faq",
                "section": section_title,
                "question": current_question,
                "type": "qa_pair",
                "has_links": len(current_links) > 0
            }
            
            docs.append(Document(page_content=content, metadata=metadata))
    
    return docs

def process_general_content(data, source_file):
    """Process general content structure (handles steps, substeps, info, media, etc.)"""
    docs = []
    document_title = data.get("document_title", source_file)
    
    for section in data.get("content", []):
        if section.get("type") == "section":
            section_title = section.get("title", "")
            section_content = []
            section_steps = []
            section_images = []
            current_step = None
            current_substeps = []
            
            for item in section.get("content", []):
                item_type = item.get("type", "")
                text = item.get("text", "")
                
                if item_type == "info":
                    section_content.append(f"Info: {text}")
                
                elif item_type == "step":
                    # Save previous step if exists
                    if current_step and current_substeps:
                        full_step = f"{current_step}\nSubsteps: {' | '.join(current_substeps)}"
                        section_steps.append(full_step)
                    elif current_step:
                        section_steps.append(current_step)
                    
                    # Start new step
                    current_step = f"Step: {text}"
                    current_substeps = []
                
                elif item_type == "substep":
                    if text:
                        current_substeps.append(text)
                
                elif item_type == "media":
                    image_path = item.get("path", "")
                    if image_path:
                        section_images.append(image_path)
                
                # Handle any other text content
                elif text and item_type not in ["step", "substep", "info", "media"]:
                    section_content.append(f"{item_type}: {text}")
            
            # Don't forget the last step
            if current_step and current_substeps:
                full_step = f"{current_step}\nSubsteps: {' | '.join(current_substeps)}"
                section_steps.append(full_step)
            elif current_step:
                section_steps.append(current_step)
            
            # Combine all content for this section
            all_content = section_content + section_steps
            
            if all_content or section_images:
                # Create comprehensive section document
                full_content = f"Document: {document_title}\nSection: {section_title}\n\n"
                if all_content:
                    full_content += "\n".join(all_content)
                
                images_str = "|".join(section_images) if section_images else ""
                
                metadata = {
                    "source": source_file,
                    "document_type": "user_guide",
                    "document_title": document_title,
                    "section": section_title,
                    "type": "full_section",
                    "images": images_str,
                    "has_images": len(section_images) > 0,
                    "content_count": len(all_content),
                    "image_count": len(section_images)
                }
                
                docs.append(Document(page_content=full_content, metadata=metadata))
                
                # Create individual documents for searchability
                for i, content_item in enumerate(all_content):
                    item_metadata = {
                        "source": source_file,
                        "document_type": "user_guide",
                        "document_title": document_title,
                        "section": section_title,
                        "item": i,
                        "type": "content_item",
                        "images": images_str,
                        "has_images": len(section_images) > 0
                    }
                    
                    docs.append(Document(
                        page_content=f"{document_title} - {section_title}: {content_item}",
                        metadata=item_metadata
                    ))
    
    return docs

def process_single_file(filepath):
    """Process a single JSON file and return documents"""
    print(f"\nProcessing: {filepath}")
    
    data = load_json(filepath)
    if not data:
        return []
    
    source_file = os.path.basename(filepath)
    document_title = data.get("document_title", source_file)
    
    print(f"Document title: {document_title}")
    
    # Determine document type and process accordingly
    if "glossary" in document_title.lower() or source_file.lower().startswith("glossary"):
        docs = process_glossary(data, source_file)
        print(f"Processed as Glossary - Created {len(docs)} documents")
    
    elif "faq" in document_title.lower() or source_file.lower().startswith("faq"):
        docs = process_faqs(data, source_file)
        print(f"Processed as FAQ - Created {len(docs)} documents")
    
    else:
        # Default to general content processing (handles steps, substeps, etc.)
        docs = process_general_content(data, source_file)
        print(f"Processed as General Content - Created {len(docs)} documents")
    
    return docs

def find_all_json_files(root_folder):
    """Recursively find all JSON files in the folder and subfolders"""
    json_files = []
    
    # Use pathlib for better path handling
    root_path = Path(root_folder)
    
    if not root_path.exists():
        print(f"Folder not found: {root_folder}")
        return []
    
    # Find all JSON files recursively
    for json_file in root_path.rglob("*.json"):
        json_files.append(str(json_file))
    
    return sorted(json_files)

def create_comprehensive_vector_db(data_folder, persist_dir):
    """Create vector database from all JSON files in data folder and subfolders"""
    
    # Check SQLite version first
    if not check_sqlite_version():
        print("\n⚠️  Warning: SQLite version may cause issues with ChromaDB")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("Exiting...")
            return None
    
    # Remove existing database
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"Removed existing database at {persist_dir}")
    
    # Find all JSON files recursively
    json_files = find_all_json_files(data_folder)
    
    if not json_files:
        print(f"No JSON files found in {data_folder}")
        return None
    
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        relative_path = os.path.relpath(file, data_folder)
        print(f"  - {relative_path}")
    
    all_docs = []
    file_stats = {}
    
    # Process each file
    for filepath in json_files:
        docs = process_single_file(filepath)
        all_docs.extend(docs)
        
        # Track stats per file
        relative_path = os.path.relpath(filepath, data_folder)
        file_stats[relative_path] = len(docs)
    
    print(f"\nTotal documents created: {len(all_docs)}")
    
    if not all_docs:
        print("No documents to process!")
        return None
    
    # Show file processing stats
    print("\nDocuments per file:")
    for file_path, count in file_stats.items():
        print(f"  {file_path}: {count} documents")
    
    # Show document type distribution
    doc_types = {}
    doc_titles = {}
    for doc in all_docs:
        doc_type = doc.metadata.get("document_type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Track document titles
        doc_title = doc.metadata.get("document_title", doc.metadata.get("source", "unknown"))
        doc_titles[doc_title] = doc_titles.get(doc_title, 0) + 1
    
    print("\nDocument type distribution:")
    for doc_type, count in doc_types.items():
        print(f"  {doc_type}: {count} documents")
    
    print("\nDocument title distribution:")
    for doc_title, count in sorted(doc_titles.items()):
        print(f"  {doc_title}: {count} documents")
    
    # Create vector database with error handling
    print(f"\nCreating vector database...")
    try:
        # Initialize embeddings
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Ensure CPU usage for compatibility
            encode_kwargs={'normalize_embeddings': True}  # Better similarity scores
        )
        
        # Create ChromaDB with batch processing for large datasets
        print("Creating ChromaDB...")
        batch_size = 100  # Process in batches to avoid memory issues
        
        if len(all_docs) <= batch_size:
            db = Chroma.from_documents(
                all_docs, 
                embeddings, 
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"}  # Better for text similarity
            )
        else:
            # Process in batches
            print(f"Processing {len(all_docs)} documents in batches of {batch_size}...")
            
            # Create initial batch
            first_batch = all_docs[:batch_size]
            db = Chroma.from_documents(
                first_batch, 
                embeddings, 
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            # Add remaining documents in batches
            for i in range(batch_size, len(all_docs), batch_size):
                batch = all_docs[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(all_docs)-1)//batch_size + 1}...")
                db.add_documents(batch)
        
        # Persist the database
        print("Persisting database...")
        db.persist()
        print(f"✅ Successfully saved vector DB to {persist_dir}")
        
        return db
        
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Check SQLite version: python -c 'import sqlite3; print(sqlite3.sqlite_version)'")
        print("2. Update ChromaDB: pip install --upgrade chromadb")
        print("3. Install pysqlite3-binary: pip install pysqlite3-binary")
        return None

def test_comprehensive_db(db_path):
    """Test the comprehensive database with various queries"""
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if not os.path.exists(db_path):
            print(f"Database not found at {db_path}")
            return
        
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Test queries for different document types
        test_queries = [
            # General content queries
            ("cycle count", "user_guide"),
            ("saved view", "user_guide"),
            ("column chooser", "user_guide"),
            ("recount", "user_guide"),
            
            # Glossary queries (if any)
            ("API", "glossary"),
            ("Asset Management", "glossary"),
            
            # FAQ queries (if any)
            ("mobile app", "faq"),
            ("licensing", "faq"),
            
            # General searches
            ("filter column", None),
            ("save view", None),
            ("needs recount", None)
        ]
        
        print(f"\n{'='*80}")
        print("TESTING COMPREHENSIVE DATABASE")
        print('='*80)
        
        for query, expected_type in test_queries:
            print(f"\nQuery: '{query}'", end="")
            if expected_type:
                print(f" (expecting {expected_type})")
            else:
                print(" (any type)")
            print("-" * 50)
            
            try:
                results = db.similarity_search(query, k=3)
                
                if not results:
                    print("  No results found")
                    continue
                
                for i, result in enumerate(results, 1):
                    doc_type = result.metadata.get("document_type", "unknown")
                    source = result.metadata.get("source", "unknown")
                    doc_title = result.metadata.get("document_title", "")
                    section = result.metadata.get("section", "")
                    
                    print(f"  Result {i}: [{doc_type}] {source}")
                    if doc_title and doc_title != source:
                        print(f"    Document: {doc_title}")
                    if section:
                        print(f"    Section: {section}")
                    print(f"    Content: {result.page_content[:150]}...")
                    
                    # Show images if available
                    if result.metadata.get("has_images"):
                        image_count = result.metadata.get("image_count", 0)
                        print(f"    Images: {image_count} image(s)")
                    
                    print()
                    
            except Exception as e:
                print(f"  Error querying '{query}': {e}")
                
    except Exception as e:
        print(f"Error testing database: {e}")

def query_with_filters(db_path, query, document_type=None, document_title=None, k=5):
    """Enhanced query function with optional filtering"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Basic similarity search
        results = db.similarity_search(query, k=k*3)  # Get more to filter
        
        # Apply filters
        filtered_results = []
        for result in results:
            include = True
            
            if document_type and result.metadata.get("document_type") != document_type:
                include = False
            
            if document_title and document_title.lower() not in result.metadata.get("document_title", "").lower():
                include = False
            
            if include:
                filtered_results.append(result)
            
            if len(filtered_results) >= k:
                break
        
        response_data = []
        
        for i, result in enumerate(filtered_results):
            # Convert images string back to list
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
                "has_images": result.metadata.get("has_images", False)
            }
            response_data.append(item)
        
        return response_data
        
    except Exception as e:
        print(f"Error querying database: {e}")
        return []

def get_database_info(db_path):
    """Get information about the database contents"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Get collection info
        collection = db._collection
        doc_count = collection.count()
        
        print(f"\n📊 Database Information:")
        print(f"Total documents: {doc_count}")
        
        # Sample a few documents to show metadata structure
        if doc_count > 0:
            sample_docs = db.similarity_search("", k=min(5, doc_count))
            
            doc_types = set()
            sources = set()
            
            for doc in sample_docs:
                doc_types.add(doc.metadata.get("document_type", "unknown"))
                sources.add(doc.metadata.get("source", "unknown"))
            
            print(f"Document types: {', '.join(doc_types)}")
            print(f"Sources: {', '.join(list(sources)[:5])}")
            if len(sources) > 5:
                print(f"... and {len(sources) - 5} more")
                
    except Exception as e:
        print(f"Error getting database info: {e}")

if __name__ == "__main__":
    # Update this path to your data folder
    data_folder = r"C:\Users\hp\Downloads\final\data"
    db_path = "comprehensive_vector_db"
    
    print("🚀 Starting Vector Database Creation Process...")
    print("=" * 60)
    
    # Create the comprehensive vector database
    print("Creating comprehensive vector database...")
    db = create_comprehensive_vector_db(data_folder, db_path)
    
    if db is None:
        print("❌ Database creation failed. Exiting...")
        sys.exit(1)
    
    # Get database info
    get_database_info(db_path)
    
    # Test the database
    print(f"\n{'='*80}")
    print("TESTING THE DATABASE")
    print('='*80)
    test_comprehensive_db(db_path)
    
    # Example queries
    print(f"\n{'='*80}")
    print("EXAMPLE QUERIES")
    print('='*80)
    
    if os.path.exists(db_path):
        example_queries = [
            ("How to create a saved view?", None),
            ("cycle count recount", None),
            ("column chooser", "user_guide")
        ]
        
        for query, doc_type in example_queries:
            print(f"\nQuery: '{query}'")
            if doc_type:
                print(f"Filter: {doc_type} only")
            print("-" * 40)
            
            results = query_with_filters(db_path, query, doc_type, k=2)
            
            if not results:
                print("  No results found")
                continue
            
            for item in results:
                print(f"  [{item['document_type']}] {item['document_title']}")
                if item['section']:
                    print(f"  Section: {item['section']}")
                print(f"  {item['content'][:100]}...")
                print()
    
    print("\n✅ Vector database creation and testing completed!")
    print(f"Database saved at: {os.path.abspath(db_path)}")
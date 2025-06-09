import os
import zipfile
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import re

def get_image_order_from_document(docx_zip):
    """
    Extract the order of images as they appear in the document by parsing the XML
    """
    image_order = []
    
    try:
        # Parse the main document XML
        with docx_zip.open('word/document.xml') as doc_xml:
            doc_content = doc_xml.read().decode('utf-8')
            
        # Find all image references in order
        # Look for drawing elements and embedded objects
        root = ET.fromstring(doc_content)
        
        # Define namespaces used in Word documents
        namespaces = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
            'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
            'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        }
        
        # Find all embedded images in order
        for elem in root.iter():
            # Look for image embeddings
            if elem.tag.endswith('}embed') or elem.tag.endswith('}link'):
                rel_id = elem.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                if rel_id:
                    image_order.append(rel_id)
        
        # Also check for alternate image references
        for blip in root.iter():
            if blip.tag.endswith('}blip'):
                embed_attr = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                link_attr = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}link')
                if embed_attr:
                    if embed_attr not in image_order:
                        image_order.append(embed_attr)
                elif link_attr:
                    if link_attr not in image_order:
                        image_order.append(link_attr)
    
    except Exception as e:
        print(f"  Warning: Could not parse document structure: {e}")
        return []
    
    return image_order

def get_relationship_mappings(docx_zip):
    """
    Get mapping between relationship IDs and actual media files
    """
    rel_to_media = {}
    
    try:
        with docx_zip.open('word/_rels/document.xml.rels') as rels_xml:
            rels_content = rels_xml.read().decode('utf-8')
        
        root = ET.fromstring(rels_content)
        
        for relationship in root.iter():
            if relationship.tag.endswith('}Relationship'):
                rel_id = relationship.get('Id')
                target = relationship.get('Target')
                rel_type = relationship.get('Type')
                
                # Check if this is an image relationship
                if (rel_type and 'image' in rel_type.lower() and 
                    target and target.startswith('media/')):
                    rel_to_media[rel_id] = f"word/{target}"
    
    except Exception as e:
        print(f"  Warning: Could not parse relationships: {e}")
    
    return rel_to_media

def extract_images_from_docx(docx_path, output_base_dir, doc_number):
    """
    Extract all images and GIFs from a Word document (.docx) in document order
    """
    doc_name = Path(docx_path).stem
    
    # Create subdirectory for this document
    doc_output_dir = Path(output_base_dir) / f"{doc_name}_{doc_number}"
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    
    image_count = 0
    
    try:
        # Open the docx file as a zip
        with zipfile.ZipFile(docx_path, 'r') as docx_zip:
            # Get the order of images in the document
            image_order = get_image_order_from_document(docx_zip)
            
            # Get relationship mappings
            rel_to_media = get_relationship_mappings(docx_zip)
            
            # Supported image extensions
            supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
            
            # If we successfully parsed the document order, use it
            if image_order and rel_to_media:
                processed_files = set()
                
                for rel_id in image_order:
                    if rel_id in rel_to_media:
                        media_file = rel_to_media[rel_id]
                        
                        # Avoid processing the same file multiple times
                        if media_file in processed_files:
                            continue
                        processed_files.add(media_file)
                        
                        if media_file in docx_zip.namelist():
                            file_extension = Path(media_file).suffix.lower()
                            
                            if file_extension in supported_extensions:
                                image_count += 1
                                
                                # Extract the file
                                with docx_zip.open(media_file) as source:
                                    # Create new filename with sequential number
                                    output_filename = f"{image_count}{file_extension}"
                                    output_path = doc_output_dir / output_filename
                                    
                                    # Save the image
                                    with open(output_path, 'wb') as target:
                                        shutil.copyfileobj(source, target)
                                    
                                    print(f"  Extracted: {output_filename} (from {Path(media_file).name})")
            
            # Fallback: if document parsing failed, extract all media files
            if image_count == 0:
                print("  Using fallback method - extracting all media files...")
                
                # Get all media files and sort them by name for consistent ordering
                media_files = [f for f in docx_zip.namelist() if f.startswith('word/media/')]
                media_files.sort()  # Sort alphabetically for consistent order
                
                for media_file in media_files:
                    file_extension = Path(media_file).suffix.lower()
                    
                    if file_extension in supported_extensions:
                        image_count += 1
                        
                        # Extract the file
                        with docx_zip.open(media_file) as source:
                            # Create new filename with sequential number
                            output_filename = f"{image_count}{file_extension}"
                            output_path = doc_output_dir / output_filename
                            
                            # Save the image
                            with open(output_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            
                            print(f"  Extracted: {output_filename} (from {Path(media_file).name})")
    
    except zipfile.BadZipFile:
        print(f"  Error: {docx_path} is not a valid .docx file")
        return 0
    except Exception as e:
        print(f"  Error processing {docx_path}: {str(e)}")
        return 0
    
    return image_count

def process_word_documents(input_directory=r"C:\Users\hp\Downloads\English-20250601T110944Z-1-001\user manual\Modules\Work Requests", output_base_dir=r"C:\Users\hp\Downloads\final"):
    """
    Process all Word documents in the specified directory or current directory
    """
    # Set input directory (current directory if not specified)
    if input_directory is None:
        input_directory = Path.cwd()
    else:
        input_directory = Path(input_directory)
    
    # Create base output directory structure
    output_base_dir = Path(output_base_dir)
    img_dir = output_base_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing Word documents from: {input_directory}")
    print(f"Output directory: {img_dir}")
    print("-" * 50)
    
    # Find all .docx files and sort them for consistent processing order
    docx_files = sorted(list(input_directory.glob("*.docx")))
    
    if not docx_files:
        print("No .docx files found in the specified directory.")
        return
    
    total_images = 0
    doc_count = 0
    
    # Process each document
    for docx_file in docx_files:
        doc_count += 1
        print(f"\nProcessing Document {doc_count}: {docx_file.name}")
        
        image_count = extract_images_from_docx(docx_file, img_dir, doc_count)
        total_images += image_count
        
        if image_count == 0:
            print(f"  No images found in {docx_file.name}")
        else:
            print(f"  Extracted {image_count} images from {docx_file.name}")
    
    print("\n" + "=" * 50)
    print(f"Processing complete!")
    print(f"Total documents processed: {doc_count}")
    print(f"Total images extracted: {total_images}")
    print(f"Images saved to: {img_dir}")

def main():
    """
    Main function - you can modify the input directory here
    """
    # Option 1: Process documents from current directory
    process_word_documents()
    
    # Option 2: Process documents from a specific directory (uncomment and modify path below)
    # process_word_documents(input_directory=r"C:\path\to\your\word\documents")

if __name__ == "__main__":
    main()
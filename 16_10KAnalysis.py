# Databricks notebook source
!pip install pdfplumber


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pdfplumber
import pypdfium2 as pdfium


# COMMAND ----------

PDF_ROOT = "/Volumes/itesm/finanzas/nvda"

# COMMAND ----------

def list_pdfs():
    base = f"{PDF_ROOT}"
    return [f"{base}/{f}" for f in os.listdir(base) if f.endswith(".pdf")]

# COMMAND ----------

import os
list_pdfs()

# COMMAND ----------

def extract_page(pdf_path, page_i, ocr):
    text = pdfplumber.open(pdf_path).pages[page_i].extract_text() or ""
    if len(text.strip()) > 40:        # simple quality threshold
        return text, "digital"        # good digital text detected
    # fallback for scanned pages
    img = pdfium.PdfDocument(pdf_path)[page_i].render(scale=2).to_pil()
    return "\n".join(ocr.readtext(img, detail=0)), "ocr"

# COMMAND ----------

import os
from concurrent.futures import ThreadPoolExecutor

def extract_pages(pdf_path, ocr):
    with ThreadPoolExecutor(max_workers=4) as executor:
        return [
            (page_i, *result)
            for page_i, result in enumerate(executor.map(lambda page_i: extract_page(pdf_path, page_i, ocr), range(len(pdfium.PdfDocument(pdf_path)))))
        ]
import os
from concurrent.futures import ThreadPoolExecutor



# COMMAND ----------

extract_page("/Volumes/itesm/finanzas/nvda/Receipt Date - 2023-Feb-24 - NVDA.OQ - NVIDIA Corp - 10-K - NVIDIA CORP 10-K - 23668751.pdf", 0, 'ocr')


# COMMAND ----------

extract_page('/Volumes/itesm/finanzas/nvda/Receipt Date - 2023-Feb-24 - NVDA.OQ - NVIDIA Corp - 10-K - NVIDIA CORP 10-K - 23668751.pdf', 1, 'ocr')

# COMMAND ----------

extract_page('/Volumes/itesm/finanzas/nvda/Receipt Date - 2023-Feb-24 - NVDA.OQ - NVIDIA Corp - 10-K - NVIDIA CORP 10-K - 23668751.pdf', 2, 'ocr')

# COMMAND ----------

# MAGIC %md
# MAGIC utilizando la función extract_page genera una nueva función que extraiga todas las páginas del pdf y las integre en una solo texto. Aplícalo sobre /Volumes/itesm/finanzas/nvda/Receipt Date - 2023-Feb-24 - NVDA.OQ - NVIDIA Corp - 10-K - NVIDIA CORP 10-K - 23668751.pdf y devuelve el resultado.

# COMMAND ----------

def extract_all_text(pdf_path, ocr):
    num_pages = len(pdfium.PdfDocument(pdf_path))
    texts = [extract_page(pdf_path, i, ocr)[0] for i in range(num_pages)]
    return "\n".join(texts)

result = extract_all_text(
    "/Volumes/itesm/finanzas/nvda/Receipt Date - 2023-Feb-24 - NVDA.OQ - NVIDIA Corp - 10-K - NVIDIA CORP 10-K - 23668751.pdf",
    'ocr'
)
display(result)

# COMMAND ----------

!pip install langchain_text_splitters

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

def perform_semantic_chunking(document, chunk_size=500, chunk_overlap=100):
    """
    Performs semantic chunking on a document using recursive character splitting 
    at logical text boundaries.
    
    Args:
        document (str): The text document to process
        chunk_size (int): The target size of each chunk in characters
        chunk_overlap (int): The number of characters of overlap between chunks
        
    Returns:
        list: The semantically chunked documents with metadata
    """
    # Create the text splitter with semantic separators
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Split the text into semantic chunks
    semantic_chunks = text_splitter.split_text(document)
    print(f"Document split into {len(semantic_chunks)} semantic chunks")
    
    # Determine section titles for enhanced metadata
    section_patterns = [
        r'^#+\s+(.+)$',      # Markdown headers
        r'^.+\n[=\-]{2,}$',  # Underlined headers
        r'^[A-Z\s]+:$'       # ALL CAPS section titles
    ]
    
    # Convert to Document objects with enhanced metadata
    documents = []
    current_section = "Introduction"
    
    for i, chunk in enumerate(semantic_chunks):
        # Try to identify section title from chunk
        chunk_lines = chunk.split('\n')
        for line in chunk_lines:
            for pattern in section_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    current_section = match.group(0)
                    break
        
        # Calculate semantic density (ratio of non-stopwords to total words)
        words = re.findall(r'\b\w+\b', chunk.lower())
        stopwords = ['the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it', 'with', 'as', 'for']
        content_words = [w for w in words if w not in stopwords]
        semantic_density = len(content_words) / max(1, len(words))
        
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "total_chunks": len(semantic_chunks),
                "chunk_size": len(chunk),
                "chunk_type": "semantic",
                "section": current_section,
                "semantic_density": round(semantic_density, 2)
            }
        )
        documents.append(doc)
    
    return documents


# COMMAND ----------


# Example usage with Databricks integration
if __name__ == "__main__":

    # Create the dummy document
    #document = create_dummy_document()
    
    # Process with semantic chunking
    chunked_docs = perform_semantic_chunking(
        result,
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Display results
    print("\n----- CHUNKING RESULTS -----")
    print(f"Total semantic chunks: {len(chunked_docs)}")
    
    # Print an example chunk
    print("\n----- EXAMPLE SEMANTIC CHUNK -----")
    middle_chunk_idx = len(chunked_docs) // 2
    example_chunk = chunked_docs[middle_chunk_idx]
    print(f"Chunk {middle_chunk_idx} content ({len(example_chunk.page_content)} characters):")
    print("-" * 40)
    print(example_chunk.page_content)
    print("-" * 40)
    print(f"Metadata: {example_chunk.metadata}")
    
    # Optional: Calculate section distribution for analysis
    section_counts = {}
    for doc in chunked_docs:
        section = doc.metadata["section"]
        section_counts[section] = section_counts.get(section, 0) + 1
    
    print("\n----- SECTION DISTRIBUTION -----")
    for section, count in section_counts.items():
        print(f"{section}: {count} chunks")
    
    # For integration with Databricks embeddings
    print("\nTo integrate with Databricks:")
    print("1. Create embeddings using the Databricks embedding API:")
    print("   from langchain_community.embeddings import DatabricksEmbeddings")
    print("   embeddings = DatabricksEmbeddings(endpoint='databricks-bge-large-en')")
    print("2. Store documents and embeddings in Delta table")
    print("3. Create Vector Search index using the semantic metadata for filtering")

# COMMAND ----------

chunked_docs

# COMMAND ----------


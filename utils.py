import hashlib
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extrai o texto de um PDF e retorna uma lista de dicionários com o conteúdo de cada página."""
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:  # Only include pages with text
            pages.append({
                "page_number": page_num + 1,
                "content": text,
                "file_name": Path(pdf_path).name
            })
    
    doc.close()
    return pages

def split_text(text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
    """Vai dividir o texto em pedaços menores com base no tamanho do chunk e na sobreposição."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)

def generate_doc_id(text: str) -> str:
    """Gera ids para cada chunk."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def clean_text(text: str) -> str:
    """Limpa o texto removendo caracteres especiais e espaços extras."""
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()

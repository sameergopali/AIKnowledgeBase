from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from typing import Optional, List
import hashlib
import tempfile
import os


class UploadService:
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt'}
    SUPPORTED_MIME_TYPES = {
        'application/pdf',
        'text/plain',
        'text/txt'
    }
    
    def __init__(self, db, embedding_model=None, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.db = db
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk(self, file_content: bytes, file_type: Optional[str] = None) -> List[str]:
        documents = self.parse_document(file_content, file_type)
        if not documents:
            raise Exception("Document contains no extractable text")
        
        full_text = "\n\n".join([doc.page_content for doc in documents])
        if not full_text.strip():
            raise Exception("Document contains no extractable text")
        
        chunks = self.text_splitter.split_text(full_text)
        return chunks
    
    def parse_document(self, file_content: bytes, file_type: Optional[str] = None) -> List:
        if not file_content:
            raise Exception("Empty file content")
        
        temp_file = None
        try:
            extension = self._get_file_extension(file_type)
            if extension not in self.SUPPORTED_EXTENSIONS:
                raise Exception(f"Unsupported file type: {extension}")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
            temp_file.write(file_content)
            temp_file.close()
            
            if extension == '.pdf':
                loader = PyPDFLoader(temp_file.name)
            else:
                loader = TextLoader(temp_file.name, autodetect_encoding=True)
            
            documents = loader.load()
            if not documents:
                raise Exception("No content extracted from document")
            
            return documents
        
        except Exception as e:
            print(f"Error parsing document: {e}")
            raise
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as cleanup_error:
                    print(f"Failed to delete temp file: {cleanup_error}")
    
    def _get_file_extension(self, file_type: Optional[str]) -> str:
        if not file_type:
            return '.txt'
        
        file_type_lower = file_type.lower()
        if 'pdf' in file_type_lower or file_type_lower.endswith('.pdf'):
            return '.pdf'
        elif 'text' in file_type_lower or 'txt' in file_type_lower or file_type_lower.endswith('.txt'):
            return '.txt'
        else:
            return '.txt'
    
    def _validate_file_type(self, filename: str, content_type: Optional[str]) -> None:
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext and file_ext not in self.SUPPORTED_EXTENSIONS:
            print(f"Unsupported file extension: {file_ext}")
            raise Exception(f"Unsupported file extension: {file_ext}")
        
        if content_type and content_type not in self.SUPPORTED_MIME_TYPES:
            if 'pdf' not in content_type.lower() and 'text' not in content_type.lower():
                print(f"Unsupported MIME type: {content_type}")
                raise Exception(f"Unsupported MIME type: {content_type}")
    
    async def index_document(self, file: UploadFile) -> str:
        try:
            self._validate_file_type(file.filename, file.content_type)
            file_content = await file.read()
            if not file_content:
                raise Exception("Empty file uploaded")
            
            doc_id = self._generate_doc_id(file.filename, file_content)
            chunks = self.chunk(file_content, file.content_type or file.filename)
            
            if not chunks:
                raise Exception("No chunks generated from document")
            
            doc_metadata = {
                "doc_id": doc_id,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(file_content),
                "num_chunks": len(chunks)
            }
            
            await self._store_document(doc_id, doc_metadata, chunks)
            return doc_id
        
        except Exception as e:
            print(f"Error indexing document: {e}")
            raise
        finally:
            await file.seek(0)
    
    def _generate_doc_id(self, filename: str, content: bytes) -> str:
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        filename_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"{filename_hash}_{content_hash}"
    
    async def _store_document(self, doc_id: str, metadata: dict, chunks: List[str]):
        try:
            self.db.add_documents(
                ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
                documents=chunks,
                metadatas=[{**metadata, "chunk_index": i} for i in range(len(chunks))]
            )
        except Exception as e:
            print(f"Error storing document: {e}")
            raise

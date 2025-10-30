
from fastapi import APIRouter, UploadFile, File, HTTPException



class UploadController:
    def __init__(self, upload_service):
        self.upload_service = upload_service
        self.router = APIRouter(prefix="/document", tags=["upload"])
        self.register_routes()
        self.uploaded_files = []  # To keep track of uploaded files

    def register_routes(self):
        @self.router.post("/", summary="Upload and index documents")
        async def upload_files(file: UploadFile = File(...)):
            try:
                doc_id = await self.upload_service.index_document(file)
                self.uploaded_files.append({"doc_id": doc_id, "filename": file.filename})
                return {"message": "Document indexed successfully", "doc_id": doc_id, "filename": file.filename}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
 
    
        @self.router.get("/", summary="List uploaded documents")
        async def list_uploaded_documents():
            return {"documents": self.uploaded_files}
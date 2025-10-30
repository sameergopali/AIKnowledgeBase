from fastapi import APIRouter, HTTPException


class ChatController:
    def __init__(self, chat_service):
        self.chat_service = chat_service
        self.router = APIRouter(prefix="/chat", tags=["chat"])
        self.register_routes()

    def register_routes(self):
        @self.router.post("/", summary="Chat endpoint")
        async def chat_endpoint(body: dict):
            try:
                message = body.get("message", "")
                rag = body.get("rag","suggestion-rag")
                response = self.chat_service.chat(message, rag)
                return response
            except Exception as e:
                print(f"Error in chat endpoint: {e}")
                raise HTTPException(status_code=400, detail=str(e))

            
    
    
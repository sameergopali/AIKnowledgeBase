from repository import ChromaDB
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller import UploadController, ChatController
from service import UploadService, ChatService
import uvicorn





load_dotenv()



def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    
    upload_service = UploadService(ChromaDB())
    upload_controller = UploadController(upload_service)

    chat_service = ChatService(ChromaDB(), ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite"))
    chat_controller = ChatController(chat_service)
    app.include_router(chat_controller.router)

    # Register routes
    app.include_router(upload_controller.router)
    app.include_router(chat_controller.router)

    return app


    

def main():
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
if __name__ == "__main__":
    main()

    
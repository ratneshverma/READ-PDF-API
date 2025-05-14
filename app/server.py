from app.BAL.UserClientPDF import process_file, FileProcessingRequest
from app.BAL.KnowledgeJSON import knowledge_base, KnowledgeRequest
from app.BAL.KnowledgePDF import knowledgePDF_base, KnowledgePDFRequest
from app.BAL.Forms import process_forms, FormsRequest 
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.runnables import RunnableLambda

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

add_routes(
    app,
    RunnableLambda(process_file).with_types(input_type=FileProcessingRequest),
    config_keys=["configurable"],
    path="/pdf",
)

add_routes(
    app,
    RunnableLambda(knowledge_base).with_types(input_type=KnowledgeRequest),
    config_keys=["configurable"],
    path="/knowledgeJSON",
)

add_routes(
    app,
    RunnableLambda(knowledgePDF_base).with_types(input_type=KnowledgePDFRequest),
    config_keys=["configurable"],
    path="/knowledgePDF",
)

add_routes(
    app,
    RunnableLambda(process_forms).with_types(input_type=FormsRequest),
    config_keys=["configurable"],
    path="/formsData",
)

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "https://edc-ai.galaxbiotech.com, http://localhost:3000, http://localhost:3001"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

origins = [
"http://localhost:3000",
"https://edc-ai.galaxbiotech.com",
"http://localhost:3001"
]

app.add_middleware(
CORSMiddleware,
allow_origins=origins,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

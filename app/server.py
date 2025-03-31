from app.Entities.User import User
from app.Entities.Client import Client
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import CustomUserType, add_routes
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from langchain_core.document_loaders import Blob
from pydantic import Field, BaseModel
import base64
from langchain_community.llms import Ollama
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, \
    HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser

llm = Ollama(model="llama3.2")
app = FastAPI()



@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


class FileProcessingRequest(CustomUserType):
    """Request including a base64 encoded file."""
    # The extra field is used to specify a widget for the playground UI.
    file: str = Field(..., extra={"widget": {"type": "base64file"}})
    calltype: str = "User"


def _process_file(request: FileProcessingRequest) -> str:
    """Extract the text from the first page of the PDF."""
    content = base64.b64decode(request.file.encode("utf-8"))
    blob = Blob(data=content)
    documents = list(PDFMinerParser().lazy_parse(blob))
    content = documents[0].page_content
    result = extract_values_from_file(content, request.calltype)
    return result

def extract_values_from_file(raw_file_data, callType):
    
    preamble = ("\n"
                "Your ability to extract and summarize this information accurately is essential for effective "
                "do document analysis. Pay close attention to the credit card statement's language, "
                "structure, and any cross-references to ensure a comprehensive and precise extraction of "
                "information. Do not use prior knowledge or information from outside the context to answer the "
                "questions. Only use the information provided in the context to answer the questions.\n")
    postamble = ""
    if callType == "User":
        postamble = ("\n"
                    "Do not include any explanation in the reply. Only include the extracted information in the reply."
                    "Answer in Json format don't include any other properties other than below "
                    "{'first_name':<your answer>, 'middle_name':<your answer>,'last_name':<your answer>, 'address':<your answer>, "
                    "'designation_id':<your answer>, 'email_id':<your answer>, 'phone':<your answer>, 'office_phone':<your answer>, "
                    "'city_id':<your answer>, 'state_id':<your answer>, 'country_id':<your answer>, 'postal_code':<your answer>, 'management_id':<your answer>}")
    else:
        postamble = ("\n"
                    "Do not include any explanation in the reply. Only include the extracted information in the reply."
                    "Answer in Json format don't include any other properties other than below "
                    "{'company_name':<your answer>, 'short_name':<your answer>,'website':<your answer>, 'address':<your answer>, "
                    "'tier':<your answer>, 'phone':<your answer>, 'office_phone':<your answer>, 'city_id':<your answer>, "
                    "'state_id':<your answer>, 'country_id':<your answer>, 'postal_code':<your answer>, 'sponsor_identification':<your answer>}")    
    
    system_template = "{preamble}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{format_instructions}\n\n{raw_file_data}\n\n{postamble}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


    
    
    print("Response from model")
    if callType == "User":
        parser = PydanticOutputParser(pydantic_object=User)
        print(parser.get_format_instructions())
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        request = chat_prompt.format_prompt(preamble=preamble,
                                        format_instructions=parser.get_format_instructions(),
                                        raw_file_data=raw_file_data,
                                        postamble=postamble).to_messages()
        print("Querying model...")
        result = llm.invoke(request, temperature=0)
        res = parser.parse(result)
        print(res.model_dump_json())
        return res.model_dump_json()
    else:
        parser = PydanticOutputParser(pydantic_object=Client)
        print(parser.get_format_instructions())
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        request = chat_prompt.format_prompt(preamble=preamble,
                                        format_instructions=parser.get_format_instructions(),
                                        raw_file_data=raw_file_data,
                                        postamble=postamble).to_messages()
        print("Querying model...")
        result = llm.invoke(request, temperature=0)
        res = parser.parse(result)
        print(res.model_dump_json())
        return res.model_dump_json()

add_routes(
    app,
    RunnableLambda(_process_file).with_types(input_type=FileProcessingRequest),
    config_keys=["configurable"],
    path="/pdf",
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
"http://localhost:3000"
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

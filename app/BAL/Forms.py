from typing import Optional

from pydantic import Field
from app.Entities.Demographic import Demographic
from langserve import CustomUserType
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, \
    HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS





llm = OllamaLLM(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model='nomic-embed-text')

class FormsRequest(CustomUserType):
    site: str
    subject: str
    visit: str
    forms: str
    #content: str = Field(default="The query")

def process_forms(request: FormsRequest) -> str:
    #Populate all fields from demographic Site1-02-01-01-V0_SCR-Demographic
    metaindex = f"{request.site}-{request.subject}-{request.visit}-{request.forms}"
    content = f"please find the values for site {request.site}, subject {request.subject} and visit {request.visit} for {request.forms}"
    
    new_vector_store = FAISS.load_local(
    "galaxbiotech-vectorDB", embeddings, allow_dangerous_deserialization=True)
    
    content_raw = new_vector_store.similarity_search(   
    content,
    k=1,
    filter={"source": {"$eq": metaindex}})
    if len(content_raw) > 0:
        raw_file_data = content_raw[0].page_content

        preamble = ("\n"
                    "Your ability to extract and summarize this information accurately is essential for effective "
                    "do document analysis. Pay close attention to the credit card statement's language, "
                    "structure, and any cross-references to ensure a comprehensive and precise extraction of "
                    "information. Do not use prior knowledge or information from outside the context to answer the "
                    "questions. Only use the information provided in the context to answer the questions.\n")
        postamble = ""
        if request.forms == "Demographic":
            postamble = ("\n"
                        "Do not include any explanation in the reply. Only include the extracted information in the reply."
                        "Answer in Json format don't include any other properties other than below "
                        "{'demographics_collection_date':<your answer>, 'birth_date':<your answer>,'age':<your answer>, 'sex':<your answer>, "
                        "'collected_ethnicity':<your answer>, 'if_mixed_ethnicity':<your answer>, 'other_ethnicity':<your answer>, 'collected_race':<your answer>, "
                        "'multiple_race_if_any':<your answer>, 'race_other':<your answer>, 'occupation':<your answer>}")
        
        
        system_template = "{preamble}"
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = "{format_instructions}\n\n{raw_file_data}\n\n{postamble}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


        
        
        print("Response from model")
        if request.forms == "Demographic":
            parser = PydanticOutputParser(pydantic_object=Demographic)
            print(parser.get_format_instructions())
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            request = chat_prompt.format_prompt(preamble=preamble,
                                            format_instructions=parser.get_format_instructions(),
                                            raw_file_data=raw_file_data,
                                            postamble=postamble).to_messages()
            print("Querying model...")
            result = llm.invoke(request)
            res = parser.parse(result)
            print(res.model_dump_json())
            return res.model_dump_json()
    else:
        return "Not Found"
    


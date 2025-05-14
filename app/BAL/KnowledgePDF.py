from langserve import CustomUserType
import pdfplumber
from pydantic import Field
import base64
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from pdfminer.high_level import extract_pages
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.document_loaders import Blob
import re
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, \
    HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from app.Entities.Demographic import Demographic
from app.Utility.VectorDBUtil import RemoveVectors


llm = OllamaLLM(model="llama3.2", temperature=0)

embeddings = OllamaEmbeddings(model='nomic-embed-text')

index = faiss.IndexFlatL2(len(embeddings.embed_query("galaxbiotech-vectorDB")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


class KnowledgePDFRequest(CustomUserType):
    file: str = Field(..., extra={"widget": {"type": "base64file"}})
    subject: str = "None"

def ReadPDFContent(output_file):
    pdf = pdfplumber.open(output_file)
    finalContent = ''
    header = ''
    for idx,page in enumerate(pdf.pages, start=1):  
        print(f"Reading page {idx}")
        contList = []
        for e in page.chars:             
            tmpRow = ["char", e["text"], e["x0"], e["y0"]]
            contList.append(tmpRow)
        for e in page.curves:
            tmpRow = ["curve", e["pts"], e["x0"], e["y0"]]
            contList.append(tmpRow)  
        contList.sort(key=lambda x: x[2])
        contList.sort(key=lambda x: x[3], reverse=True)
        workContent = []    
        workText = ""
        workDistCharX = False
        prev=0
        i=0
        for e in contList:
            if e[0] == "char":
                if workDistCharX != False and \
                    (e[2] - workDistCharX > 20 or e[3] - workDistCharY < -2):
                    workText += " / "
                workText += e[1]
                workDistCharX = e[2]
                workDistCharY = e[3]
                continue
            if e[0] == "curve":
                if workText != "":
                    workContent.append(workText)
                    workText = ""
                if len(e[1]) == 34:
                    tmpVal = "SELECT-YES"
                    if workContent[len(workContent)-1] == "SELECT-YES":
                        workContent[len(workContent)-1] = "SELECT-NO"
                    else:
                        workContent.append(tmpVal)
            
        #   else:
        #     tmpVal = "SELECT-NO"
        header = workContent[0]
        if idx == 2:
            finalContent = finalContent + "\n"
        finalContent = finalContent + "\n".join(workContent[1:])
    return finalContent, header

def extract_key_value(text, pattern):
    """
    Extracts the key and value from a string using a regular expression pattern.

    Args:
        text (str): The input string to search within.
        pattern (str): The regular expression pattern with capturing groups for key and value.

    Returns:
        dict: A dictionary containing the extracted key-value pairs, or an empty dictionary if no match is found.
    """
    match = re.search(pattern, text)
    if match:
        return {match.group(1): match.group(2)}
    return {} 

def knowledgePDF_base(request: KnowledgePDFRequest) -> str:
    content = base64.b64decode(request.file.encode("utf-8"))
    output_file = "output.pdf"
    with open(output_file, "wb") as pdf_file:
        pdf_file.write(content)

    finalContent, header = ReadPDFContent(output_file=output_file)
    print(finalContent)
    pattern = r"(Form):(\s+[A-Za-z]+)"
    formNameKey = extract_key_value(header, pattern)
    formname= formNameKey['Form'].strip(' ')
    if formname == "Demographic":
        raw_file_data, sitename, subjectname, vistname = CleanDemographicData(finalContent, header)
        if request.subject == "None" and subjectname == '':
            return "Subject value is not present in PDF or in parameter"
        if request.subject != "None":
            subjectname = request.subject
        if sitename == '':
            return "Site value is not present in PDF"
        if vistname == '':
            return "Visit value is not present in PDF"
        

        if vistname == '':
            return "Visit value is not present in PDF"

        res = RAGDemographic(raw_file_data)
        
        if res.multiple_race_if_any == "SELECT-YES" or res.multiple_race_if_any == "SELECT-NO":
            res.multiple_race_if_any = ""

        SaveDemographicVectors(res, sitename, subjectname, vistname, formname)
        print(res.model_dump_json())
        return res.model_dump_json()
    else:
        return "Supported only Demographic PDF for time being"
    
def RAGDemographic(raw_file_data):
    preamble = ("\n"
                    "Your ability to extract and summarize this information accurately is essential for effective "
                    "do document analysis. Pay close attention to the credit card statement's language, "
                    "structure, and any cross-references to ensure a comprehensive and precise extraction of "
                    "information. Do not use prior knowledge or information from outside the context to answer the "
                    "questions. Only use the information provided in the context to answer the questions.\n")
    postamble = ""
    
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
    return res

def CleanDemographicData(finalContent, header):
    finalContent = finalContent.replace("|", " ")
    finalContent = finalContent.replace("/", "")
    finalContent = finalContent.replace("(", "")
    finalContent = finalContent.replace(")", "")

    updated_text = re.sub(r"Sex   SEX\nSELECT-YES\n  Male\nSELECT-NO\n  Female", "Sex: Male", finalContent)
    updated_text = re.sub(r"Sex   SEX\nSELECT-NO\n  Male\nSELECT-YES\n  Female", "Sex: Female", updated_text)
    updated_text = re.sub(r"Demographics Collection Date   DMDAT  Day   DMDATDMonth   DMDATM  Year   DMDATY", "Demographics Collection Date:", updated_text)
    updated_text = re.sub(r"Birth Date   BRTHDAT  Birth Day   BRTHDDBirth Month   BRTHMOBirth Year   BRTHYY", "Birth Date:", updated_text)
    updated_text = re.sub(r"Age   AGE", "Age:", updated_text)

    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-NO\n  HISPANIC OR LATINO\nSELECT-NO\n  NOT HISPANIC OR LATINO\nSELECT-NO\n  MIDDLE EASTERNER\nSELECT-NO\n  INDIAN INDIAN SUBCONTINENT\nSELECT-NO\n  JAPANESE\nSELECT-NO\n  MIXED ETHNICITY\nSELECT-NO\n  NOT REPORTED\nSELECT-NO\n  OTHER\nSELECT-NO\n  UNKNOWN", "Collected Ethnicity: None", updated_text)
    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-YES\n  HISPANIC OR LATINO\nSELECT-NO\n  NOT HISPANIC OR LATINO\nSELECT-NO\n  MIDDLE EASTERNER\nSELECT-NO\n  INDIAN INDIAN SUBCONTINENT\nSELECT-NO\n  JAPANESE\nSELECT-NO\n  MIXED ETHNICITY\nSELECT-NO\n  NOT REPORTED\nSELECT-NO\n  OTHER\nSELECT-NO\n  UNKNOWN", "Collected Ethnicity: HISPANIC OR LATINO", updated_text)
    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-NO\n  HISPANIC OR LATINO\nSELECT-YES\n  NOT HISPANIC OR LATINO\nSELECT-NO\n  MIDDLE EASTERNER\nSELECT-NO\n  INDIAN INDIAN SUBCONTINENT\nSELECT-NO\n  JAPANESE\nSELECT-NO\n  MIXED ETHNICITY\nSELECT-NO\n  NOT REPORTED\nSELECT-NO\n  OTHER\nSELECT-NO\n  UNKNOWN", "Collected Ethnicity: NOT HISPANIC OR LATINO", updated_text)
    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-NO\n  HISPANIC OR LATINO\nSELECT-NO\n  NOT HISPANIC OR LATINO\nSELECT-YES\n  MIDDLE EASTERNER\nSELECT-NO\n  INDIAN INDIAN SUBCONTINENT\nSELECT-NO\n  JAPANESE\nSELECT-NO\n  MIXED ETHNICITY\nSELECT-NO\n  NOT REPORTED\nSELECT-NO\n  OTHER\nSELECT-NO\n  UNKNOWN", "Collected Ethnicity: MIDDLE EASTERNER", updated_text)
    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-NO\n  HISPANIC OR LATINO\nSELECT-NO\n  NOT HISPANIC OR LATINO\nSELECT-NO\n  MIDDLE EASTERNER\nSELECT-YES\n  INDIAN INDIAN SUBCONTINENT\nSELECT-NO\n  JAPANESE\nSELECT-NO\n  MIXED ETHNICITY\nSELECT-NO\n  NOT REPORTED\nSELECT-NO\n  OTHER\nSELECT-NO\n  UNKNOWN", "Collected Ethnicity: INDIAN (INDIAN SUBCONTINENT)", updated_text)
    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-NO\n  HISPANIC OR LATINO\nSELECT-NO\n  NOT HISPANIC OR LATINO\nSELECT-NO\n  MIDDLE EASTERNER\nSELECT-NO\n  INDIAN INDIAN SUBCONTINENT\nSELECT-YES\n  JAPANESE\nSELECT-NO\n  MIXED ETHNICITY\nSELECT-NO\n  NOT REPORTED\nSELECT-NO\n  OTHER\nSELECT-NO\n  UNKNOWN", "Collected Ethnicity: JAPANESE", updated_text)
    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-NO\n  HISPANIC OR LATINO\nSELECT-NO\n  NOT HISPANIC OR LATINO\nSELECT-NO\n  MIDDLE EASTERNER\nSELECT-NO\n  INDIAN INDIAN SUBCONTINENT\nSELECT-NO\n  JAPANESE\nSELECT-YES\n  MIXED ETHNICITY\nSELECT-NO\n  NOT REPORTED\nSELECT-NO\n  OTHER\nSELECT-NO\n  UNKNOWN", "Collected Ethnicity: MIXED ETHNICITY", updated_text)
    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-NO\n  HISPANIC OR LATINO\nSELECT-NO\n  NOT HISPANIC OR LATINO\nSELECT-NO\n  MIDDLE EASTERNER\nSELECT-NO\n  INDIAN INDIAN SUBCONTINENT\nSELECT-NO\n  JAPANESE\nSELECT-NO\n  MIXED ETHNICITY\nSELECT-YES\n  NOT REPORTED\nSELECT-NO\n  OTHER\nSELECT-NO\n  UNKNOWN", "Collected Ethnicity: NOT REPORTED", updated_text)
    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-NO\n  HISPANIC OR LATINO\nSELECT-NO\n  NOT HISPANIC OR LATINO\nSELECT-NO\n  MIDDLE EASTERNER\nSELECT-NO\n  INDIAN INDIAN SUBCONTINENT\nSELECT-NO\n  JAPANESE\nSELECT-NO\n  MIXED ETHNICITY\nSELECT-NO\n  NOT REPORTED\nSELECT-YES\n  OTHER\nSELECT-NO\n  UNKNOWN", "Collected Ethnicity: OTHER", updated_text)
    updated_text = re.sub(r"Collected Ethnicity   CETHNIC\nSELECT-NO\n  HISPANIC OR LATINO\nSELECT-NO\n  NOT HISPANIC OR LATINO\nSELECT-NO\n  MIDDLE EASTERNER\nSELECT-NO\n  INDIAN INDIAN SUBCONTINENT\nSELECT-NO\n  JAPANESE\nSELECT-NO\n  MIXED ETHNICITY\nSELECT-NO\n  NOT REPORTED\nSELECT-NO\n  OTHER\nSELECT-YES\n  UNKNOWN", "Collected Ethnicity: UNKNOWN", updated_text)


    updated_text = re.sub(r"Collected Race   CRACE\nSELECT-NO\n  AMERICAN INDIAN OR ALASKA NATIVE\nSELECT-NO\n  ASIAN\nSELECT-NO\n  BLACK OR AFRICAN AMERICAN\nSELECT-NO\n  CAUCASIAN\nSELECT-NO\n  NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER\nSELECT-NO\n  WHITE\nSELECT-NO\n  MULTIPLE\nSELECT-NO\n  OTHER", "Collected Race: None", updated_text)
    updated_text = re.sub(r"Collected Race   CRACE\nSELECT-YES\n  AMERICAN INDIAN OR ALASKA NATIVE\nSELECT-NO\n  ASIAN\nSELECT-NO\n  BLACK OR AFRICAN AMERICAN\nSELECT-NO\n  CAUCASIAN\nSELECT-NO\n  NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER\nSELECT-NO\n  WHITE\nSELECT-NO\n  MULTIPLE\nSELECT-NO\n  OTHER", "Collected Race: AMERICAN INDIAN OR ALASKA NATIVE", updated_text)
    updated_text = re.sub(r"Collected Race   CRACE\nSELECT-NO\n  AMERICAN INDIAN OR ALASKA NATIVE\nSELECT-YES\n  ASIAN\nSELECT-NO\n  BLACK OR AFRICAN AMERICAN\nSELECT-NO\n  CAUCASIAN\nSELECT-NO\n  NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER\nSELECT-NO\n  WHITE\nSELECT-NO\n  MULTIPLE\nSELECT-NO\n  OTHER", "Collected Race: ASIAN", updated_text)
    updated_text = re.sub(r"Collected Race   CRACE\nSELECT-NO\n  AMERICAN INDIAN OR ALASKA NATIVE\nSELECT-NO\n  ASIAN\nSELECT-YES\n  BLACK OR AFRICAN AMERICAN\nSELECT-NO\n  CAUCASIAN\nSELECT-NO\n  NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER\nSELECT-NO\n  WHITE\nSELECT-NO\n  MULTIPLE\nSELECT-NO\n  OTHER", "Collected Race: BLACK OR AFRICAN AMERICAN", updated_text)
    updated_text = re.sub(r"Collected Race   CRACE\nSELECT-NO\n  AMERICAN INDIAN OR ALASKA NATIVE\nSELECT-NO\n  ASIAN\nSELECT-NO\n  BLACK OR AFRICAN AMERICAN\nSELECT-YES\n  CAUCASIAN\nSELECT-NO\n  NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER\nSELECT-NO\n  WHITE\nSELECT-NO\n  MULTIPLE\nSELECT-NO\n  OTHER", "Collected Race: CAUCASIAN", updated_text)
    updated_text = re.sub(r"Collected Race   CRACE\nSELECT-NO\n  AMERICAN INDIAN OR ALASKA NATIVE\nSELECT-NO\n  ASIAN\nSELECT-NO\n  BLACK OR AFRICAN AMERICAN\nSELECT-NO\n  CAUCASIAN\nSELECT-YES\n  NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER\nSELECT-NO\n  WHITE\nSELECT-NO\n  MULTIPLE\nSELECT-NO\n  OTHER", "Collected Race: NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER", updated_text)
    updated_text = re.sub(r"Collected Race   CRACE\nSELECT-NO\n  AMERICAN INDIAN OR ALASKA NATIVE\nSELECT-NO\n  ASIAN\nSELECT-NO\n  BLACK OR AFRICAN AMERICAN\nSELECT-NO\n  CAUCASIAN\nSELECT-NO\n  NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER\nSELECT-YES\n  WHITE\nSELECT-NO\n  MULTIPLE\nSELECT-NO\n  OTHER", "Collected Race: WHITE", updated_text)
    updated_text = re.sub(r"Collected Race   CRACE\nSELECT-NO\n  AMERICAN INDIAN OR ALASKA NATIVE\nSELECT-NO\n  ASIAN\nSELECT-NO\n  BLACK OR AFRICAN AMERICAN\nSELECT-NO\n  CAUCASIAN\nSELECT-NO\n  NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER\nSELECT-NO\n  WHITE\nSELECT-YES\n  MULTIPLE\nSELECT-NO\n  OTHER", "Collected Race: MULTIPLE", updated_text)
    updated_text = re.sub(r"Collected Race   CRACE\nSELECT-NO\n  AMERICAN INDIAN OR ALASKA NATIVE\nSELECT-NO\n  ASIAN\nSELECT-NO\n  BLACK OR AFRICAN AMERICAN\nSELECT-NO\n  CAUCASIAN\nSELECT-NO\n  NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER\nSELECT-NO\n  WHITE\nSELECT-NO\n  MULTIPLE\nSELECT-YES\n  OTHER", "Collected Race: OTHER", updated_text)


    updated_text = re.sub(r"Occupation   OCCUPATION", "Occupation:", updated_text)
    updated_text = re.sub(r"Race Other   RACEOTH", "Race Other:", updated_text)
    updated_text = re.sub(r"Other, Ethnicity   ETHOTHER", "Other Ethnicity:", updated_text)
    

    
    
    pattern = r"(Site):(\s+[A-Za-z0-9]+)"
    siteKey = extract_key_value(header, pattern)
    pattern = r"(Subject):(\s+[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)"
    subjectKey = extract_key_value(header, pattern)
    pattern = r"(Visit):(\s+[A-Za-z0-9]+_[A-Za-z0-9]+)"
    vistKey = extract_key_value(header, pattern)
    pattern = r"(Demographics Collection Date):(\s+(\d(\s+\d)+)\s+([A-Za-z]+(\s+[A-Za-z]+)+)\s+(\d(\s+\d)+))"
    demodateKey = extract_key_value(updated_text, pattern)
    updated_text = re.sub(r"Demographics Collection Date:\s+(\d(\s+\d)+)\s+([A-Za-z]+(\s+[A-Za-z]+)+)\s+(\d(\s+\d)+)", "Demographics Collection Date: " + demodateKey['Demographics Collection Date'].replace(" ", ""), updated_text)
    pattern = r"(Birth Date):(\s+(\d(\s+\d)+)\s+([A-Za-z]+(\s+[A-Za-z]+)+)\s+(\d(\s+\d)+))"
    birthdateKey = extract_key_value(updated_text, pattern)
    updated_text = re.sub(r"Birth Date:\s+(\d(\s+\d)+)\s+([A-Za-z]+(\s+[A-Za-z]+)+)\s+(\d(\s+\d)+)", "Birth Date: " + birthdateKey['Birth Date'].replace(" ", ""), updated_text)
    # pattern = r"(Age):(\s+[A-Za-z0-9]+)"
    # ageKey = extract_key_value(updated_text, pattern)
    # pattern = r"(Sex):(\s+[A-Za-z]+)"
    # sexKey = extract_key_value(updated_text, pattern)

    sitename = ''
    subjectname = ''
    vistname = ''
    if len(siteKey) > 0:
        sitename = siteKey['Site'].strip(' ')
    if len(subjectKey) > 0:
        subjectname = subjectKey['Subject'].strip(' ')
    if len(vistKey) > 0:
        vistname = vistKey['Visit'].strip(' ')
    
    return updated_text, sitename, subjectname, vistname
    
    # demodate= demodateKey['Demographics Collection Date'].replace(" ", "")
    # birthdate= birthdateKey['Birth Date'].replace(" ", "")
    # age = ageKey['Age'].strip(' ')
    # sex = sexKey['Sex'].strip(' ')

def SaveDemographicVectors(res: Demographic, sitename, subjectname, vistname, formname):
    documents = []
    fieldList = []
    metaindex = f"{sitename}-{subjectname}-{vistname}-{formname}"
    fieldList.append("'{}':'{}'".format("Demographics Collection Date", res.demographics_collection_date))
    fieldList.append("'{}':'{}'".format("Birth Date", res.birth_date))
    fieldList.append("'{}':'{}'".format("Age", res.age))
    fieldList.append("'{}':'{}'".format("Sex", res.sex))
    fieldList.append("'{}':'{}'".format("Collected Ethnicity", res.collected_ethnicity))
    fieldList.append("'{}':'{}'".format("Other Ethnicity", res.other_ethnicity))
    fieldList.append("'{}':'{}'".format("Collected Race", res.collected_race))
    fieldList.append("'{}':'{}'".format("Race Other", res.race_other))
    fieldList.append("'{}':'{}'".format("Occupation", res.occupation))
    fieldList.append("'{}':'{}'".format("site",sitename))
    fieldList.append("'{}':'{}'".format("subject",subjectname))
    fieldList.append("'{}':'{}'".format("visit",vistname))
    fieldList.append("'{}':'{}'".format("form",formname))
    jsoncontent = "{" + ','.join(fieldList) + "}"
    document_1 = Document(
        page_content=jsoncontent,
        metadata={"source": metaindex})
    documents.append(document_1)
    uuids = [str(uuid4()) for _ in range(len(documents))]

    #delete existing
    reqcontent = f"please find the values for site {sitename}, subject {subjectname} and visit {vistname} for {formname}"
    new_vector_store = FAISS.load_local(
    "galaxbiotech-vectorDB", embeddings, allow_dangerous_deserialization=True)
    content_raw = new_vector_store.similarity_search(   
                    reqcontent,
                    k=1,
                    filter={"source": {"$eq": metaindex}})
    if len(content_raw) > 0:
        idList = []
        idList.append(content_raw[0].id)
        RemoveVectors(new_vector_store, idList)
        #new_vector_store.delete(ids=content_raw[0].id)

    new_vector_store.add_documents(documents=documents, ids=uuids)
    new_vector_store.save_local("galaxbiotech-vectorDB")


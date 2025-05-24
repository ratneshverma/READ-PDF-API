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
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance
import time
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError


# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
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
    site: str = "None"
    subject: str = "None"
    visit: str = "None"

def IsPDFContentDigital(output_file):
    text = ''
    with pdfplumber.open(output_file) as pdf:
        for page in pdf.pages:
            text = text + page.extract_text()
    return True if text != '' else False

def ReadAWSOCRPDFContent(output_file):
    bucket_name = "chatbot-galax-bucket"
    file_name = output_file
    object_name = output_file
    try:
        object_name = object_name or file_name
        s3_client.upload_file(file_name, bucket_name, object_name)
        print(f"File '{file_name}' uploaded to bucket '{bucket_name}' as '{object_name}'.")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
    # Start document text detection
    response = textract_client.start_document_analysis(
        DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': file_name}},
            FeatureTypes=['FORMS']
    )

    # Get the Job ID
    job_id = response['JobId']
    print(f"Job started with ID: {job_id}")

    # Wait for the job to complete
    while True:
        status = textract_client.get_document_analysis(JobId=job_id)
        if status['JobStatus'] in ['SUCCEEDED', 'FAILED']:
            break
        print("Waiting for job to complete...")
        time.sleep(5)
    
    if status['JobStatus'] == 'SUCCEEDED':
        # Collect all pages
        pages = []
        next_token = None

        while True:
            if next_token:
                result = textract_client.get_document_analysis(JobId=job_id, NextToken=next_token)
            else:
                result = textract_client.get_document_analysis(JobId=job_id)

            pages.extend(result['Blocks'])

            next_token = result.get('NextToken')
            if not next_token:
                break
        # Extract text from the response
        text = ""
        for block in pages:
            if block['BlockType'] == 'LINE':
                text += block['Text'] + "\n"
        
        res = text.split('\n')
        resText = ' '.join(res)
        resText = resText.replace(",", " ")
        resText = resText.replace("|", "")
        return resText
    else:
        raise Exception("Text detection failed!")

def ReadOCRPDFContent(output_file):
    # images = convert_from_path(output_file, poppler_path=r"poppler\poppler-24.08.0\Library\bin", dpi=300)
    images = convert_from_path(output_file, poppler_path=r"/usr/bin/", dpi=300)
    enhanced_images = []
    image_file_list = []
    for img in images:
        # Enhance brightness
        # enhancer = ImageEnhance.Brightness(img)
        # img = enhancer.enhance(1.5)  # Increase brightness (1.0 = original)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.5)  # Increase contrast (1.0 = original)
        enhanced_images.append(img)
        

    # Save enhanced images
    for i, enhanced_img in enumerate(enhanced_images):
        fileName = f"./images/page_{i}.png"
        grayscale_img = enhanced_img.convert('L')
        grayscale_img.save(fileName)
        image_file_list.append(fileName)


    # for i, image in enumerate(images):
    #     fileName = f"./images/page_{i}.jpg"
    #     image.save(fileName, "JPEG")
    #     image_file_list.append(fileName)
    content = ''
    for i, fileImg in enumerate(image_file_list):
        #custom_config = r'--oem 3 --psm 6 -l eng'
        content = content + pytesseract.image_to_string(fileImg)
        print(content)
    res = content.split('\n')
    resText = ','.join(res)
    resText = resText.replace(",", " ")
    resText = resText.replace("|", "")
    return resText    

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
    finalContent = ''
    header = ''
    if IsPDFContentDigital(output_file):
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
            if request.site == "None" and sitename == '':
                return "Site value is not present in PDF"
            if request.site != "None":
                sitename = request.site
            if request.visit == "None" and vistname == '':
                return "Visit value is not present in PDF"
            if request.visit != "None":
                vistname = request.visit
            

            res = RAGDemographic(raw_file_data)
            
            democolDateStr = res.demographics_collection_date.strip().replace(" ","")
            if len(democolDateStr) >= 9:
                democolDateStr = democolDateStr[0:2] + '-' + democolDateStr[2:5] + '-' + democolDateStr[5:9]
            res.demographics_collection_date = democolDateStr

            birthDateStr = res.birth_date.strip().replace(" ","")
            if len(birthDateStr) >= 9:
                birthDateStr = birthDateStr[0:2] + '-' + birthDateStr[2:5] + '-' + birthDateStr[5:9]
            res.birth_date = birthDateStr

            SaveDemographicVectors(res, sitename, subjectname, vistname, formname)

            print(res.model_dump_json())
            return res.model_dump_json()
        else:
            return "Supported only Demographic PDF for time being"
    else:
        finalContent = ReadOCRPDFContent(output_file=output_file)
        finalAWSContent = ReadAWSOCRPDFContent(output_file=output_file)
        pattern = r"(Form):(\s+[A-Za-z]+)"
        formNameKey = extract_key_value(finalContent, pattern)
        formname= formNameKey['Form'].strip(' ')
        if formname == "Demographic":
            demo, sitename, subjectname, vistname  = CleanOCRDemographicData(finalContent, finalAWSContent)
            if request.subject == "None" and subjectname == '':
                return "Subject value is not present in PDF or in parameter"
            if request.subject != "None":
                subjectname = request.subject
            if request.site == "None" and sitename == '':
                return "Site value is not present in PDF"
            if request.site != "None":
                sitename = request.site
            if request.visit == "None" and vistname == '':
                return "Visit value is not present in PDF"
            if request.visit != "None":
                vistname = request.visit
            # res = RAGDemographic(raw_file_data)
            SaveDemographicVectors(demo, sitename, subjectname, vistname, formname)
            
            print(demo.model_dump_json())
            return demo.model_dump_json()
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

def find_all_occurrences(text, substring):
    return [match.start() for match in re.finditer(re.escape(substring), text)]

def CleanOCRDemographicData(Content, ContentAWS):
    demo = Demographic(demographics_collection_date="None", birth_date="None", age="None", sex="None", collected_ethnicity="None", if_mixed_ethnicity="None", other_ethnicity="None", collected_race="None", multiple_race_if_any="None", race_other="None", occupation="None")
    
    pattern = r"(Site):(\s+[A-Za-z0-9]+)"
    siteKey = extract_key_value(Content, pattern)
    pattern = r"(Subject):(\s+[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)"
    subjectKey = extract_key_value(Content, pattern)
    pattern = r"(Visit):(\s+[A-Za-z0-9]+_[A-Za-z0-9]+)"
    vistKey = extract_key_value(Content, pattern)

    sitename = ''
    subjectname = ''
    vistname = ''
    if len(siteKey) > 0:
        sitename = siteKey['Site'].strip(' ')
    if len(subjectKey) > 0:
        subjectname = subjectKey['Subject'].strip(' ')
    if len(vistKey) > 0:
        vistname = vistKey['Visit'].strip(' ')
    
    

    
    strIndex =Content.index("Demographics Collection Date")
    Content = Content[strIndex:]
    Content = Content.replace(";", "")
    conIndex = Content.index("Confidential")
    firstStr = Content[0:conIndex]
    thuIndex = find_all_occurrences(Content, "Therapeutics")
    secondStr = Content[thuIndex[1]:]
    secondStr = secondStr.replace("Therapeutics", "")
    Content = firstStr + secondStr

    strIndex =ContentAWS.index("Demographics Collection Date")
    ContentAWS = ContentAWS[strIndex:]
    ContentAWS = ContentAWS.replace(";", "")
    conIndex = ContentAWS.index("Confidential")
    firstStr = ContentAWS[0:conIndex]
    thuIndex = find_all_occurrences(ContentAWS, "Therapeutics")
    secondStr = ContentAWS[thuIndex[1]:]
    secondStr = secondStr.replace("Therapeutics", "")
    ContentAWS = firstStr + secondStr

    Content = Content.replace("©", "@")
    Content = Content.replace("®", "@")
    Content = Content.replace("@@", "@")
    pattern = f"[^a-zA-Z0-9{re.escape("@")} ]"
    Content = re.sub(pattern, "", Content)
    Content = Content.replace(";", "")
    Content = Content.replace(":", "")
    Content = Content.replace("    ", " ")
    Content = Content.replace("   ", " ")
    Content = Content.replace("  ", " ")
    Content = Content.replace("This is an electronically authenticated report generated by Galax CTMS System", "")
    Content = Content.replace("Confidential", "")
    Content = Content.replace("Page 1 of2", "")
    Content = Content.replace("100 DRAFT", "")
    Content = Content.replace("Page 20f2", "")
    Content = Content.replace("Page 2o0f2", "")
    Content = Content.replace("Page 2 of2", "")
    Content = Content.replace("Page 1 of 2", "")
    Content = Content.replace("Page 2 of 2", "")

    ContentAWS = ContentAWS.replace(" I ", " ")
    ContentAWS = ContentAWS.replace(";", "")
    ContentAWS = ContentAWS.replace(":", "")
    ContentAWS = ContentAWS.replace("    ", " ")
    ContentAWS = ContentAWS.replace("   ", " ")
    ContentAWS = ContentAWS.replace("  ", " ")
    ContentAWS = ContentAWS.replace("This is an electronically authenticated report generated by Galax CTMS System", "")
    ContentAWS = ContentAWS.replace("Confidential", "")
    ContentAWS = ContentAWS.replace("Page 1 of2", "")
    ContentAWS = ContentAWS.replace("1.00 DRAFT", "")
    ContentAWS = ContentAWS.replace("Page 20f2", "")
    ContentAWS = ContentAWS.replace("Page 2o0f2", "")
    ContentAWS = ContentAWS.replace("Page 2 of2", "")
    ContentAWS = ContentAWS.replace("Page 1 of 2", "")
    ContentAWS = ContentAWS.replace("Page 2 of 2", "")
    
    
    democolDateStr = ContentAWS[ContentAWS.index("DMDATY")+6:ContentAWS.index("Date of Birth")].strip().replace(" ","")
    if len(democolDateStr) >= 9:
        democolDateStr = democolDateStr[0:2] + '-' + democolDateStr[2:5] + '-' + democolDateStr[5:9]
    demo.demographics_collection_date = democolDateStr
    
    birthDateStr = ContentAWS[ContentAWS.index("BRTHYY")+6:ContentAWS.index("Age")].strip().replace(" ","")
    if len(birthDateStr) >= 9:
        birthDateStr = birthDateStr[0:2] + '-' + birthDateStr[2:5] + '-' + birthDateStr[5:9]
    demo.birth_date = birthDateStr

    demo.age = ContentAWS[ContentAWS.index("AGE")+3:ContentAWS.index("Age Units")].strip()
    
    if "Sex SEX Male @ Female" in Content:
        demo.sex = "Female"
    else:
        demo.sex = "Male"
    
    filterRaceStr = Content[Content.index("Race RACE")+9:Content.index("Multiple Race if any")].strip()
    filterRaceStr = filterRaceStr.lower()
    if "@ american indian or alaska native" in filterRaceStr:
        demo.collected_race = "American indian or alaska native"
    elif "@ asian" in filterRaceStr:
        demo.collected_race = "Asian"
    elif "@ black or african american" in filterRaceStr:
        demo.collected_race = "Black or african american"
    elif "@ caucasian" in filterRaceStr:
        demo.collected_race = "Caucasian"
    elif "@ native hawaiian or other pacific islander" in filterRaceStr:
        demo.collected_race = "Native hawaiian or other pacific islander"
    elif "@ white" in filterRaceStr:
        demo.collected_race = "White"
    elif "@ multiple race" in filterRaceStr:
        demo.collected_race = "Multiple race"
    elif "@ other" in filterRaceStr:
        demo.collected_race = "Other"
    else:
        demo.collected_race = "None"

    demo.multiple_race_if_any =  ContentAWS[ContentAWS.index("RACEMULT")+8:ContentAWS.index("Race Other RACEOTH")].strip()
    demo.race_other = ContentAWS[ContentAWS.index("RACEOTH")+7:ContentAWS.index("Ethnicity ETHNIC")].strip()
    filterethniStr = Content[Content.index("Ethnicity ETHNIC")+16:Content.index("If Mixed ethnicity")].strip()
    filterethniStr = filterethniStr.lower()
    if "@ hispanic or latino" in filterethniStr:
        demo.collected_ethnicity = "Hispanic or latino"
    elif "@ not hispanic or latino" in filterethniStr:
        demo.collected_ethnicity = "Not hispanic or latino"
    elif "@ middle easterner" in filterethniStr:
        demo.collected_ethnicity = "Middle easterner"
    elif "@ chinese" in filterethniStr:
        demo.collected_ethnicity = "Chinese"
    elif "@ indian (indian subcontinent)" in filterethniStr:
        demo.collected_ethnicity = "Indian (Indian subcontinent)"
    elif "@ japanese" in filterethniStr:
        demo.collected_ethnicity = "Japanese"
    elif "@ mixed ethnicity" in filterethniStr:
        demo.collected_ethnicity = "Mixed ethnicity"
    elif "@ unknown" in filterethniStr:
        demo.collected_ethnicity = "Unknown"
    elif "@ other" in filterethniStr:
        demo.collected_ethnicity = "Other"
    else:
        demo.collected_ethnicity = "None"
    
    demo.if_mixed_ethnicity = ContentAWS[ContentAWS.index("ETHMIXED")+8:ContentAWS.index("Other Ethnicity")].strip()
    demo.other_ethnicity = ContentAWS[ContentAWS.index("ETHOTHER")+8:].strip()
    return demo, sitename, subjectname, vistname

def CleanDemographicData(finalContent, header):
    finalContent = finalContent.split('\n')
    finalContent = ','.join(finalContent)
    finalContent = finalContent.replace("|", " ")
    finalContent = finalContent.replace("/", "")
    finalContent = finalContent.replace("(", "")
    finalContent = finalContent.replace(")", "")
    finalContent = finalContent.replace("\xa0", "")
    finalContent = finalContent.replace(",", " ")
    finalContent = finalContent.strip()
    finalContent = finalContent.replace("  ", " ")
    finalContent = finalContent.replace("  ", " ")
    finalContent = finalContent.replace("AGEU Age Units SELECT-YES YEARS", "")

    finalContent = re.sub(r"Demographics Collection Date DMDAT Day DMDATDMonth DMDATM Year DMDATY", "Demographics Collection Date:", finalContent)
    finalContent = re.sub(r"Birth Date BRTHDAT Birth Day BRTHDDBirth Month BRTHMOBirth Year BRTHYY", "Birth Date:", finalContent)
    finalContent = re.sub(r"Age AGE", "Age:", finalContent)
    finalContent = re.sub(r"Sex SEX SELECT-NO Male SELECT-YES Female", "Sex: Female", finalContent)
    finalContent = re.sub(r"Sex SEX SELECT-YES Male SELECT-NO Female", "Sex: Male", finalContent)
    
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-NO HISPANIC OR LATINO SELECT-NO NOT HISPANIC OR LATINO SELECT-NO MIDDLE EASTERNER SELECT-NO INDIAN INDIAN SUBCONTINENT SELECT-NO JAPANESE SELECT-NO MIXED ETHNICITY SELECT-NO NOT REPORTED SELECT-NO OTHER SELECT-NO UNKNOWN", "Collected Ethnicity: None", finalContent)
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-YES HISPANIC OR LATINO SELECT-NO NOT HISPANIC OR LATINO SELECT-NO MIDDLE EASTERNER SELECT-NO INDIAN INDIAN SUBCONTINENT SELECT-NO JAPANESE SELECT-NO MIXED ETHNICITY SELECT-NO NOT REPORTED SELECT-NO OTHER SELECT-NO UNKNOWN", "Collected Ethnicity: HISPANIC OR LATINO", finalContent)
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-NO HISPANIC OR LATINO SELECT-YES NOT HISPANIC OR LATINO SELECT-NO MIDDLE EASTERNER SELECT-NO INDIAN INDIAN SUBCONTINENT SELECT-NO JAPANESE SELECT-NO MIXED ETHNICITY SELECT-NO NOT REPORTED SELECT-NO OTHER SELECT-NO UNKNOWN", "Collected Ethnicity: NOT HISPANIC OR LATINO", finalContent)
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-NO HISPANIC OR LATINO SELECT-NO NOT HISPANIC OR LATINO SELECT-YES MIDDLE EASTERNER SELECT-NO INDIAN INDIAN SUBCONTINENT SELECT-NO JAPANESE SELECT-NO MIXED ETHNICITY SELECT-NO NOT REPORTED SELECT-NO OTHER SELECT-NO UNKNOWN", "Collected Ethnicity: MIDDLE EASTERNER", finalContent)
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-NO HISPANIC OR LATINO SELECT-NO NOT HISPANIC OR LATINO SELECT-NO MIDDLE EASTERNER SELECT-YES INDIAN INDIAN SUBCONTINENT SELECT-NO JAPANESE SELECT-NO MIXED ETHNICITY SELECT-NO NOT REPORTED SELECT-NO OTHER SELECT-NO UNKNOWN", "Collected Ethnicity: INDIAN (INDIAN SUBCONTINENT)", finalContent)
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-NO HISPANIC OR LATINO SELECT-NO NOT HISPANIC OR LATINO SELECT-NO MIDDLE EASTERNER SELECT-NO INDIAN INDIAN SUBCONTINENT SELECT-YES JAPANESE SELECT-NO MIXED ETHNICITY SELECT-NO NOT REPORTED SELECT-NO OTHER SELECT-NO UNKNOWN", "Collected Ethnicity: JAPANESE", finalContent)
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-NO HISPANIC OR LATINO SELECT-NO NOT HISPANIC OR LATINO SELECT-NO MIDDLE EASTERNER SELECT-NO INDIAN INDIAN SUBCONTINENT SELECT-NO JAPANESE SELECT-YES MIXED ETHNICITY SELECT-NO NOT REPORTED SELECT-NO OTHER SELECT-NO UNKNOWN", "Collected Ethnicity: MIXED ETHNICITY", finalContent)
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-NO HISPANIC OR LATINO SELECT-NO NOT HISPANIC OR LATINO SELECT-NO MIDDLE EASTERNER SELECT-NO INDIAN INDIAN SUBCONTINENT SELECT-NO JAPANESE SELECT-NO MIXED ETHNICITY SELECT-YES NOT REPORTED SELECT-NO OTHER SELECT-NO UNKNOWN", "Collected Ethnicity: NOT REPORTED", finalContent)
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-NO HISPANIC OR LATINO SELECT-NO NOT HISPANIC OR LATINO SELECT-NO MIDDLE EASTERNER SELECT-NO INDIAN INDIAN SUBCONTINENT SELECT-NO JAPANESE SELECT-NO MIXED ETHNICITY SELECT-NO NOT REPORTED SELECT-YES OTHER SELECT-NO UNKNOWN", "Collected Ethnicity: OTHER", finalContent)
    finalContent = re.sub(r"Collected Ethnicity CETHNIC SELECT-NO HISPANIC OR LATINO SELECT-NO NOT HISPANIC OR LATINO SELECT-NO MIDDLE EASTERNER SELECT-NO INDIAN INDIAN SUBCONTINENT SELECT-NO JAPANESE SELECT-NO MIXED ETHNICITY SELECT-NO NOT REPORTED SELECT-NO OTHER SELECT-YES UNKNOWN", "Collected Ethnicity: UNKNOWN", finalContent)

    finalContent = re.sub(r"Collected Race CRACE SELECT-NO AMERICAN INDIAN OR ALASKA NATIVE SELECT-NO ASIAN SELECT-NO BLACK OR AFRICAN AMERICAN SELECT-NO CAUCASIAN SELECT-NO NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER SELECT-NO WHITE SELECT-NO MULTIPLE SELECT-NO OTHER", "Collected Race: None", finalContent)
    finalContent = re.sub(r"Collected Race CRACE SELECT-YES AMERICAN INDIAN OR ALASKA NATIVE SELECT-NO ASIAN SELECT-NO BLACK OR AFRICAN AMERICAN SELECT-NO CAUCASIAN SELECT-NO NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER SELECT-NO WHITE SELECT-NO MULTIPLE SELECT-NO OTHER", "Collected Race: AMERICAN INDIAN OR ALASKA NATIVE", finalContent)
    finalContent = re.sub(r"Collected Race CRACE SELECT-NO AMERICAN INDIAN OR ALASKA NATIVE SELECT-YES ASIAN SELECT-NO BLACK OR AFRICAN AMERICAN SELECT-NO CAUCASIAN SELECT-NO NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER SELECT-NO WHITE SELECT-NO MULTIPLE SELECT-NO OTHER", "Collected Race: ASIAN", finalContent)
    finalContent = re.sub(r"Collected Race CRACE SELECT-NO AMERICAN INDIAN OR ALASKA NATIVE SELECT-NO ASIAN SELECT-YES BLACK OR AFRICAN AMERICAN SELECT-NO CAUCASIAN SELECT-NO NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER SELECT-NO WHITE SELECT-NO MULTIPLE SELECT-NO OTHER", "Collected Race: BLACK OR AFRICAN AMERICAN", finalContent)
    finalContent = re.sub(r"Collected Race CRACE SELECT-NO AMERICAN INDIAN OR ALASKA NATIVE SELECT-NO ASIAN SELECT-NO BLACK OR AFRICAN AMERICAN SELECT-YES CAUCASIAN SELECT-NO NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER SELECT-NO WHITE SELECT-NO MULTIPLE SELECT-NO OTHER", "Collected Race: CAUCASIAN", finalContent)
    finalContent = re.sub(r"Collected Race CRACE SELECT-NO AMERICAN INDIAN OR ALASKA NATIVE SELECT-NO ASIAN SELECT-NO BLACK OR AFRICAN AMERICAN SELECT-NO CAUCASIAN SELECT-YES NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER SELECT-NO WHITE SELECT-NO MULTIPLE SELECT-NO OTHER", "Collected Race: NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER", finalContent)
    finalContent = re.sub(r"Collected Race CRACE SELECT-NO AMERICAN INDIAN OR ALASKA NATIVE SELECT-NO ASIAN SELECT-NO BLACK OR AFRICAN AMERICAN SELECT-NO CAUCASIAN SELECT-NO NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER SELECT-YES WHITE SELECT-NO MULTIPLE SELECT-NO OTHER", "Collected Race: WHITE", finalContent)
    finalContent = re.sub(r"Collected Race CRACE SELECT-NO AMERICAN INDIAN OR ALASKA NATIVE SELECT-NO ASIAN SELECT-NO BLACK OR AFRICAN AMERICAN SELECT-NO CAUCASIAN SELECT-NO NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER SELECT-NO WHITE SELECT-YES MULTIPLE SELECT-NO OTHER", "Collected Race: MULTIPLE", finalContent)
    finalContent = re.sub(r"Collected Race CRACE SELECT-NO AMERICAN INDIAN OR ALASKA NATIVE SELECT-NO ASIAN SELECT-NO BLACK OR AFRICAN AMERICAN SELECT-NO CAUCASIAN SELECT-NO NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER SELECT-NO WHITE SELECT-NO MULTIPLE SELECT-YES OTHER", "Collected Race: OTHER", finalContent)
    
    finalContent = re.sub(r"Occupation OCCUPATION", "Occupation:", finalContent)
    finalContent = re.sub(r"Race Other RACEOTH", "Race Other:", finalContent)
    finalContent = re.sub(r"Other Ethnicity ETHOTHER", "Other Ethnicity:", finalContent)
    
    pattern = r"(Site):(\s+[A-Za-z0-9]+)"
    siteKey = extract_key_value(header, pattern)
    pattern = r"(Subject):(\s+[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+)"
    subjectKey = extract_key_value(header, pattern)
    pattern = r"(Visit):(\s+[A-Za-z0-9]+_[A-Za-z0-9]+)"
    vistKey = extract_key_value(header, pattern)
    # pattern = r"(Demographics Collection Date):(\s+(\d(\s+\d)+)\s+([A-Za-z]+(\s+[A-Za-z]+)+)\s+(\d(\s+\d)+))"
    # demodateKey = extract_key_value(updated_text, pattern)
    # updated_text = re.sub(r"Demographics Collection Date:\s+(\d(\s+\d)+)\s+([A-Za-z]+(\s+[A-Za-z]+)+)\s+(\d(\s+\d)+)", "Demographics Collection Date: " + demodateKey['Demographics Collection Date'].replace(" ", ""), updated_text)
    # pattern = r"(Birth Date):(\s+(\d(\s+\d)+)\s+([A-Za-z]+(\s+[A-Za-z]+)+)\s+(\d(\s+\d)+))"
    # birthdateKey = extract_key_value(updated_text, pattern)
    # updated_text = re.sub(r"Birth Date:\s+(\d(\s+\d)+)\s+([A-Za-z]+(\s+[A-Za-z]+)+)\s+(\d(\s+\d)+)", "Birth Date: " + birthdateKey['Birth Date'].replace(" ", ""), updated_text)
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
    
    return finalContent, sitename, subjectname, vistname

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


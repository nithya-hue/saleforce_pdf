
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import fitz
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

allowed_origins = [ "*" ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  
    allow_credentials=True,         
    allow_methods=["*"],            
    allow_headers=["*"])

apikey = None
store = {}
class StringRequestModel(BaseModel):
    string1: str
    string2: str

@app.post("/store_strings")
def store_strings(request: StringRequestModel):

    global apikey
 
    apikey  = request.string1
    store['string2'] = request.string2

    return {"message": "Strings stored successfully"}


class PDF(BaseModel):
    filename: str

def extract_text_pymupdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            extracted = page.get_text()
            text += extracted
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def process_text(text):
    #load_dotenv()    
    #OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
    #print(OPENAI_API_KEY )
    global apikey
    if not apikey:
        raise HTTPException(status_code=500, detail="OpenAI API key not found")
    OPENAI_API_KEY = apikey
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(text_chunks):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    
    def get_conversation_chain(vectorstore):
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory   )
        return conversation_chain

    # Process text
    text_chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)

    # Hardcoded query
    #query = "what is survey plan"
    #docs = vectorstore.similarity_search(query)
    #response_text = docs[0].page_content  # Assuming you want the content of the most similar document
    #return response_text
    user_input1="what is survey plan?   "
    response = conversation_chain({'question': user_input1})
    #print("Response:")
    #print(response['answer'])
    response_text= user_input1+"  "+response['answer']
    user_input2="what is  fire safety measures   "
    response = conversation_chain({'question': user_input2})
    response_text= response_text+" " +user_input2+"  "+response['answer']
    return response_text



@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())       
        extracted_text = extract_text_pymupdf(file.filename)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from the PDF.")       
        response = process_text(extracted_text)    
        os.remove(file.filename)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

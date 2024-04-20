from llamaapi import LlamaAPI
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.llms import ChatLlamaAPI
from langchain.memory import ChatMessageHistory
from datasets import load_dataset

# Replace 'Your_API_Token' with your actual API token
llama = LlamaAPI("LL-L2JQThjl0h8dB9BSjHJPwJDEMYRY8vWGCJOMYhhVQqukwqELfsFl7K9HEZC0vJDb")

#Initialize Llama API
model = ChatLlamaAPI(client=llama)

#Load medical dialog dataset
dataset = load_dataset("medical_dialog", 'processed.en', trust_remote_code=True)

#Get patient data from dataset
chunks = dataset['train']['utterances']

patient_data = []
for i in range(len(chunks)):
  patient_utterance = chunks[i][0]
  patient_data.append(patient_utterance)

#Embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
patient_data_embeddings = embedding_model.encode(patient_data)

#Chroma DB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="rag_llama2")

collection.add(
    embeddings = patient_data_embeddings,
    documents=patient_data,
    ids= [str(i) for i in range(len(patient_data))]
)

def retrieve_vector_db(query, n_results=5):
    results = collection.query(
    query_embeddings = embedding_model.encode(query).tolist(),
    n_results=n_results
    )
    return results['ids']


def rag_function(query):
    retrieved_results = retrieve_vector_db(query)
    
    patient1 = dataset['train']['utterances'][int(retrieved_results[0][0])][0]
    doctor1 = dataset['train']['utterances'][int(retrieved_results[0][0])][1]

    patient2 = dataset['train']['utterances'][int(retrieved_results[0][1])][0]
    doctor2 = dataset['train']['utterances'][int(retrieved_results[0][1])][1]

    patient3 = dataset['train']['utterances'][int(retrieved_results[0][2])][0]
    doctor3 = dataset['train']['utterances'][int(retrieved_results[0][2])][1]

    patient4 = dataset['train']['utterances'][int(retrieved_results[0][3])][0]
    doctor4 = dataset['train']['utterances'][int(retrieved_results[0][3])][1]

    patient5 = dataset['train']['utterances'][int(retrieved_results[0][4])][0]
    doctor5 = dataset['train']['utterances'][int(retrieved_results[0][4])][1]
    
    few_shot_string = f"{patient1} {doctor1} --- {patient2} {doctor2} --- {patient3} {doctor3} --- {patient4} {doctor4} --- {patient5} {doctor5} --- patient: {query} doctor:"
    
    return few_shot_string
    


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced doctor specializing in General Medicine. 
            Always give analogies and explain medical terminology. 
            You diagnose patients based on their symptoms. 
            Ask the patient follow-up questions one at a time. 
            Finally, Hypothesize a problem and suggest tests to confirm the hypothesized diagnosis.
            Do not generate the user side of the conversation under any circumstances.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

ai="Hello! How can I help you?"
print("Here is a doctor to interact with.....")
#inp="[INST]Hello Doctor![/INST]"


history = ChatMessageHistory()
#history.add_user_message(inp)
history.add_ai_message(ai)
#print(ai)
while True:
    uinput=input("Patient: \n")
    inp1="[INST]"+uinput+"[/INST]"
    if(uinput.lower()=="exit" or uinput.lower()=="Thank you"):
        break
    else:
        few_shot_string = rag_function(inp1)
        history.add_user_message(few_shot_string)
        out=chain.invoke({"messages":history.messages}).content
        history.add_ai_message(out)
        print("Doctor:"+out)
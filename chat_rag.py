from llamaapi import LlamaAPI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.llms import ChatLlamaAPI
from langchain.memory import ChatMessageHistory
from datasets import load_dataset
import streamlit as st


llama = LlamaAPI("LL-L2JQThjl0h8dB9BSjHJPwJDEMYRY8vWGCJOMYhhVQqukwqELfsFl7K9HEZC0vJDb")


model = ChatLlamaAPI(client=llama)


dataset = load_dataset("medical_dialog", 'processed.en', trust_remote_code=True)


chunks = dataset['train']['utterances']

patient_data = []
for i in range(len(chunks)):
  patient_utterance = chunks[i][0]
  patient_data.append(patient_utterance)

#Embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
patient_data_embeddings = embedding_model.encode(patient_data)

# chroma_client = chromadb.HttpClient(host='127.0.0.1', port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
chroma_client = chromadb.Client()
collection_name = "rag_llama7"


try:
    existing_collection = chroma_client.get_or_create_collection(name=collection_name)
except chromadb.errors.CollectionNotFound:
    print(f"Collection '{collection_name}' not found. Creating new.")

# Create a new collection
existing_collection.add(embeddings=patient_data_embeddings, documents=patient_data, ids=[str(i) for i in range(len(patient_data))])
print("Added embeddings to new collection.")

def retrieve_vector_db(query, n_results=5):
    results = existing_collection.query(
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
    
    few_shot_string = f"[INST]Answer the patient's last question, based on these examples: {patient1} {doctor1} --- {patient2} {doctor2} --- {patient3} {doctor3} --- {patient4} {doctor4} --- {patient5} {doctor5}[/INST] patient: {query} doctor:"
    
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

# Streamlit interaction
# pip install streamlit
# You can run the Streamlit code by doing this: streamlit run <python file name>

st.title('RAG-Enhanced Medical Diagnosis Chatbot')
st.write('Welcome to the RAG-enhanced medical diagnosis chatbot. Please enter your symptoms or questions below.')

if 'history' not in st.session_state:
    st.session_state['history'] = ChatMessageHistory()

with st.form("user_input_form"):
    user_input = st.text_input("Type your message and press enter:", key="user_input")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    st.session_state.history.add_user_message(HumanMessage(content=f"You: {user_input}"))
    
    context = " --- ".join([msg.content for msg in st.session_state.history.messages])
    enriched_prompt = rag_function(context + user_input)
    response = model.invoke([HumanMessage(content=enriched_prompt)]).content

    final_response = response.split('doctor:')[-1].strip()
    st.session_state.history.add_user_message(HumanMessage(content=f"Doctor: {final_response}"))

conversation = ""
for message in st.session_state.history.messages:
    conversation += message.content + "\n"

st.text_area("Conversation", value=conversation, height=300, disabled=True)




# Terminal Based Interaction

# chain = prompt | model

# ai="Hello! How can I help you?"
# print("Here is a doctor to interact with.....")
# inp="[INST]Hello Doctor![/INST]"

# history = ChatMessageHistory()
# #history.add_user_message(inp)
# history.add_ai_message(ai)
# #print(ai)
# while True:
#     uinput=input("Patient: \n")
#     inp1="[INST]"+uinput+"[/INST]"
#     if(uinput.lower()=="exit" or uinput.lower()=="Thank you"):
#         break
#     else:
#         few_shot_string = rag_function(inp1)
#         history.add_user_message(few_shot_string)
#         out=chain.invoke({"messages":history.messages}).content
#         history.add_ai_message(out)
#         print("Doctor:"+out)
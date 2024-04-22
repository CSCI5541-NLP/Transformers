import streamlit as st
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory


llama = LlamaAPI("LL-VsB2ZNB7Upm9qXR15QkZxoZbn1dCG5GG0KGpvdX8MLlpaA7Li3aFb2Uhr7X3JRoT")
model = ChatLlamaAPI(client=llama)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced doctor specializing in General Medicine.
            Always give analogies and explain medical terminology.
            You diagnose patients based on their symptoms.
            Ask the patient follow-up questions one at a time.
            Finally, hypothesize a problem and suggest tests to confirm the diagnosis.
            Do not generate the user side of the conversation under any circumstances.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model


st.title('Medical Diagnosis Chatbot')
st.write('Welcome to the medical diagnosis chatbot. Please enter your symptoms or questions below.')


if 'history' not in st.session_state:
    st.session_state['history'] = ChatMessageHistory()

with st.form("user_input_form"):
    user_input = st.text_input("Type your message and press enter:", key="user_input")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:

    formatted_input = f"User: {user_input}"
    st.session_state['history'].add_user_message(HumanMessage(content=formatted_input))
    

    ai_response = chain.invoke({"messages": st.session_state['history'].messages}).content
    formatted_response = f"Doctor: {ai_response}"
    st.session_state['history'].add_user_message(HumanMessage(content=formatted_response))


conversation = ""
for message in st.session_state['history'].messages:
    prefix, content = message.content.split(": ", 1)
    if prefix == "User":
        conversation += "You: " + content + "\n"
    else:
        conversation += "Doctor: " + content + "\n"

st.text_area("Conversation", value=conversation, height=300, disabled=True)

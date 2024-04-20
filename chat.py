from llamaapi import LlamaAPI

# Replace 'Your_API_Token' with your actual API token
llama = LlamaAPI("LL-VsB2ZNB7Upm9qXR15QkZxoZbn1dCG5GG0KGpvdX8MLlpaA7Li3aFb2Uhr7X3JRoT")
from langchain_experimental.llms import ChatLlamaAPI
model = ChatLlamaAPI(client=llama)
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

ai="Hello! May I know your name and your age?"
print("Here is a doctor to interact with.....")
inp="[INST]Hello Doctor![/INST]"

from langchain.memory import ChatMessageHistory
history = ChatMessageHistory()
history.add_user_message(inp)
history.add_ai_message(ai)
print(ai)
while True:
    uinput=input("Human: \n")
    inp1="[INST]"+uinput+"[/INST]"
    if(uinput.lower()=="exit" or uinput.lower()=="Thank you"):
        break
    else:
        history.add_user_message(inp1)
        out=chain.invoke({"messages":history.messages}).content
        history.add_ai_message(out)
        print("Doctor:"+out)
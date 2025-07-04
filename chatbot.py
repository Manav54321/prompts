from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Load ChatModel from Hugging Face
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

# Initialize the chat model
model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
    user_input = input("You:")
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print(f"AI: {result.content}")

print(f"The entire chat history is: {chat_history}")
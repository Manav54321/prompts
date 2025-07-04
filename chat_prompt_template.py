from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_prompt = ChatPromptTemplate([
    ('system', 'You are a helpful assistant in the field of {domain}.'),
    ('human', 'Discuss the topic of {topic} in detail.'),
])

prompt = chat_prompt.invoke({'domain': "AI", 'topic': "machine learning"})

print(prompt)
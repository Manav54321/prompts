# Dynamic Prompting with Hugging Face LLMs
# with f-string for variable injection

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Load LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

# Your dynamic inputs
paper_input = "Attention Is All You Need"
style_input = "Beginner-Friendly"
length_input = "Medium (3-5 paragraphs)"

# Create Chat Model
model = ChatHuggingFace(llm=llm)

# Use f-string to inject variables
prompt = f"""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  

2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""

# Invoke the model
result = model.invoke(prompt)

# Print the result
print(result.content if hasattr(result, 'content') else result)

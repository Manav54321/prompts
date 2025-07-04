from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import load_prompt
import gradio as gr

# Load environment variables
load_dotenv()

# Load LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

# Create Chat Model
model = ChatHuggingFace(llm=llm)

# Load your prompt template
template = load_prompt("template.json")

# Define function to summarize
def summarize(paper_input, style_input, length_input):
    chain = template | model
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    return result.content if hasattr(result, 'content') else result

# Dropdown options
paper_options = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
]

style_options = ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
length_options = ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]

# Gradio UI
demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Dropdown(choices=paper_options, label="Select Research Paper Name"),
        gr.Dropdown(choices=style_options, label="Select Explanation Style"),
        gr.Dropdown(choices=length_options, label="Select Explanation Length")
    ],
    outputs=gr.Textbox(label="Summary Output"),
    title="Research Paper",
    description="Select a paper, choose your preferred style and length, and get a summary powered by LLaMA-3.1!"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()

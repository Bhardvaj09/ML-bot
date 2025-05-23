import os 
import openai
import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.agents import Tool
from langchain import hub

# Set OpenAI API key
openai.api_key = "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"

# Set environment variable for LangChain
os.environ["OPENAI_API_KEY"] = "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"

# Create ML prompt template
ml_prompt = PromptTemplate(
    input_variables=["task_description"],  # Fixed: should be a list of strings
    template=("You are an AI code assistant who is capable of writing complex code for topics related to AI, Machine learning, deep learning, and NLP tasks. "
              "You should be able to write optimized code for the given task description. "
              "Include all the necessary libraries needed for the task description.\n\n"
              "Task: {task_description}\n\n"
              "Code:")
)

# Initialize LLM
llm = OpenAI(temperature=0.7)  # Fixed: Use OpenAI class, not openai function

# Create Tools for Code Generation and Explanation
def generate_code(task_description):
    chain = LLMChain(prompt=ml_prompt, llm=llm)
    return chain.run(task_description)

def explain_code(code_snippet):
    explain_prompt = PromptTemplate(
        input_variables=["code_snippet"],
        template="Explain the following code in detail:\n\n{code_snippet}\n\nExplanation:"
    )
    chain = LLMChain(prompt=explain_prompt, llm=llm)
    return chain.run(code_snippet)

# Define Tools for LangChain Agent using Tool class
code_generation_tool = Tool(
    name="CodeGenerator",
    func=generate_code,
    description="Generates machine learning, deep learning, NLP, and AI code based on a task description."
)

explanation_tool = Tool(
    name="CodeExplainer", 
    func=explain_code,
    description="Provides detailed explanations for code snippets."
)

# Create LangChain Agent
try:
    # Get a chat prompt for the agent
    prompt = hub.pull("hwchase17/react")
    
    # Create agent
    agent = create_react_agent(llm, [code_generation_tool, explanation_tool], prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[code_generation_tool, explanation_tool],
        verbose=True,
        handle_parsing_errors=True
    )
except Exception as e:
    st.error(f"Error creating agent: {e}")
    agent_executor = None

# Streamlit UI
st.title("AI and ML Code Assistant")
st.write(
    "This assistant can help generate and explain code for various tasks in machine learning, "
    "deep learning, NLP, and general AI development."
)

# User Input for Task Description
task_description = st.text_input("Enter your task description:", "")

if st.button("Generate Code"):
    if task_description:
        with st.spinner("Generating code..."):
            try:
                code = generate_code(task_description)
                st.subheader("Generated Code:")
                st.code(code, language="python")
                
                # Explanation Section
                explain_choice = st.checkbox("Explain the generated code")
                if explain_choice:
                    with st.spinner("Explaining code..."):
                        explanation = explain_code(code)
                        st.subheader("Code Explanation:")
                        st.write(explanation)
            except Exception as e:
                st.error(f"Error generating code: {e}")
    else:
        st.error("Please provide a task description to generate code.")

# Alternative: Use agent executor if available
st.subheader("Or use the AI Agent:")
agent_query = st.text_input("Ask the agent anything about code generation or explanation:", "")

if st.button("Ask Agent") and agent_executor:
    if agent_query:
        with st.spinner("Processing your request..."):
            try:
                response = agent_executor.run(agent_query)
                st.write("Agent Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Agent error: {e}")
    else:
        st.error("Please provide a query for the agent.")

# Sidebar for recent queries or saved code templates
st.sidebar.title("Recent Queries")
st.sidebar.write("Here you can find your recent queries and generated code for quick access.")

# Add some example tasks in sidebar
st.sidebar.subheader("Example Tasks:")
st.sidebar.write("• Create a linear regression model with scikit-learn")
st.sidebar.write("• Build a CNN for image classification with TensorFlow")
st.sidebar.write("• Implement text preprocessing for NLP")
st.sidebar.write("• Create a decision tree classifier")
st.sidebar.write("• Build a simple neural network with PyTorch")

import os 
import openai
import streamlit as st
from langchain import openai,LLMChain
from langchain.prompts import PromptTemplate
from langchain import AgentExecutor , create_sql_agent
from langchain.tools import tool 

openai.api_key = "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"

ml_prompt = PromptTemplate(
    input_variables=[task_description],
    template=("you are an ai code assistant who is capable of writing complex code and topics related to topics like AI,Machine learning,deep learning ,nlp task."
              "you should be able to write optimized code for the given"
              "Include all the necessary libraries needeed to the task description"
              "Task:{task_description}\n\n"
              "code:"
    )
)

llm = openai(temperature=0.7)

# Create Tools for Code Generation and Explanation
def generate_code(task_description):
    chain = LLMChain(prompt=ml_prompt, llm=llm)
    return chain.run(task_description)

def explain_code(code_snippet):
    explain_prompt = f"Explain the following code in detail:\n\n{code_snippet}\n\nExplanation:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=explain_prompt,
        temperature=0.5,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Define Tools for LangChain Agent
code_generation_tool = tool(
    name="CodeGenerator",
    func=generate_code,
    description="Generates machine learning, deep learning, NLP, and AI code based on a task description."
)

explanation_tool = tool(
    name="CodeExplainer",
    func=explain_code,
    description="Provides detailed explanations for code snippets."
)

# Create LangChain Agent
agent_executor = AgentExecutor(
    tools=[code_generation_tool, explanation_tool],
    llm=llm
)

# Streamlit UI
st.title("AI and ML assistant")
st.write(
    "This assistant can help generate and explain code for various tasks in machine learning, "
    "deep learning, NLP, and general AI development."
)

# User Input for Task Description
task_description = st.text_input("Enter your task description:", "")

if st.button("Generate Code"):
    if task_description:
        with st.spinner("Generating code..."):
            code = generate_code(task_description)
            st.code(code, language="python")
            
            # Explanation Section
            explain_choice = st.checkbox("Explain the generated code")
            if explain_choice:
                with st.spinner("Explaining code..."):
                    explanation = explain_code(code)
                    st.write(explanation)
    else:
        st.error("Please provide a task description to generate code.")

# Sidebar for recent queries or saved code templates
st.sidebar.title("Recent Queries")
st.sidebar.write("Here you can find your recent queries and generated code for quick access.")

'''
This file "app1.py is an updated version of file "app.py" wherein, I've introduced a MODEL PRIORITY list.
This list of MODEL PRIORITY simply enables the live app to show end-result even when 1/2 of all model decline to perform the operation.

Other thing introduced here, is the clean_mermaid_output() function. 
Enabling the live app to throw the respective flowchart for any given input (Python Code) given by user.
'''

# Neccessary import statements 
import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Loading the API key from .env file
load_dotenv()

app = FastAPI(title="Flowchart Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

# MODEL PRIORITY for uninterrupted working of models
MODEL_PRIORITY = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3.1-flash", "gemini-3.1-flash-lite", "gemini-3.1-pro"]

# FUNCTION clean_mermaid_output to output the mermaid.js file with no errors
def clean_mermaid_output(text: str) -> str:
    # 1. Remove literal escaped newlines and quotes that cloud headers sometimes add
    text = text.replace('\\n', '\n').replace('\\"', '"')
    
    # 2. Strip all markdown wrappers
    text = re.sub(r"```mermaid", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    
    # 3. Find the ACTUAL start of the graph
    # This ignores any "Here is your flowchart" conversational filler
    match = re.search(r"graph\s+(TD|LR|TB|BT|RL)", text, re.IGNORECASE)
    if match:
        text = text[match.start():]
    
    # 4. Remove any trailing backticks or notes
    text = text.split("```")[0]
    
    return text.strip()

# The actual setup that passes the user input with the available model and returns 
template = """
You are a strict, expert compiler that translates Python code into Mermaid.js flowchart syntax.
Your ONLY job is to output the raw Mermaid code. 

RULES:
1. Start the graph with 'graph TD'.
2. Use standard flowchart shapes with Node IDs: 
   - NodeID["Rectangle text"] for internal processes
   - NodeID{{"Diamond text"}} for decisions
   - NodeID(("Circle text")) for start/end points
   - NodeID[/"Parallelogram text"/] for print/input
3. CRITICAL: DO NOT include markdown backticks (```).
4. CRITICAL: DO NOT include the word 'mermaid' unless it is part of the graph syntax.
5. THE DOUBLE QUOTE RULE: Wrap every label in double quotes to protect parentheses and brackets.
   - Example: A["print('Hello')"]

Translate the following Python code into a Mermaid flowchart:
{python_code}
"""

prompt = PromptTemplate(input_variables=["python_code"], template=template)

@app.post("/generate")
async def generate_flowchart(request: CodeRequest):
    last_error = ""
    
    for model_name in MODEL_PRIORITY:
        try:
            print(f"DEBUG: Attempting with {model_name}...")
            
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                # This is the key! It looks for the key you'll paste into Render
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            flowchart_chain = prompt | llm | StrOutputParser()
            raw_response = flowchart_chain.invoke({"python_code": request.code})
            
            # Calling clean_mermaid_output() to alter the output, the right way.
            final_syntax = clean_mermaid_output(raw_response)
            
            print(f"DEBUG: Success with {model_name}!")
            return {"mermaid_syntax": final_syntax}
            
        except Exception as e:
            last_error = str(e)
            print(f"DEBUG: {model_name} failed. Error: {last_error[:50]}...")
            continue 
            
    raise HTTPException(
        status_code=500, 
        detail=f"All models exhausted. Last error: {last_error}"
    )

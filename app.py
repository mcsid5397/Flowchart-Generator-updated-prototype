'''
File "app.py" is the engineered (langchain + Google API + mermaid.js (translation)) version of the project https://github.com/mcsid5397/Python-Flowchart-Generator-v1.0_prototype
The v1.0 prototype was entirely built in Python (flask + Google API (code sanitization) + firebase (user database)) and was part of group project.  
'''

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # <-- THE MAGIC FIX

# Load the API key
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

# --- USING YOUR PREVIEW MODEL ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0, 
)

template = """
You are a strict, expert compiler that translates Python code into Mermaid.js flowchart syntax.
Your ONLY job is to output the raw Mermaid code. 

RULES:
1. Start the graph with 'graph TD'.
2. Use standard flowchart shapes with Node IDs: 
   - NodeID["Rectangle text"] for internal processes and variable assignments (e.g., x = 10)
   - NodeID{{"Diamond text"}} for decisions (e.g., if/else conditions)
   - NodeID(("Circle text")) for start/end points
   - NodeID[/"Parallelogram text"/] for Input/Output operations (e.g., print() or input() statements)
3. DO NOT output any conversational text.
4. DO NOT wrap the output in markdown code blocks (no ```mermaid). Just the raw text.
5. THE DOUBLE QUOTE RULE (CRITICAL): To preserve real Python syntax (like parentheses (), brackets [], and commas), you MUST wrap the shape's label in double quotes. Change any inside double quotes to single quotes.
   - BAD:  A[/print("Hello")/]
   - GOOD: A[/"print('Hello')"/]
   - BAD:  B[pos = bisect.bisect_left(a, 8)]
   - GOOD: B["pos = bisect.bisect_left(a, 8)"]
6. DATA STRUCTURES: For arrays/lists, keep using the semantic words "list:" or "tuple:" followed by the items, but ensure the whole label is still protected by the double quotes from Rule 5. 
   - BAD:  C[b = ["Geeks", 4]]
   - GOOD: C["b = list: 'Geeks', 4"]

Translate the following Python code into a Mermaid flowchart:
{python_code}
"""

prompt = PromptTemplate(input_variables=["python_code"], template=template)

# --- THE PARSER IS ADDED HERE ---
flowchart_chain = prompt | llm | StrOutputParser()

@app.post("/generate")
async def generate_flowchart(request: CodeRequest):
    try:
        # 'response' is now guaranteed to be a string, no matter what Gemini 3 outputs!
        response = flowchart_chain.invoke({"python_code": request.code})
        
        # Safe to strip() now!
        return {"mermaid_syntax": response.strip()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
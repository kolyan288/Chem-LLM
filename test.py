# pip and built-in dependencies
import os
import rdkit
import random
import importlib
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langchain.schema import Document
from typing_extensions import TypedDict
from langchain_community.llms import Ollama
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.chat_models import ChatOllama
from typing import Any, Dict, List, Optional, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import render_text_description
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# custom internal dependencies
from llm_utils import feature_extract, MyDoc

with open('API_tokens.txt') as f:
    keys = eval(f.read())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = keys['langchain']
os.environ["NOMIC_API_KEY"] = keys['nomic']
os.environ["TAVILY_API_KEY"] = keys['tavily']
os.environ["OPENAI_API_KEY"] = keys['openai']
mistral_api_key = keys['mistral'] 
os.environ['ANTHROPIC_API_KEY'] = keys['anthropic']
os.environ['GOOGLE_API_KEY'] = keys['google']


output = value['generation']

@tool
def start_generation():
    """If you are sure that a generative algorithm 
    is needed to solve a given problem, 
    run the "start_generation" function"""
    
    print('***ЗАПУЩЕН ГЕНЕРАТИВНЫЙ АЛГОРИТМ***')
    df = pd.read_csv('dataset.csv')
    print(df.head())
    

@tool
def start_classification():
    """If you are sure that a classification algorithm 
    is needed to solve a given problem, 
    run the "start_classification" function"""
    
    print('***ЗАПУЩЕН АЛГОРИТМ КЛАССИФИКАЦИИ***')
    df = pd.read_csv('dataset.csv')
    print(df.head())
    
@tool
def start_regression():
    """If you are sure that a regression algorithm 
    is needed to solve a given problem, 
    run the "start_classification" function"""
    
    print('***ЗАПУЩЕН АЛГОРИТМ РЕГРЕССИИ***')
    df = pd.read_csv('dataset.csv')
    print(df.head())

tools = [start_generation, start_classification, start_regression]

model = Ollama(model="llama3:8b")

rendered_tools = render_text_description(tools)

system_prompt = f"""\
You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

The `arguments` should be a dictionary, with keys corresponding 
to the argument names and the values corresponding to the requested values.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

class ToolCallRequest(TypedDict):
    """A typed dict that shows the inputs into the invoke_tool function."""

    name: str
    arguments: Dict[str, Any]

def invoke_tool(tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None):
    
    """A function that we can use the perform a tool invocation.

    Args:
        tool_call_request: a dict that contains the keys name and arguments.
            The name must match the name of a tool that exists.
            The arguments are the arguments to that tool.
        config: This is configuration information that LangChain uses that contains
            things like callbacks, metadata, etc.See LCEL documentation about RunnableConfig.

    Returns:
        output from the requested tool
    """
    
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)

chain = prompt | model | JsonOutputParser() | invoke_tool
chain.invoke({"input": output})


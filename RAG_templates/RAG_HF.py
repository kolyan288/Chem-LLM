###-----------------------------------IMPORT LIBRARIES------------------------------------###

import os
import bs4
import torch
from torch import cuda
from huggingface_hub import login
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from transformers import BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

###-------------------------------------LOAD DOCUMENTS------------------------------------###

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

###---------------------------------------LOAD KEYS---------------------------------------###

with open('API_tokens.txt') as f:
    keys = eval(f.read())

###------------------------------------EMBED DOCUMENTS------------------------------------###

os.environ['LANGCHAIN_TRACING_V2'] = 'True'
os.environ['LANGCHAIN_API_KEY'] = keys['langchain']
os.environ['OPENAI_API_KEY'] = keys['openai']
os.environ['HUGGINGFACEHUB_API_TOKEN'] = keys['hf']

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
docs_as_strs = [doc.page_content for doc in splits]
embedded_docs = embed_model.embed_documents(docs_as_strs)

###------------------------------------CREATE RETRIEVER-----------------------------------###

db = Chroma.from_documents(splits, embed_model)
retriever = db.as_retriever()

###----------------------------------------INIT LLM---------------------------------------###

login(keys['hf'])
model_id = "meta-llama/Meta-Llama-Guard-2-8B"
device = "cuda"
dtype = torch.bfloat16
access_token = keys['hf']

quantization_config = BitsAndBytesConfig(load_in_8bit = True, llm_int8_enable_fp32_cpu_offload = True)

llm = AutoModelForCausalLM.from_pretrained(model_id,
                                           torch_dtype=dtype, 
                                           device_map=device,
                                           trust_remote_code = True,
                                           quantization_config = quantization_config)

###------------------------------------BUILD PIPELINE-------------------------------------###

tokenizer = AutoTokenizer.from_pretrained(model_id,token = access_token)
pipe = pipeline('text-generation', model = llm, tokenizer = tokenizer, max_length = 1000)
model = HuggingFacePipeline(pipeline=pipe)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# question = 

response = rag_chain.invoke({"input": "What are you doing today?"})

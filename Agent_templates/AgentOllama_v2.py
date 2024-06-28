import os
import pprint
from torch import cuda
from typing import List
from langchain import hub
from langchain.schema import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.llms.ollama_functions import OllamaFunctions

###--------------------------------------OS ENVIRONMENTS------------------------------------###

with open('API_tokens.txt') as f:
    keys = eval(f.read())

os.environ["COHERE_API_KEY"] = keys['cohere']
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = keys['langchain']
os.environ['TAVILY_API_KEY'] = keys['tavily']
local_llm = 'llama3:8b'

###-------------------------------------GRAPH STATE INIT------------------------------------###

class GraphState(TypedDict):
    """|
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

###-------------------------------------BUILD RETRIEVER-------------------------------------###

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embd = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=0)

doc_splits = text_splitter.split_documents(docs_list)
doc_as_strs = [doc.page_content for doc in doc_splits]

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embd,
)

retriever = vectorstore.as_retriever()

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---РЕТРИВЕР---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    #documents = docs_list
    return {"documents": documents, "question": question}


    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---LLM Fallback---")
    question = state["question"]
    generation = llm_chain.invoke({"question": question})
    return {"question": question, "generation": generation}

###-----------------------------------------BUILD RAG---------------------------------------###

class Output(BaseModel):
    #model: str #= Field(description=description_models, required=True)
    #dataset: str #= Field(description=description_datasets, required=True)
    task: list #= Field(description=description_tasks, required=True)

prompt = hub.pull("rlm/rag-prompt")

llm = OllamaFunctions(model=local_llm, format = 'json', temperature=0)
structured_llm = llm.with_structured_output(Output)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | structured_llm 

def generate(state):
    """
    Generate answer using the vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

###----------------------------------------BULD GRADER--------------------------------------###

llm = ChatOllama(model=local_llm, format="json", temperature=0)

# prompt = PromptTemplate(
#     template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
#     Here is the retrieved document: \n\n {document} \n\n
#     Here is the user question: {question} \n
#     If the document contains keywords related to the user question, grade it as relevant. \n
#     It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
#     Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
#     input_variables=["question", "document"],
# )

prompt = PromptTemplate(
    template="""Вы — оценщик, оценивающий соответствие полученного документа вопросу пользователя. \n
    Вот полученный документ: \n\n {document} \n\n
    Вот вопрос пользователя: {question} \n
    Если документ содержит ключевые слова, связанные с вопросом пользователя, оцените его как релевантный. \n
    Это не обязательно должен быть строгий тест. Цель состоит в том, чтобы отфильтровать ошибочные запросы. \n
    Дайте двоичную оценку «yes» или no», чтобы указать, имеет ли документ отношение к вопросу. \n
    Предоставьте двоичную оценку в формате JSON с одним ключом «score» без преамбулы или пояснений.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---ПРОВЕРКА РЕЛЕВАНТНОСТИ ДОКУМЕНТОВ---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: ДОКУМЕНТ РЕЛЕВАНТНЫЙ---")
            filtered_docs.append(d)
        else:
            print("---GRADE: ДОКУМЕНТ НЕРЕЛЕВАНТНЫЙ---")
            continue
    return {"documents": filtered_docs, "question": question}

###----------------------------------BUILD WEB SEARCH TOOL----------------------------------###

web_search_tool = TavilySearchResults(k=3)

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}

###-------------------------------------QUESTION ROUTER-------------------------------------###

llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""Вы являетесь экспертом в перенаправлении вопросов пользователя в векторную базу данных или в веб-поиск. \n
    Используйте векторную базу данных, чтобы найти документы, относящиеся к вопросу пользователя \n
    Не нужно быть строгим с ключевыми словами в вопросе, относящемся к этим темам. \n
    В противном случае используйте веб-поиск. Дайте двоичный выбор «web_search» или «vectorstore» в зависимости от вопроса. \n
    Верните JSON с одним ключом «datasource» без преамбулы или объяснения. \n
    Вопрос для направления: {question}""",
    input_variables=["question"],
)

question_router = prompt | llm | JsonOutputParser()

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---АДРЕСАЦИЯ ПОЛУЧЕННОГО ВОПРОСА---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ВОПРОС АДРЕСОВАН К ВЕБ-ПОИСКУ---")
        return "web_search"
    elif source["datasource"] == "vectorstore":
        print("---ВОПРОС АДРЕСОВАН К ВЕКТОРНОМУ ХРАНИЛИЩУ---")
        return "vectorstore"
   
###------------------------------------DECIDE TO GENETATE-----------------------------------###

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ПРОЦЕСС ОЦЕНИВАНИЯ ДОКУМЕНТОВ---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ВСЕ ДОКУМЕНТЫ НЕРЕЛЕВАНТНЫ, ПЕРЕФОРМУЛИРОВКА ВОПРОСА---"
        )
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

   


###-------------------------------------GRADE GENERATION------------------------------------###

###-------------------------HALLUCINATION GRADER------------------------###

llm = ChatOllama(model=local_llm, format="json", temperature=0)

# prompt = PromptTemplate(
#     template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
#     Here are the facts:
#     \n ------- \n
#     {documents} 
#     \n ------- \n
#     Here is the answer: {generation}
#     Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
#     Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
#     input_variables=["generation", "documents"],
# )

prompt = PromptTemplate(
    template="""Вы оценщик, оценивающий, основан ли ответ на наборе фактов или подкреплен им. \n
    Вот факты:
    \n ------- \n
    {documents} 
    \n ------- \n
    Вот ответ: {generation}
    Присвойте двоичную оценку «yes» или «no», чтобы указать, основан ли ответ на наборе фактов или подкреплен им. \n
    Предоставьте двоичную оценку в формате JSON с одним ключом «score» без преамбулы или пояснений. На вы""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()

###------------------------------LLM GRADER-----------------------------###

llm = ChatOllama(model=local_llm, format="json", temperature=0)

# prompt = PromptTemplate(
#     template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
#     Here is the answer:
#     \n ------- \n
#     {generation} 
#     \n ------- \n
#     Here is the question: {question}
#     Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
#     Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
#     input_variables=["generation", "question"],
# )

prompt = PromptTemplate(
    template="""Вы оценщик, оценивающий, полезен ли ответ для решения вопроса. \n
    Вот ответ:
    \n ------- \n
    {generation} 
    \n ------- \n
    Вот вопрос: {question}
    Дайте двоичную оценку «yes» или «no», чтобы указать, полезен ли ответ для решения вопроса. \n
    Предоставьте двоичную оценку в формате JSON с одним ключом «score» без преамбулы или пояснений. \n
    Ты ОБЯЗАН поставить оценку 'yes' когда ответ представляет собой \n
    JSON Output, в котором есть ключ 'task', а значение - список из необходимых задач, имеющих то или \n 
    иное отношение к документу. """,
    
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print(grade)
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


###--------------------------------------BUILD PIPELINE-------------------------------------###

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # rag

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",  # Hallucinations: re-generate
        "not useful": "web_search",  # Fails to answer question: fall-back to web-search
        "useful": END,
    },
)

app = workflow.compile()

###--------------------------------------BUILD PIPELINE-------------------------------------###

inputs = {
    "question": "What player are the Bears expected to draft first in the 2024 NFL draft?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint.pprint(f"Node '{key}':")
        # Optional: print full state at each node
    pprint.pprint("\n---\n")

# Final generation
pprint.pprint(value["generation"])
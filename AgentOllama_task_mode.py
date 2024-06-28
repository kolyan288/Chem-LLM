import os
import pandas as pd
from torch import cuda
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

from llm_utils import MyDoc

def agentollama_task_mode(prompt_value):

    ###---------------------------------------OS ENVIRONMENTS-------------------------------------###

    with open('API_tokens.txt') as f:
        keys = eval(f.read())

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = keys['langchain']
    os.environ["NOMIC_API_KEY"] = keys['nomic']
    os.environ["TAVILY_API_KEY"] = keys['tavily']
    os.environ["OPENAI_API_KEY"] = keys['openai']
    mistral_api_key = keys['mistral'] 

    ###--------------------------------------BUILD RETRIEVER--------------------------------------###

    #local_llm = 'solar'
    local_llm = 'llama3:8b'
    #local_llm = 'mistral'

    folder_path = "RAG_documents/"
    texts = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(MyDoc(file.read()))

    docs_list = texts
    print(len(docs_list))

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs_list)

    # embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    # device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    # embd = HuggingFaceEmbeddings(
    #     model_name=embed_model_id,
    #     model_kwargs={'device': device},
    #     encode_kwargs={'device': device, 'batch_size': 32}
    # )

    embd = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode='local')

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embd
    )

    retriever = vectorstore.as_retriever()

    ###--------------------------------------QUESTION ROUTER--------------------------------------###

    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # prompt = PromptTemplate(
    #     template="""You are an expert at routing a user question to a vectorstore or web search. \n
    #     Use the vectorstore for questions on LLM  agents, prompt engineering, and adversarial attacks. \n
    #     You do not need to be stringent with the keywords in the question related to these topics. \n
    #     Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. \n
    #     Return the a JSON with a single key 'datasource' and no premable or explanation. \n
    #     Question to route: {question}""",
    #     input_variables=["question"],
    # )

    prompt = PromptTemplate(
        template="""Вы являетесь экспертом в перенаправлении вопросов пользователя в векторную базу данных или в веб-поиск. \n
        Используйте векторную базу данных, чтобы задать вопросы об агентах LLM, оперативном проектировании и состязательных атаках. \n
        Не нужно быть строгим с ключевыми словами в вопросе, относящемся к этим темам. \n
        В противном случае используйте веб-поиск. Дайте двоичный выбор «web_search» или «vectorstore» в зависимости от вопроса. \n
        Верните JSON с одним ключом «datasource» без преамбулы или объяснения. \n
        Вопрос для направления: {question}""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()

    ###-----------------------------------------BULD GRADER---------------------------------------###

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

    ###------------------------------------------BUILD RAG----------------------------------------###

    class Output(BaseModel):
        #model: str #= Field(description=description_models, required=True)
        #dataset: str #= Field(description=description_datasets, required=True)
        task: list #= Field(description='list of tasks', required=True)

    prompt = PromptTemplate(
        
        template = """
        Вопрос пользователя: {question}
        Ты - русскоязычный ml ассистент. 
        Тебе нужно построить пайплайн машинного обучения, состоящего из нескольких блоков.
        Я тебе в этом помогу. Во-первых, внимательно прочитай текст, данный пользователем, 
        и разбей его на смысловые куски. Сначала нужно понять, с какой задачей мы имеем дело.
        Для определения задачи тебе помогут следующие документы: {documents}
        
        Тебе нужно понять, какую задачу хочет решить пользователь и дать ответ в формате JSON. 
        JSON должен содержать следующие поля, основываять на тексте пользователя,
        с возможными значениями:
        
        'task_mode': [Задача, которая должна решаться пользователем],
        Ответом должен быть исключительно JSON. В
        Создайте словарь в формате JSON, где ключ 'task_mode' связан со строкой, которая обозначает ту
        или иную задачу. Выбирать ты можешь исключительно одну из трех задач: 'classification', 'regression', 'generation'
        
        
        Если нет уверенности в значении какого либо поля - ставь null.
        
        
        """,
        input_variables=["question", "documents"],
    )

    llm = ChatOllama(model=local_llm, format = 'json', temperature=0)
    #llm = ChatMistralAI(model="mistral-large-latest")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | JsonOutputParser() 

    ###------------------------------------HALLUCINATION GRADER-----------------------------------###

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
        Предоставьте двоичную оценку в формате JSON с одним ключом «score» без преамбулы или пояснений. \n 
        Ты ОБЯЗАН поставить оценку 'yes' когда ответ представляет собой \n
        Словарь python, в котором есть ключ 'task_mode', а значение - 'classification', 'regression' или 'generation'\n """,
        
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()

    ###----------------------------------------ANSWER GRADER--------------------------------------###

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
        JSON Output, в котором есть ключ 'task_mode', а значение - 'classification', 'regression' или 'generation'\n 
        
        """,
        
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()

    ###--------------------------------------QUESTION REWRITER------------------------------------###

    llm = ChatOllama(model=local_llm, temperature=0)

    # re_write_prompt = PromptTemplate(
    #     template="""You a question re-writer that converts an input question to a better version that is optimized \n 
    #      for vectorstore retrieval. Look at the initial and formulate an improved question. \n
    #      Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    #     input_variables=["generation", "question"],
    # )

    re_write_prompt = PromptTemplate(
        template="""Вы переписываете вопрос, который преобразует входной вопрос в лучшую версию, оптимизированную \n 
        для поиска в векторном хранилище. Посмотрите на исходную и сформулируйте улучшенный вопрос. \n
        Вот первоначальный вопрос: \n\n {question}. Улучшенный вопрос без преамбулы: \n""",
        input_variables=["generation", "question"],
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    ###-----------------------------------BUILD WEB SEARCH TOOL-----------------------------------###

    web_search_tool = TavilySearchResults(k=3)

    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            documents: list of documents
        """

        question: str
        generation: str
        documents: List[str]

    ###-------------------------------------BUILD GRAPH NODES-------------------------------------###

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

    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        
        print("---ГЕНЕРАЦИЯ---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain.invoke({"documents": documents, "question": question})
        print(generation)
        return {"documents": documents, "question": question, "generation": generation}

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

    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---ПЕРЕФОРМУЛИРОВКА ВОПРОСА---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        print(better_question)
        return {"documents": documents, "question": better_question}

    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---ВЕБ ПОИСК---")
        question = state["question"]

        # Web search
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

    ###-------------------------------------BUILD GRAPH EDGES-------------------------------------###

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
                "---РЕШЕНИЕ: ВСЕ ДОКУМЕНТЫ НЕРЕЛЕВАНТНЫ, ПЕРЕФОРМУЛИРОВКА ВОПРОСА---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---РЕШЕНИЕ: ГЕНЕРАЦИЯ---")
            return "generate"

    def grade_generation_v_documents_and_question(state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---ПРОВЕРКА НА ГАЛЛЮЦИНАЦИИ---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        print(generation)
        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---РЕШЕНИЕ: ГЕНЕРАЦИЯ СООТВЕТСТВУЕТ ОТОБРАННЫМ ДОКУМЕНТАМ---")
            # Check question-answering
            print("---ОЦЕНКА СГЕНЕРИРОВАННОГО ОТВЕТА И ВОПРОСА---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---РЕШЕНИЕ: ГЕНЕРАЦИЯ СООТВЕТСТВУЕТ ВОПРОСУ---")
                return "useful"
            else:
                print("---РЕШЕНИЕ: ГЕНЕРАЦИЯ НЕ СООТВЕТСТВУЕТ ВОПРОСУ---")
                return "not useful"
        else:
            print(grade)
            print("---РЕШЕНИЕ: ГЕНЕРАЦИЯ НЕ СООТВЕТСВУЕТ ОТОБРАННЫМ ДОКУМЕНТАМ---")
            return "not supported"

    ###----------------------------------------BUILD GRAPH----------------------------------------###

    workflow = StateGraph(GraphState)

    workflow.add_node("web_search", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("transform_query", transform_query)  # transform_query

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
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "transform_query",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    app = workflow.compile()

    ###-------------------------------------------RUN APP-----------------------------------------###

    my_prompt = prompt_value + ' Выбери на основании этого запроса задачу, которую нужно решить: classification, regression или generation'
    
    inputs = {"question": my_prompt}

    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            print(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        print('\n---\n')

    # Final generation
    print(value["generation"])

    ###---------------------------------------WORK WITH TOOLS-------------------------------------###

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
    
    You must write JSON in one line so that the parser can recognize it
    
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
    
if __name__ == '__main__':
    agentollama_task_mode("Нужно предсказать токсичность белков при помощи графовой нейронной сети")
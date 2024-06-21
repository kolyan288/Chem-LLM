from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Local Llama3
llm = ChatOllama(
    model="llama3:8b",
    keep_alive=-1,  # Модель не будет выгружаться
    temperature=-1,
    max_new_tokens=512,
)


with open('datasets_list.txt') as file:
    lines = file.read().split('\n')
        
# prompt = ChatPromptTemplate.from_template(
#     "Write me a 500 word article on {topic} from the perspective of a {profession}. "
# )


prompt = ChatPromptTemplate.from_template(  
    """
    You are an expert in the field of chemical datasets. You need to write in as much 
    detail as possible what data is present in the dataset and what values ​​are target variables. 
    Describe the data in the dataset in MAXIMUM detail. So, tell me about the dataset "{value}".
    """
    )
    
    
chain = prompt | llm | StrOutputParser()

# print(chain.invoke({"topic": "LLMs", "profession": "shipping magnate"}))
for i in lines:
    print(i)
    response = chain.invoke({'value': i})
    print('-' * 80)
    with open(f'RAG_documents/dataset_{i}.txt', 'a', encoding = 'utf-8') as file1:
        for string in response.split('\n'):
            file1.write(f'{string}')
   
    # for chunk in chain.stream({"value": eval(i)[0]}):
    #     print(chunk, end="", flush=True)
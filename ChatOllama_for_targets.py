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


with open('targets_advanced.txt') as file:
    lines = file.read().split('\n')
        
# prompt = ChatPromptTemplate.from_template(
#     "Write me a 500 word article on {topic} from the perspective of a {profession}. "
# )


prompt = ChatPromptTemplate.from_template(  
    """
    You are an expert on the ChemBL chemical database. In this database there are so-called “activities”, 
    from where you can get the name standard_type. You need to describe in MAXIMUM detail what meaning 
    this name makes from a chemistry point of view. So, you need to tell me what {value} is in the ChemBL database
    """
    )
    
    
chain = prompt | llm | StrOutputParser()

# print(chain.invoke({"topic": "LLMs", "profession": "shipping magnate"}))
for i in lines:
    print(eval(i)[0])
    response = chain.invoke({'value': eval(i)[0]})
    print('-' * 80)
    with open('features_ChemBL.txt', 'a', encoding = 'utf-8') as file:
        for string in response.split('\n'):
            file.write(f'{string}')
   
    # for chunk in chain.stream({"value": eval(i)[0]}):
    #     print(chunk, end="", flush=True)
import subprocess
import os

from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


with open('description_models.txt', 'r') as f1:
    description_models = f1.read().replace('\n', ' ')

with open('description_datasets.txt', 'r') as f2:
    description_datasets = f2.read().replace('\n', ' ')

with open('description_tasks.txt', 'r') as f3:
    description_tasks = f3.read().replace('\n', ' ')

with open('description_function.txt', 'r') as f4:
    description_function = f4.read().replace('\n', ' ')

class Person(BaseModel):
    #model: str #= Field(description=description_models, required=True)
    #dataset: str #= Field(description=description_datasets, required=True)
    task: list #= Field(description=description_tasks, required=True)

context = description_function

# Prompt template
# prompt = PromptTemplate.from_template(

#     """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     You are a smart assistant take the following context and question below and return your answer in JSON.
#     <|eot_id|><|start_header_id|>user<|end_header_id|>
# QUESTION: {question} \n
# CONTEXT: {context} \n
# JSON:
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """
# )

prompt = ChatPromptTemplate.from_template(
    
    
    
    """
    Вопрос пользователя: {question}
    Ты - русскоязычный ml ассистент. 
    Тебе нужно построить пайплайн машинного обучения, состоящего из нескольких блоков.
    Я тебе в этом помогу. Во-первых, внимательно прочитай текст, данный пользователем, 
    и разбей его на смысловые куски. Сначала нужно понять, с каким датасетом мы имеем дело.
    Обычно пользователь указывает целевые таргеты или целевые признаки - значения, которые нужно 
    спрогнозировать при помощи ML модели. И на этом этапе необходимо получить первую пару словаря 
    ключ-значение. Эта пара выглядит следующим образом: 'task': ['target_1', 'target_2' ... 'target_n'],
    где 'target_1', 'target_2' и 'target_n' - некоторые целевые значения, которые тебе необходимо вытащить.
    Их может быть необязательно много. Их может быть один 'task': ['target_1'] или два
    'task': ['target_1', 'target_2']. Либо же может быть указан напрямую датасет, который
    необходимо обработать. Он должен попадать в следующий список: {description_datasets}
     
    
    Тебе нужно понять, какую задачу хочет решить пользователь и дать ответ в формате JSON. 
    JSON должен содержать следующие поля, основываять на тексте пользователя,
    с возможными значениями:
    
    
    
    'task': ['target_1', 'target_2' ... 'target_n],
    Ответом должен быть исключительно JSON.
    Если нет уверенности в значении какого либо поля - ставь null.
    """
)

llm = OllamaFunctions(model="llama3:8b", format="json", temperature = 0)
structured_llm = llm.with_structured_output(Person)
chain = prompt | structured_llm


question = 'Нужно предсказать токсичность белков при помощи графовой нейронной сети'

question = 'Нужно предсказать значения IC50 и EC50 для датасета молекул, а также провести предсказания на датасете липофильности молекул logp'
# question = 'Нужно предсказать значение липофильности для молекул'

response = chain.invoke({'question': question, 'description_datasets': description_datasets})

print(response)

if response.model == 'GNN_Classifier':
    
    if response.dataset == 'tox21':
        os.chdir('/home/kolyan288/Pyprojects/OpenChem_local')
        subprocess.run('python3 launch.py --nproc_per_node=1 run.py --config_file="./example_configs/tox21_rnn_config.py" --mode="train_eval"', shell = True)
        subprocess.run(f'python3 launch.py --nproc_per_node=1 run.py --config_file="./example_configs/tox21_rnn_config.py" --mode="predict"', shell = True)
        with open('/home/kolyan288/Pyprojects/OpenChem_local/logs/tox21_rnn_log/predictions.txt') as f:
            for line in f:
                print(line)
                print()
        
elif response.model == 'GNN_Regressor':
        
    if response.dataset == 'logp':
        os.chdir('/home/kolyan288/Pyprojects/OpenChem_local')
        subprocess.run(f'python3 launch.py --nproc_per_node=1 run.py --config_file="./example_configs/logp_gcnn_config.py" --mode="train_eval"', shell = True)
        subprocess.run(f'')
        with open('/home/kolyan288/Pyprojects/OpenChem_local/logs/logp_gcnn_logs/predictions.txt') as f:
            for line in f:
                print(line)
                print()
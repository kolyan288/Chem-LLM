from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

model = OllamaFunctions(model="llama3", format="json")

with open('description_models.txt', 'r') as f1:
    description_models = f1.read().replace('\n', ' ')

with open('description_datasets.txt', 'r') as f2:
    description_datasets = f2.read().replace('\n', ' ')

with open('description_tasks.txt', 'r') as f3:
    description_tasks = f3.read().replace('\n', ' ')

with open('description_function.txt', 'r') as f4:
    description_function = f4.read().replace('\n', ' ')

class Person(BaseModel):
    model: str = Field(description=description_models, required=True)
    dataset: str = Field(description=description_datasets, required=True)
    task: str = Field(description=description_tasks, required=True)

context = description_function

# Prompt template
prompt = PromptTemplate.from_template(

    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a smart assistant take the following context and question below and return your answer in JSON.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
QUESTION: {question} \n
CONTEXT: {context} \n
JSON:
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
)
# model = model.bind_tools(
#     tools=[
#         {
#             "name": "build_pipeline",
#             "description": description_function,
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "model": {
#                         "type": "string",
#                         "description": description_models,
#                         "must_to_select": ["GNN_Classifier", 
#                                  "GNN_Regressor", 
#                                  "XGBoostClassifier", 
#                                  "XGBoostRegressor"]
#                     },
#                     "dataset": {
#                         "type": "string",
#                         "description": description_datasets,
#                         "must_to_select": ["BACE",
#                                  "BBBC001",
#                                  "BBBC002",
#                                  "BBBC003",
#                                  "BBBP",
#                                  "ESOL_Delaney", 
#                                  "Factors",
#                                  "tox21"],
#                     },
#                     "task": {
#                         "type": "string",
#                         "description": description_tasks,
#                         "must_to_select": ["classification", 
#                                  "regression", 
#                                  "generation"]
#                     }
#                 },
#                 "required": ["model", "dataset", "task"],
#             },
#         } 
#     ],
#     function_call={
#         "name": "build_pipeline"
#     },  # Этот параметр ЗАСТАВЛЯЕТ модель использовать функцию ОБЯЗАТЕЛЬНО
# )

# response = model.invoke("Нужно предсказать токсичность белков при помощи графовой нейронной сети")

# print(response)

llm = OllamaFunctions(model="llama3", format="json")
structured_llm = llm.with_structured_output(Person)
chain = prompt | structured_llm

alex = chain.invoke({'question': 'Сделай нормальный пайплайн такую штуку короче молекулу классифицировать при попощи графовой нейросети', 'context': context})
alex


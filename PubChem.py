import pubchempy as pcp
import pandas as pd



# # Создание датасета
# dataset = []
# for compound in compounds:
#     data = {}
#     # Получение SMILES-строки
#     data['smiles'] = compound.isomeric_smiles
    
#     # Получение target-переменной (здесь пример с молекулярной массой)
#     data['target'] = compound.molecular_weight
    
#     dataset.append(data)

# # Преобразование списка словарей в DataFrame
# df = pd.DataFrame(dataset)

# # Вывод первых 5 записей датасета
# print(df.head())

def get_bioactivity_data(cid):
    compound = pcp.Compound.from_cid(cid)
    smiles = compound.isomeric_smiles
    # Получение результатов биоассеев
    bioassays = pcp.get_bioassays(cid)
    # Выбор конкретного биоассея или агрегация результатов
    # Это примерный код; вам нужно будет адаптировать его под вашу задачу
    bioactivity = bioassays[0]['results'][0]['value'] if bioassays else None
    return smiles, bioactivity

'a'
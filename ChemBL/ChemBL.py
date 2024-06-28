# import pandas as pd
# from chembl_webresource_client.new_client import new_client

# target = new_client.target
# target_query = target.search('predict protein toxicity using a graph neural network')
# targets = pd.DataFrame.from_dict(target_query)
# targets

from chembl_webresource_client.new_client import new_client

# Создание нового клиента для доступа к данным
molecule = new_client.molecule
target = new_client.target
activity = new_client.activity

# Задание target-переменной, например, 'coronavirus'
target_query = 'predict protein toxicity using a graph neural network'
targets = target.search(target_query)

# Получение первого совпадения (можно настроить поиск под конкретные нужды)
target_id = targets[0]['target_chembl_id']

# Получение активностей для заданного target
activities = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")

# Создание датасета
dataset = []
for act in activities:
    data = {}
    # Получение SMILES-строки
    chembl_id = act['molecule_chembl_id']
    molecule_info = molecule.get(chembl_id)
    data['smiles'] = molecule_info['molecule_structures']['canonical_smiles']
    
    # Получение target-переменной
    data['target'] = act['standard_value']
    
    dataset.append(data)

# Вывод первых 5 записей датасета
print(dataset[:5])


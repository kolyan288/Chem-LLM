import importlib
from rdkit.Chem import MolFromSmiles
import numpy as np

def __can_import(module_names, function_name):
    
    return_nan = lambda x: np.nan
    
    for module_name in module_names:
        
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, function_name):
                print('1')
                feat = getattr(module, function_name)
                return feat
               
        except ImportError as e:
            print('3')
            print(f"Ошибка импорта: {e}")
            return return_nan

    return return_nan

def feature_extract(prop, mol):
    
    module_names = ['rdkit.Chem.GraphDescriptors', 'rdkit.Chem.Crippen']
    feat = __can_import(module_names, prop)
    return feat(MolFromSmiles(mol))

if __name__ == '__main__':

    import rdkit
    mol = 'CC(=O)C=CC=C'

    lol = feature_extract('иди на', mol)
    print(lol)
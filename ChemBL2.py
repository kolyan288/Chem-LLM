from chembl_webresource_client.new_client import new_client
import random

molecule = new_client.molecule
target = new_client.target
activity = new_client.activity



with open('targets.txt', 'r', encoding = 'utf-8') as f:
    targets = [line.strip() for line in f]

while True:
    chh = random.choice(activity)
    if chh['standard_type'] not in targets:
        print(chh['standard_type'])
        targets.append(chh['standard_type'])
         
        
        
        
        

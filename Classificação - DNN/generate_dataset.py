import pandas as pd
from math import log
from random import shuffle, seed

propertiesdata = pd.read_csv('../molecules_properties.csv')

print(propertiesdata.columns)

plist = propertiesdata['corante_00'] | propertiesdata['corante_max']

propertiesdata['iscorante'] = [int(x) for x in plist]
print(propertiesdata['iscorante'].value_counts())

print('*************************** RDKIT ***************************')
rdkit = pd.read_csv('../rdkit_descriptors.csv')
print(rdkit.columns)
rdkit.drop(['Unnamed: 0', 'cid', 'isomeric smiles'], axis='columns', inplace=True)
print(rdkit.columns)

rdkit['MW'] = [log(x+0.001) for x in rdkit['MW']]

print('*************************** KULIK ***************************')
kulik = pd.read_csv('../kulik.csv')
print(kulik.columns)
kulik.drop(['Unnamed: 0', 'cid'], axis='columns', inplace=True)
print(kulik.columns)

kulik = kulik.map(lambda x: log(x+0.001))

print('*************************** FINAL DF ***************************')

rdkit = {**rdkit, **kulik}   

rdkit['iscorante'] = propertiesdata['iscorante']


df = pd.DataFrame.from_dict(rdkit)
df_false = df[df['iscorante'] == False]
index_array = list(df_false.index)
# for i in range(4):
#     seed()
#     shuffle(index_array)
index_array = index_array[0:round(len(index_array)/2)]
df = df.drop(index_array, axis = 0)

print()

print(df.shape)
df = df.fillna(0)
print(df.shape)
print(df.columns)

df.to_csv("dataset.csv", index = False)
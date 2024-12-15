import pandas as pd
import numpy as np
import kagglehub as kghub
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Lipinski

def create_property_descriptors(smiles, depth, prop, prop_index=0):
    '''
    Função:
        create_property_descriptors(smiles, depth, prop, prop_index=0)

    Descrição:
        Permite criar descritores baseados em propriedades atômicas e conectividade básica entre os átomos

    Argumentos:
        smiles:     string SMILES da molécula que se deseja gerar os descritores. Exemplo: 'C(C)CO'.
        depth:      profundidade, parâmetro que permite definir o nivel máximo de conectividade que será
                    usado na geração dos descritores. Exemplo, se depth=3, três descritores serão gerados,
                    com conectividade iguais a um, dois e três.
        prop:       dicionário formado por pares 'átomo': propriedade ou 'átomo':[propriedades] ou
                    'átomo':(propriedades).
        prop_index: índice da propriedade de interesse nos valores de 'prop'. Se estes valores não forem
                    listas ou tuplas, este parâmetro é ignorado. Se forem, este parâmetro deve ser
                    configurado com um número maior ou igual a zero e menor do que o número de itens nas
                    listas ou tuplas. O valor default deste parâmetro é zero, ou seja, a função pegará a
                    primeira propriedade da lista.
    '''
    # Criar a estrutura a partir do SMILES da molécula, adicionando os H faltantes
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Preparar listas de resultados (separação por depth=0, 1, 2, 3...)
    result_list = [0] * (depth+1) # Se depth = 2, precisamos de 3 itens (para d=[0,1,2])

    # Obter a matriz de distâncias entre cada um dos átomos e a lista de átomos
    atom_list = [ atom.GetSymbol() for atom in mol.GetAtoms() ]
    n_atoms = len(atom_list)
    dist_matrix = np.tril(Chem.rdmolops.GetDistanceMatrix(mol, force=True))

    # Iterar toda a matriz, extraindo os dados de interesse e separando por valor de depth
    for i in range(0, n_atoms):
        for j in range(0, n_atoms):
            d = int(dist_matrix[i][j])

            if d <= depth and d == 0 and i == j:
                a_i = atom_list[i]
                p_i = 0

                if prop[a_i] is list or tuple:
                    p_i = prop[a_i][prop_index]   # caso haja + de 1 propriedade p/ calcular

                else:
                    p_i = prop[a_i]            # caso haja somente 1 propriedade p/ calcular

                result_list[d] += round(p_i * p_i, 3)

            if d <= depth and d > 0:
                a_i = atom_list[i]
                a_j = atom_list[j]
                p_i, p_j = 0, 0

                if prop[a_i] is list or tuple:
                    p_i = prop[a_i][prop_index]   # caso haja + de 1 propriedade p/ calcular
                else:
                    p_i = prop[a_i]            # caso haja somente 1 propriedade p/ calcular
                if prop[a_j] is list or tuple:
                    p_j = prop[a_j][prop_index]
                else:
                    p_j = prop[a_j]
                result_list[d] += round(p_i * p_j, 3)    # posição [d] na lista de resultados é incrementada

    # Finalizando
    return result_list

path = kghub.dataset_download("fanconic/smiles-toxicity")

data_smiles= pd.read_csv(path+'/data/NR-ER-train/names_smiles.csv')
data_smiles.columns = ['name', 'smiles']

elem_data = pd.read_csv('Atomic properties DB.csv')
properties_dict = elem_data.set_index('Symbol').T.to_dict('list')

RACs_result = []

for smile, cid in zip(data_smiles['smiles'], data_smiles['name']):
    try:
        mass = create_property_descriptors(smile, 3, properties_dict, 2)
        EN = create_property_descriptors(smile, 3, properties_dict, 3)
        In = create_property_descriptors(smile, 3, properties_dict, 4)
        aRadius = create_property_descriptors(smile, 3, properties_dict, 5)
        VdW = create_property_descriptors(smile, 3, properties_dict, 6)
        covRadius = create_property_descriptors(smile, 3, properties_dict, 7)
        valence = create_property_descriptors(smile, 3, properties_dict, 8)

        dict_RACs = {
            'name': cid,
            'mass dZero': mass[0],
            'mass dOne': mass[1],
            'mass dTwo': mass[2],
            'mass dThree': mass[3],
            'EN dZero': EN[0],
            'EN dOne': EN[1],
            'EN dTwo': EN[2],
            'EN dThree': EN[3],
            'In dZero': In[0],
            'In dOne': In[1],
            'In dTwo': In[2],
            'In dThree': In[3],
            'aRadius dZero': aRadius[0],
            'aRadius dOne': aRadius[1],
            'aRadius dTwo': aRadius[2],
            'aRadius dThree': aRadius[3],
            'VdW dZero': VdW[0],
            'VdW dOne': VdW[1],
            'VdW dTwo': VdW[2],
            'VdW dThree': VdW[3],
            'covRadius dZero': covRadius[0],
            'covRadius dOne': covRadius[1],
            'covRadius dTwo': covRadius[2],
            'covRadius dThree': covRadius[3],
            'valence dZero': valence[0],
            'valence dOne': valence[1],
            'valence dTwo': valence[2],
            'valence dThree': valence[3]
        }

        RACs_result.append(dict_RACs)

    except:

        dict_RACs = {
            'cid': cid,
            'mass dZero': np.nan,
            'mass dOne': np.nan,
            'mass dTwo': np.nan,
            'mass dThree': np.nan,
            'EN dZero': np.nan,
            'EN dOne': np.nan,
            'EN dTwo': np.nan,
            'EN dThree': np.nan,
            'In dZero': np.nan,
            'In dOne': np.nan,
            'In dTwo': np.nan,
            'In dThree': np.nan,
            'aRadius dZero': np.nan,
            'aRadius dOne': np.nan,
            'aRadius dTwo': np.nan,
            'aRadius dThree': np.nan,
            'VdW dZero': np.nan,
            'VdW dOne': np.nan,
            'VdW dTwo': np.nan,
            'VdW dThree': np.nan,
            'covRadius dZero': np.nan,
            'covRadius dOne': np.nan,
            'covRadius dTwo': np.nan,
            'covRadius dThree': np.nan,
            'valence dZero': np.nan,
            'valence dOne': np.nan,
            'valence dTwo': np.nan,
            'valence dThree': np.nan
        }

        RACs_result.append(dict_RACs)


RACs = pd.DataFrame(RACs_result)
RACs.to_csv('kulik.csv')
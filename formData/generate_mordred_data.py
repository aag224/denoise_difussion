# Modelo: XGBoost
# Propiedad: E_ST
# Descriptores: Mordredx|


import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from mordred.error import Error, Missing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

data = pd.read_csv("C:\\Users\\mafo_\\Desktop\\mk_predictions\\SolutionDataIII.csv")
est = data["raw_value"]
molecules = data["name.smiles"]
print(est, molecules)

# ****************************************************************************************************** #
# ----------------------------------    Generate descriptors matrix    --------------------------------- #
#_______________________________________________________________________________________________________ #


# Descomentar y primero ejecutar esto
"""
Calculator = Calculator(descriptors, ignore_3D=True)
def smiles_to_matrix_descriptors(smiles):
    list_smile = [Chem.MolFromSmiles(smi) for smi in smiles]
    descriptor = Calculator.pandas(list_smile, nproc=3)
    descriptor = pd.DataFrame(descriptor)
    descrip_cl = descriptor.map(lambda x: np.nan if isinstance(x, (Error, Missing)) else x)
    descrip_cl = descrip_cl.dropna(axis=1, how="all")
    descrip_cl = descrip_cl.dropna(axis=1, thresh=len(descriptor)*0.8)
    #features_names = descrip_cl.columns.tolist()
    x = descrip_cl.to_numpy()

    return x

if __name__ == "__main__":
    tadf_mtrx = smiles_to_matrix_descriptors(molecules)
    print(tadf_mtrx)
    data_descriptors = pd.DataFrame(tadf_mtrx)
    data_descriptors.to_excel("tadf_mordredComplete.xlsx", index=False)

"""

# Despues ejecutar esto
tadf_mordred = pd.read_excel("C:\\Users\\mafo_\\Desktop\\mk_predictions\\tadf_mordredComplete.xlsx")


selector = VarianceThreshold(threshold=0.01)
tadf_clean = selector.fit_transform(tadf_mordred)
cols_save = tadf_mordred.columns[selector.get_support()]
tadf_filtered = pd.DataFrame(tadf_clean, columns=cols_save)

# Drop highly correlated features
corr_matrix = tadf_filtered.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

tadf_final = tadf_filtered.drop(columns=to_drop)
print(f"Descriptores reducidos de {tadf_mordred.shape[1]} a {tadf_final.shape[1]}")
tadf_final.fillna(0)

smiles_series = pd.Series(molecules, name="SMILES").reset_index(drop=True)
tadf_final = tadf_final.reset_index(drop=True)

tadf_final = pd.concat([smiles_series, tadf_final], axis=1)

est_series = pd.Series(est, name="raw_value").reset_index(drop=True)
tadf_final = pd.concat([tadf_final, est_series], axis=1)
                       
tadf_final.to_excel("tadf_mordredComplete_filtered.xlsx", index=False)

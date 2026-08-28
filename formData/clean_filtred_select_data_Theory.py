import rdkit
import pandas as pd
import numpy as np

# ******************************************************************************************************** #
# ------------------------------------ GENERATE EXPERIMENTAL DATASET ------------------------------------- #
# ________________________________________________________________________________________________________ #

data = pd.read_csv(r"C:\\Users\\mafo_\\Desktop\\mk_predictions\\trying_one\\singlet-tripletEnergy.csv",
                   encoding='latin1')
#print(data.head())

exp_est = data[data["experimental"] == 0]
print(exp_est)

exp_est.to_csv("OnlyTeoData.csv", index=False)
ss = exp_est["state"].unique()
print(ss)
molecule_in_calc = exp_est
print(molecule_in_calc)
print(molecule_in_calc["state"].unique())

smile = pd.read_csv(r"C:\\Users\\mafo_\\Desktop\\mk_predictions\\trying_one\\smilesMolecules.csv",
                   encoding='latin1')
print(smile)

col_with_came = molecule_in_calc.columns[12] 
bring_column = ['raw_value','raw_units','experimental', 'state']
bring_column.append(col_with_came)

# Join the two datasets based on the 'nickname' column
merge_smiles = pd.merge(
    smile,
    molecule_in_calc[['nickname']+bring_column],
    left_on = 'nickname',
    right_on = 'nickname',
    how = 'inner'
)
merge_name = pd.merge(
    smile,
    molecule_in_calc[['nickname']+bring_column],
    left_on = 'name.compound',
    right_on = 'nickname',
    how = 'inner'
)
#print(merge_smiles)
#print(merge_name)

# Drop duplicates and unnecessary columns, then save the final dataset
df_finil = pd.concat([merge_smiles, merge_name], ignore_index=True)
df = df_finil.drop(columns=['Unnamed: 0','nickname_x','nickname_y'])
print(df_finil)
df.to_csv("TheoryData.csv", index=False)

# %%
import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc
# %%
df = pd.read_csv("archive/train.csv")

# %%

df.head()
df.info()
df.isna().sum()
df.isnull().sum()
# %%
new_columns = df["Description"].str.split(" ", expand=True)

# %%
df["prefix"] = df["Description"].str.extract(r'^([A-Za-z]+)')

# %%
df = pd.concat([df, new_columns], axis=1)

# %%
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].str.replace(",", "")

# %%
features = df[['Symbol','GeneType','GeneGroupMethod','prefix',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
target = df['NucleotideSequence']

# %%
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
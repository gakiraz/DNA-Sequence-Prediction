# %%
import pandas as pd
import matplotlib as plt
import nltk
import numpy as np

# %%

df = pd.read_csv("archive/train.csv")

# %%

df.head()
df.info()
df.isna().sum()

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
df.columns
# %%
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]):
        print(df[col].value_counts().head(20))
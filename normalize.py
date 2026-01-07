import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset.csv")

ids = df["id"]
X = df.drop(columns=["id"])

scaler = StandardScaler()
Xz = scaler.fit_transform(X)

df_z = pd.DataFrame(Xz, columns=X.columns)
df_z.insert(0, "id", ids)

df_z.to_csv("normalized_dataset.csv", index=False)
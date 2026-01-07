import pandas as pd
from sklearn.decomposition import PCA

# Cargar CSV
df = pd.read_csv("normalized_dataset.csv")

# Separar ID y features
ids = df["id"]
X = df.drop(columns=["id"]).values

# PCA (99% de varianza)
PERCENTAGE=0.80
pca = PCA(n_components=PERCENTAGE)
X_pca = pca.fit_transform(X)

# Crear nombres de columnas para PCA
pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]

# Construir nuevo DataFrame
df_pca = pd.DataFrame(X_pca, columns=pca_columns)
df_pca.insert(0, "id", ids)

# Guardar a CSV
df_pca.to_csv(f"pca_{PERCENTAGE}.csv", index=False)

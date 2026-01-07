import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
plt.rcParams["figure.figsize"] = (10, 6)

PLOT_DIR = "plot"
os.makedirs(PLOT_DIR, exist_ok=True)

kd_df = pd.read_csv("resultados_experimentos_kdtree.csv")
vp_df = pd.read_csv("resultados_experimentos_vptree.csv")

kd_df["arbol"] = "KD-Tree"
vp_df["arbol"] = "VP-Tree"

data = pd.concat([kd_df, vp_df], ignore_index=True)

K = 1
DIM_FIJA = data["dimensiones"].min()
N_FIJO = data["datos_entrenamiento"].max()

data_dim = data[(data["k_vecinos"] == K) & (data["datos_entrenamiento"] == N_FIJO)]

data_size = data[(data["k_vecinos"] == K) & (data["dimensiones"] == DIM_FIJA)]


def plot_kd_vs_vp(kd, vp, x, y, title, ylabel, filename):
    df = pd.concat([kd, vp])

    plt.figure()
    sns.lineplot(data=df, x=x, y=y, hue="arbol", marker="o")

    plt.title(title)
    plt.xlabel(x.capitalize())
    plt.ylabel(ylabel)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


for tipo in ["balanceado", "desbalanceado"]:
    kd_dim = data_dim[
        (data_dim["arbol"] == "KD-Tree") & (data_dim["tipo_arbol"] == tipo)
    ]
    kd_size = data_size[
        (data_size["arbol"] == "KD-Tree") & (data_size["tipo_arbol"] == tipo)
    ]

    vp_dim = data_dim[data_dim["arbol"] == "VP-Tree"]
    vp_size = data_size[data_size["arbol"] == "VP-Tree"]

    # -------------------------
    # Tiempo NN vs dimensiones
    # -------------------------
    plot_kd_vs_vp(
        kd_dim,
        vp_dim,
        x="dimensiones",
        y="tiempo_busqueda_nn_promedio_ns",
        title=f"Tiempo NN vs dimensiones (KD {tipo} vs VP)",
        ylabel="Tiempo (ns)",
        filename=f"tiempo_nn_dim_kd_{tipo}_vs_vp.png",
    )

    # -------------------------
    # Tiempo kNN vs dimensiones
    # -------------------------
    plot_kd_vs_vp(
        kd_dim,
        vp_dim,
        x="dimensiones",
        y="tiempo_busqueda_knn_promedio_ns",
        title=f"Tiempo kNN vs dimensiones (KD {tipo} vs VP)",
        ylabel="Tiempo (ns)",
        filename=f"tiempo_knn_dim_kd_{tipo}_vs_vp.png",
    )

    # -------------------------
    # Profundidad vs tamaño
    # -------------------------
    plot_kd_vs_vp(
        kd_size,
        vp_size,
        x="datos_entrenamiento",
        y="profundidad_arbol",
        title=f"Profundidad vs tamaño (KD {tipo} vs VP)",
        ylabel="Profundidad",
        filename=f"profundidad_size_kd_{tipo}_vs_vp.png",
    )

    # -------------------------
    # Uso de memoria vs tamaño
    # -------------------------
    plot_kd_vs_vp(
        kd_size,
        vp_size,
        x="datos_entrenamiento",
        y="memoria_estimada_kb",
        title=f"Uso de memoria vs tamaño (KD {tipo} vs VP)",
        ylabel="Memoria (KB)",
        filename=f"memoria_size_kd_{tipo}_vs_vp.png",
    )

    # -------------------------
    # Tiempo de construcción vs tamaño
    # -------------------------
    plot_kd_vs_vp(
        kd_size,
        vp_size,
        x="datos_entrenamiento",
        y="tiempo_construccion_ns",
        title=f"Tiempo de construcción vs tamaño (KD {tipo} vs VP)",
        ylabel="Tiempo (ns)",
        filename=f"construccion_size_kd_{tipo}_vs_vp.png",
    )

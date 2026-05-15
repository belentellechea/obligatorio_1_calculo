import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import f

IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

def suma_inferior(N):
    x = np.linspace(-1, 1, N + 1) # cantidad de subintervalos
    dx = x[1] - x[0]
    mins = np.minimum(f(x[:-1]), f(x[1:]))
    return np.sum(mins * dx)

def suma_superior(N):
    x = np.linspace(-1, 1, N + 1)
    dx = x[1] - x[0]
    maxs = np.maximum(f(x[:-1]), f(x[1:]))
    return np.sum(maxs * dx)

# --- Tablas ---

def crear_tabla(inicio, fin, paso):
    datos = []

    for N in range(inicio, fin + 1, paso):

        inf = suma_inferior(N)
        sup = suma_superior(N)

        error_inf = abs(inf - np.pi)
        error_sup = abs(sup - np.pi)

        datos.append([
            N,
            inf,
            error_inf,
            sup,
            error_sup
        ])

    tabla = pd.DataFrame(datos, columns=[
        "N",
        "Suma Inferior",
        "Residuo Inferior",
        "Suma Superior",
        "Residuo Superior"
    ])

    return tabla

print("\n================ TABLA 1 ================")
tabla1 = crear_tabla(10, 100, 10)
print(tabla1)

print("\n================ TABLA 2 ================")
tabla2 = crear_tabla(100, 1000, 100)
print(tabla2)

print("\n================ TABLA 3 ================")
tabla3 = crear_tabla(1000, 10000, 1000)
print(tabla3)

# --- Gráficas ---

N_vals = list(range(10, 10001, 10))
s_inf_vals = [suma_inferior(N) for N in N_vals]
s_sup_vals = [suma_superior(N) for N in N_vals]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Parte 1")

# sin zoom
ax = axes[0]
ax.axhline(np.pi, color="forestgreen", linestyle="--", linewidth=1.5, label=f"π")
ax.plot(N_vals, s_inf_vals, label="Suma inferior", color="royalblue", linewidth=1.2)
ax.plot(N_vals, s_sup_vals, label="Suma superior", color="mediumvioletred", linewidth=1.2)
ax.set_xlabel("N")
ax.set_ylabel("aproximación de π")
ax.set_title("gráfico — N: 10 - 10000")
ax.legend()
ax.grid(True, alpha=0.3)

# zoom
ax = axes[1]
N_zoom = list(range(100, 4001, 100))
ax.axhline(np.pi, color="forestgreen", linestyle="--", linewidth=1.5, label=f"π")
ax.plot(N_zoom, [suma_inferior(N) for N in N_zoom], label="Suma inferior", color="royalblue")
ax.plot(N_zoom, [suma_superior(N) for N in N_zoom], label="Suma superior", color="mediumvioletred")
ax.set_xlabel("N")
ax.set_ylabel("aproximación de π")
ax.set_title("gráfico con zoom — N: 100 - 4000")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "parte 1.png"))
plt.show()
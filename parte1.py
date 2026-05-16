import numpy as np
import matplotlib.pyplot as plt
from utils import (
    IMAGES_DIR,
    crear_dataframe,
    crear_directorio_imagenes,
    imprimir_tabla,
    suma_inferior,
    suma_superior
)
crear_directorio_imagenes()

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

    return crear_dataframe(datos, [
        "N",
        "Suma Inferior",
        "Residuo Inferior",
        "Suma Superior",
        "Residuo Superior"
    ])

imprimir_tabla(crear_tabla(10, 100, 10), ["N", "Suma Inferior", "Residuo Inferior", "Suma Superior", "Residuo Superior"], "1: N de 10 a 100 (paso 10)")
imprimir_tabla(crear_tabla(100, 1000, 100), ["N", "Suma Inferior", "Residuo Inferior", "Suma Superior", "Residuo Superior"], "2: N de 100 a 1000 (paso 100)")
imprimir_tabla(crear_tabla(1000, 10000, 1000), ["N", "Suma Inferior", "Residuo Inferior", "Suma Superior", "Residuo Superior"], "3: N de 1000 a 10000 (paso 1000)")

# --- Gráficas ---

N_vals = list(range(10, 10001, 10))
s_inf_vals = [suma_inferior(N) for N in N_vals]
s_sup_vals = [suma_superior(N) for N in N_vals]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Parte 1: Convergencia de sumas")

# sin zoom
ax = axes[0]
ax.axhline(np.pi, color="forestgreen", linestyle="--", linewidth=1.5, label=f"π teórico")
ax.plot(N_vals, s_inf_vals, label="Suma inferior", color="royalblue", linewidth=1.2)
ax.plot(N_vals, s_sup_vals, label="Suma superior", color="mediumvioletred", linewidth=1.2)
ax.set_xlabel("N")
ax.set_ylabel("Aproximación de π")
ax.set_title("Gráfico — N: 10 - 10000")
ax.legend()
ax.grid(True, alpha=0.3)

# con zoom
ax = axes[1]
N_zoom = list(range(100, 4001, 100))
ax.axhline(np.pi, color="forestgreen", linestyle="--", linewidth=1.5, label=f"π teórico")
ax.plot(N_zoom, [suma_inferior(N) for N in N_zoom], label="Suma inferior", color="royalblue")
ax.plot(N_zoom, [suma_superior(N) for N in N_zoom], label="Suma superior", color="mediumvioletred")
ax.set_xlabel("N")
ax.set_ylabel("Aproximación de π")
ax.set_title("Gráfico con zoom — N: 100 - 4000")
ax.legend()
ax.grid(True, alpha=0.3)

# Guardar y mostrar

plt.tight_layout()
plt.savefig(f"{IMAGES_DIR}/gráficas parte 1.png")
plt.show()
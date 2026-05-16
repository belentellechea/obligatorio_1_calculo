import numpy as np
import matplotlib.pyplot as plt
from utils import (
    IMAGES_DIR,
    crear_dataframe,
    crear_directorio_imagenes,
    imprimir_tabla,
    f,
    particion_aleatoria,
    particion_coseno,
    particion_equiespaciada,
    suma_riemann
)

crear_directorio_imagenes()

# --- Tablas ---

def crear_tabla(inicio, fin, paso):

    datos = []

    for N in range(inicio, fin + 1, paso):

        eq = suma_riemann(particion_equiespaciada(N))
        al = suma_riemann(particion_aleatoria(N))
        co = suma_riemann(particion_coseno(N))

        err_eq = abs(eq - np.pi)
        err_al = abs(al - np.pi)
        err_co = abs(co - np.pi)

        datos.append([
            N,
            eq, err_eq,
            al, err_al,
            co, err_co
        ])

    return crear_dataframe(datos, [
        "N",
        "Equiespaciada", "Residuo Eq.",
        "Aleatoria", "Residuo Aleatorio",
        "Cosenoidal", "Residuo Coseno"
    ])

headers = ["N",
            "Equiespaciada.", "Residuo Eq.",
            "Aleatoria", "Residuo Aleatorio",
            "Coseno", "Residuo Coseno"]

imprimir_tabla(crear_tabla(10, 100, 10), headers, "1: N de 10 a 100 (paso 10)")
imprimir_tabla(crear_tabla(100, 1000, 100), headers, "2: N de 100 a 1000 (paso 100)")
imprimir_tabla(crear_tabla(1000, 10000, 1000), headers, "3: N de 1000 a 10000 (paso 1000)")

# --- Gráfica de convergencia ---

N_vals = list(range(10, 5001, 20))
np.random.seed(42)

eq_vals   = [suma_riemann(particion_equiespaciada(N)) for N in N_vals]
cos_vals  = [suma_riemann(particion_coseno(N)) for N in N_vals]
rand_vals = [np.mean([suma_riemann(particion_aleatoria(N)) for _ in range(5)]) for N in N_vals]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Parte 2: Influencia de la partición")

for ax, title, xlim in zip(axes, ["Gráfico — N: 10 - 5000", "Gráfico con zoom — N: 100 - 1000"], [(10, 5000), (100, 1000)]):
    mask = (np.array(N_vals) >= xlim[0]) & (np.array(N_vals) <= xlim[1])
    Ns = np.array(N_vals)[mask]
    ax.plot(Ns, np.array(eq_vals)[mask], label="Equiespaciada", color="steelblue")
    ax.plot(Ns, np.array(rand_vals)[mask], label="Aleatoria", color="orange")
    ax.plot(Ns, np.array(cos_vals)[mask], label="Coseno", color="purple")
    ax.axhline(np.pi, color="forestgreen", linestyle="--", linewidth=1.5, label="π teórico")
    ax.set_xlabel("N")
    ax.set_ylabel("Aproximación de π")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{IMAGES_DIR}/gráficas parte 2 convergencia.png", dpi=150)
plt.show()

# --- Gráfica de rectángulos para N = 100 ---

def graficar_rectangulos(ax, particion_fn, N, titulo, color):
    puntos = np.sort(particion_fn(N))
    x_curva = np.linspace(-1, 1, 500)
    ax.plot(x_curva, f(x_curva), "k-", linewidth=1, label="f(x)")
    ax.axhline(0, color="k", linewidth=0.5)

    for i in range(len(puntos) - 1):
        x0, x1 = puntos[i], puntos[i+1]
        h = max(f(x0), f(x1))
        rect = plt.Rectangle((x0, 0), x1 - x0, h,
                            alpha=0.3, color=color, linewidth=0.3)
        ax.add_patch(rect)

    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.1, 2.3)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Parte 2: Rectángulos para N = 100")

np.random.seed(0)
graficar_rectangulos(axes[0], particion_equiespaciada, 100, "Partición equiespaciada", "steelblue")
graficar_rectangulos(axes[1], particion_aleatoria, 100, "Partición aleatoria", "orange")
graficar_rectangulos(axes[2], particion_coseno, 100, "Partición coseno", "purple")

# Guardar y mostrar

plt.tight_layout()
plt.savefig(f"{IMAGES_DIR}/gráficas parte 2 rectángulos.png", dpi=150)
plt.show()
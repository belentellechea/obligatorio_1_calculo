import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import math

def f(x):
    return 2 * np.sqrt(0, 1 - x**2)

def suma_riemann(puntos):
    puntos = np.sort(puntos)
    dx = np.diff(puntos)
    maxs = np.maximum(f(puntos[:-1]), f(puntos[1:]))
    return np.sum(maxs * dx)

# def suma_riemann(particion):
#     total = 0

#     for i in range(len(particion) - 1):
#         a = particion[i]
#         b = particion[i + 1]
#         dx = b - a
#         total += f(a) * dx

#     return total

def particion_equiespaciada(N):
    return np.linspace(-1, 1, N + 1)

def particion_aleatoria(N):
    puntos = np.random.uniform(-1, 1, N - 1)
    return np.sort(np.concatenate(([-1.0], puntos, [1.0])))

def particion_coseno(N):
    i = np.arange(N + 1)
    return np.sort(np.cos(i * np.pi / N))

# --- Tablas ---

def generar_tabla(valores_N, repeticiones_aleatorio=10):
    filas = []
    for N in valores_N:
        s_eq  = suma_riemann(particion_equiespaciada(N))
        s_cos = suma_riemann(particion_coseno(N))
        s_rand = np.mean([suma_riemann(particion_aleatoria(N)) for _ in range(repeticiones_aleatorio)])
        filas.append([
            N,
            round(s_eq,   8), round(s_eq   - np.pi, 8),
            round(s_rand, 8), round(s_rand - np.pi, 8),
            round(s_cos,  8), round(s_cos  - np.pi, 8),
        ])
    return filas

# def crear_tabla(inicio, fin, paso):

#     datos = []

#     for N in range(inicio, fin + 1, paso):

#         eq = suma_riemann(particion_equiespaciada(N))
#         al = suma_riemann(particion_aleatoria(N))
#         co = suma_riemann(particion_cosenoidal(N))

#         err_eq = abs(eq - math.pi)
#         err_al = abs(al - math.pi)
#         err_co = abs(co - math.pi)

#         datos.append([
#             N,
#             eq, err_eq,
#             al, err_al,
#             co, err_co
#         ])

#     tabla = pd.DataFrame(datos, columns=[
#         "N",
#         "Equiespaciada", "Residuo Eq",
#         "Aleatoria", "Residuo Aleatorio",
#         "Cosenoidal", "Residuo Coseno"
#     ])

#     return tabla

headers = ["N",
            "Equiespaciada.", "Residuo eq.",
            "Aleatoria", "Residuo random",
            "Coseno", "Residuo coseno"]

print("=" * 90)
print("================ TABLA 1 =================")
print("=" * 90)
print(tabulate(generar_tabla(range(10, 101, 10)), headers=headers, floatfmt=".8f"))

print("\n" + "=" * 90)
print("================ TABLA 2 =================")
print("=" * 90)
print(tabulate(generar_tabla(range(100, 1001, 100)), headers=headers, floatfmt=".8f"))

print("\n" + "=" * 90)
print("================ TABLA 3 =================")
print("=" * 90)
print(tabulate(generar_tabla(range(1000, 10001, 1000)), headers=headers, floatfmt=".8f"))

# --- Gráfica de convergencia ---

N_vals = list(range(10, 5001, 20))
np.random.seed(42)

eq_vals   = [suma_riemann(particion_equiespaciada(N)) for N in N_vals]
cos_vals  = [suma_riemann(particion_coseno(N))        for N in N_vals]
rand_vals = [np.mean([suma_riemann(particion_aleatoria(N)) for _ in range(5)]) for N in N_vals]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Parte 2")

for ax, title, xlim in zip(axes,
                            ["N: 10 → 5000", "Zoom"],
                            [(10, 5000), (100, 1000)]):
    mask = (np.array(N_vals) >= xlim[0]) & (np.array(N_vals) <= xlim[1])
    Ns = np.array(N_vals)[mask]
    ax.plot(Ns, np.array(eq_vals)[mask],   label="Equiespaciada", color="steelblue")
    ax.plot(Ns, np.array(rand_vals)[mask], label="Aleatoria",     color="orange", alpha=0.8)
    ax.plot(Ns, np.array(cos_vals)[mask],  label="Coseno",        color="purple", linewidth=2)
    ax.axhline(np.pi, color="green", linestyle="--", linewidth=1.5, label="π teórico")
    ax.set_xlabel("N"); ax.set_ylabel("Aproximación de π")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gráficas parte 2 convergencia.png", dpi=150)
plt.show()

# --- Gráfica de rectángulos para N=100 ---

def graficar_rectangulos(ax, particion_fn, N, titulo, color):
    puntos = np.sort(particion_fn(N))
    x_curva = np.linspace(-1, 1, 500)
    ax.plot(x_curva, f(x_curva), "k-", linewidth=2, label="f(x)")
    ax.axhline(0, color="k", linewidth=0.5)

    for i in range(len(puntos) - 1):
        x0, x1 = puntos[i], puntos[i+1]
        h = max(f(x0), f(x1))
        rect = plt.Rectangle((x0, 0), x1 - x0, h,
                            alpha=0.3, color=color, edgecolor=color, linewidth=0.3)
        ax.add_patch(rect)

    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.1, 2.3)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Parte 2 - Rectángulos de aproximación (N = 100)")

np.random.seed(0)
graficar_rectangulos(axes[0], particion_equiespaciada, 100, "Partición equiespaciada", "steelblue")
graficar_rectangulos(axes[1], particion_aleatoria,     100, "Partición aleatoria",     "orange")
graficar_rectangulos(axes[2], particion_coseno,        100, "Partición coseno",        "purple")

plt.tight_layout()
plt.savefig("gráficas parte 2.png", dpi=150)
plt.show()
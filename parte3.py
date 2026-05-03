import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from utils import f 

def particion_equiespaciada(N):
    x = np.linspace(-1, 1, N + 1)
    dx = x[1] - x[0]
    return x, dx

def metodo_rectangulos(N):
    x, dx = particion_equiespaciada(N)
    return np.sum(f(x[:-1]) * dx)


def metodo_trapecio(N):
    x, dx = particion_equiespaciada(N)
    return np.sum((f(x[:-1]) + f(x[1:])) / 2 * dx)


def metodo_punto_medio(N):
    x, dx = particion_equiespaciada(N)
    midpoints = (x[:-1] + x[1:]) / 2
    return np.sum(f(midpoints) * dx)


#tablas
def generar_tabla(valores_N):
    filas = []
    for N in valores_N:
        r  = metodo_rectangulos(N)
        t  = metodo_trapecio(N)
        pm = metodo_punto_medio(N)
        filas.append([
            N,
            round(r,  8), round(r  - np.pi, 8),
            round(t,  8), round(t  - np.pi, 8),
            round(pm, 8), round(pm - np.pi, 8),
        ])
    return filas

headers = ["N",
            "Rectángulos", "Res. Rect.",
            "Trapecio",    "Res. Trap.",
            "Pto. Medio",  "Res. P.M."]

print("=" * 90)
print("TABLA 1: N de 10 a 100 (paso 10)")
print("=" * 90)
print(tabulate(generar_tabla(range(10, 101, 10)), headers=headers, floatfmt=".8f"))

print("\n" + "=" * 90)
print("TABLA 2: N de 100 a 1000 (paso 100)")
print("=" * 90)
print(tabulate(generar_tabla(range(100, 1001, 100)), headers=headers, floatfmt=".8f"))

print("\n" + "=" * 90)
print("TABLA 3: N de 1000 a 10000 (paso 1000)")
print("=" * 90)
print(tabulate(generar_tabla(range(1000, 10001, 1000)), headers=headers, floatfmt=".8f"))


#graficas
N_vals = list(range(10, 5001, 10))

r_vals  = [metodo_rectangulos(N)  for N in N_vals]
t_vals  = [metodo_trapecio(N)     for N in N_vals]
pm_vals = [metodo_punto_medio(N)  for N in N_vals]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Comparación de métodos", fontsize=14, fontweight="bold")

for ax, title, xlim in zip(axes,
                            ["N: 10 → 5000", "Zoom"],
                            [(10, 5000), (10, 500)]):
    mask = (np.array(N_vals) >= xlim[0]) & (np.array(N_vals) <= xlim[1])
    Ns = np.array(N_vals)[mask]
    ax.plot(Ns, np.array(r_vals)[mask],  label="Rectángulos", color="steelblue")
    ax.plot(Ns, np.array(t_vals)[mask],  label="Trapecio",    color="tomato")
    ax.plot(Ns, np.array(pm_vals)[mask], label="Punto medio",  color="purple")
    ax.axhline(np.pi, color="green", linestyle="--", linewidth=1.5, label="π")
    ax.set_xlabel("N"); ax.set_ylabel("Aproximación de π")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graficas_parte3.png", dpi=150)
plt.show()

N_demo = 8
x_demo = np.linspace(-1, 1, N_demo + 1)
dx     = x_demo[1] - x_demo[0]
x_curva = np.linspace(-1, 1, 400)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Sección 3 - Ilustración geométrica de los métodos (N = 8)",
             fontsize=13, fontweight="bold")

titulos = ["Rectángulos (extremo izq.)", "Trapecio", "Punto medio"]
colores = ["steelblue", "tomato", "purple"]

for idx, (ax, titulo, color) in enumerate(zip(axes, titulos, colores)):
    ax.plot(x_curva, f(x_curva), "k-", linewidth=2, label="f(x)", zorder=5)
    ax.axhline(0, color="k", linewidth=0.5)

    for i in range(N_demo):
        x0, x1 = x_demo[i], x_demo[i+1]

        #rectangulos
        if idx == 0:  
            h = f(x0)
            rect = plt.Rectangle((x0, 0), dx, h,
                                  alpha=0.35, color=color, edgecolor=color)
            ax.add_patch(rect)

        #trapecio
        elif idx == 1:  
            ax.fill_between([x0, x1], [0, 0], [f(x0), f(x1)],
                             alpha=0.35, color=color, edgecolor=color, linewidth=0.8)

        #punto medio
        else:           
            xm = (x0 + x1) / 2
            rect = plt.Rectangle((x0, 0), dx, f(xm),
                                  alpha=0.35, color=color, edgecolor=color)
            ax.add_patch(rect)

    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.1, 2.3)
    ax.grid(True, alpha=0.3)
    ax.legend(["f(x)"])

plt.tight_layout()
plt.savefig("gráficas parte 3 ilustracion.png", dpi=150)
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    IMAGES_DIR,
    crear_directorio_imagenes,
    imprimir_tabla,
)

crear_directorio_imagenes()

# --- Método de Montecarlo ---

def monte_carlo_pi(N, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, N)
    y = rng.uniform(0, 1, N)
    dentro = (x**2 + y**2) <= 1.0
    return 4 * np.sum(dentro) / N

# --- Tablas ---

def generar_tabla(valores_N, repeticiones=10):
    filas = []
    for N in valores_N:
        estimaciones = [monte_carlo_pi(N) for _ in range(repeticiones)]
        media = np.mean(estimaciones)
        std = np.std(estimaciones)
        filas.append([N, round(media, 8), round(media - np.pi, 8), round(std, 8)])
    return filas

headers = ["N", "π estimado (media)", "Residuo", "Desv. Estándar"]

imprimir_tabla(generar_tabla(range(10, 101, 10)), headers, "1: N de 10 a 100 (paso 10)")
imprimir_tabla(generar_tabla(range(100, 1001, 100)), headers, "2: N de 100 a 1000 (paso 100)")
imprimir_tabla(generar_tabla(range(1000, 10001, 1000)), headers, "3: N de 1000 a 10000 (paso 1000)")

# --- Gráfica de convergencia y error ---

np.random.seed(42)
N_vals = [int(n) for n in np.logspace(1, 5, 200)]

estimaciones = [monte_carlo_pi(N) for N in N_vals]

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Parte 4: Integración de Montecarlo")

ax.semilogx(N_vals, estimaciones, color="darkorange", alpha=0.7, linewidth=1, label="Estimación Monte Carlo")
ax.axhline(np.pi, color="green", linestyle="--", linewidth=2, label=f"π = {np.pi:.6f}")
ax.fill_between(N_vals,
                np.pi - 1/np.sqrt(np.array(N_vals)),
                np.pi + 1/np.sqrt(np.array(N_vals)),
                alpha=0.2, color="green", label="±1/√N (banda teórica)")
ax.set_xlabel("N (escala logarítmica)")
ax.set_ylabel("Estimación de π")
ax.set_title("Convergencia (escala log)")
ax.legend()
ax.grid(True, alpha=0.3)

# Guardar y mostrar

plt.tight_layout()
plt.savefig(f"{IMAGES_DIR}/gráficas parte 4 convergencia.png", dpi=150)
plt.show()

# --- Visualización clásica del método ---

rng = np.random.default_rng(7)
N_vis = 5000
x = rng.uniform(0, 1, N_vis)
y = rng.uniform(0, 1, N_vis)
dentro = (x**2 + y**2) <= 1

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x[ dentro], y[ dentro], s=1, color="steelblue", alpha=0.5, label="Dentro")
ax.scatter(x[~dentro], y[~dentro], s=1, color="tomato",    alpha=0.5, label="Fuera")
theta = np.linspace(0, np.pi/2, 300)
ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
ax.set_aspect("equal")
ax.set_title(f"Monte Carlo: (N={N_vis}): π ≈ {4*np.sum(dentro)/N_vis:.5f}", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.tight_layout()
plt.savefig(f"{IMAGES_DIR}/gráficas parte 4 visualizacion.png", dpi=150)
plt.show()
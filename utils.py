import os
import numpy as np
import pandas as pd
from tabulate import tabulate

def f(x):
    return 2 * np.sqrt(1 - x**2)

IMAGES_DIR = "images"

def crear_directorio_imagenes():
    os.makedirs(IMAGES_DIR, exist_ok=True)

def imprimir_tabla(tabla, headers, nombre, floatfmt=".8f"):
    print(f"\n" + "=" * 50 + f" TABLA {nombre} " + "=" * 50)
    print(tabulate(tabla, headers=headers, floatfmt=floatfmt))
    print("\n")

def particion_equiespaciada(N):
    return np.linspace(-1, 1, N + 1)

def suma_extrema(N, funcion_extrema):
    x = particion_equiespaciada(N)
    dx = x[1] - x[0]
    extremos = funcion_extrema(f(x[:-1]), f(x[1:]))
    return np.sum(extremos * dx)

def suma_inferior(N):
    return suma_extrema(N, np.minimum)

def suma_superior(N):
    return suma_extrema(N, np.maximum)

def suma_riemann(puntos):
    puntos = np.sort(puntos)
    dx = np.diff(puntos)
    maxs = np.maximum(f(puntos[:-1]), f(puntos[1:]))
    return np.sum(maxs * dx)

def particion_aleatoria(N, seed=None):
    rng = np.random.default_rng(seed)
    puntos = rng.uniform(-1, 1, N - 1)
    return np.sort(np.concatenate(([-1.0], puntos, [1.0])))

def particion_coseno(N):
    i = np.arange(N + 1)
    return np.sort(np.cos(i * np.pi / N))

def particion_equiespaciada_dx(N):
    x = particion_equiespaciada(N)
    dx = x[1] - x[0]
    return x, dx

def crear_dataframe(datos, columnas):
    return pd.DataFrame(datos, columns=columnas)
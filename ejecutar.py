# -*- coding: utf-8 -*-
"""Ejecuta las celdas de celdas.py y genera notebook_proyecto.ipynb con las salidas incluidas."""
import ast
import base64
import io
import json
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from celdas import CELDAS  # noqa: E402


def lineas(texto):
    partes = texto.split("\n")
    return [p + "\n" for p in partes[:-1]] + [partes[-1]]


class Captura:
    def __init__(self):
        self.buf = []

    def write(self, s):
        self.buf.append(s)
        return len(s)

    def flush(self):
        pass

    def vaciar(self):
        t = "".join(self.buf)
        self.buf = []
        return t


ns = {"__name__": "__main__"}
celdas_nb = []
n_exec = 0
captura = Captura()
salidas_actuales = []


def volcar_stdout():
    t = captura.vaciar()
    if t:
        salidas_actuales.append({"output_type": "stream", "name": "stdout", "text": lineas(t)})


def show_parcheado(*a, **k):
    volcar_stdout()
    for num in plt.get_fignums():
        fig = plt.figure(num)
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=110, bbox_inches="tight")
        salidas_actuales.append({
            "output_type": "display_data",
            "data": {"image/png": base64.b64encode(bio.getvalue()).decode(), "text/plain": ["<Figure>"]},
            "metadata": {},
        })
    plt.close("all")


plt.show = show_parcheado

for tipo, fuente in CELDAS:
    if tipo == "md":
        celdas_nb.append({"cell_type": "markdown", "metadata": {}, "source": lineas(fuente)})
        continue
    n_exec += 1
    salidas_actuales = []
    arbol = ast.parse(fuente)
    ultima = None
    if arbol.body and isinstance(arbol.body[-1], ast.Expr):
        ultima = ast.Expression(arbol.body.pop().value)
    stdout_original = sys.stdout
    sys.stdout = captura
    try:
        exec(compile(arbol, "<celda>", "exec"), ns)
        resultado = eval(compile(ultima, "<celda>", "eval"), ns) if ultima is not None else None
        volcar_stdout()
        if resultado is not None:
            datos = {"text/plain": lineas(repr(resultado))}
            if hasattr(resultado, "to_html"):
                datos["text/html"] = lineas(resultado.to_html(max_rows=60, max_cols=30))
                datos["text/plain"] = lineas(resultado.to_string(max_rows=60, max_cols=30))
            salidas_actuales.append({"output_type": "execute_result", "execution_count": n_exec, "data": datos, "metadata": {}})
    except Exception:
        sys.stdout = stdout_original
        print("ERROR en la celda", n_exec)
        print(fuente[:400])
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout = stdout_original
    # copia legible para el log
    for o in salidas_actuales:
        if o["output_type"] == "stream":
            print("".join(o["text"]), end="")
        elif o["output_type"] == "execute_result":
            print("".join(o["data"]["text/plain"]))
        else:
            print("[figura]")
    print(f"----- fin celda {n_exec} -----")
    celdas_nb.append({
        "cell_type": "code", "metadata": {}, "execution_count": n_exec,
        "source": lineas(fuente), "outputs": salidas_actuales,
    })

nb = {
    "cells": celdas_nb,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
with open("notebook_proyecto.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook escrito.")

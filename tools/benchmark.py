"""Mede executáveis Barenn com a mesma entrada, alternando a ordem das versões."""

import argparse
import re
import statistics
import subprocess
import time
from pathlib import Path


TEMPOS = {
    "treino": "Tempo de treino",
    "avaliacao": "Tempo de avaliacao",
    "gpu_h2d": "CUDA H2D",
    "gpu_kernel": "CUDA kernels",
    "gpu_d2h": "CUDA D2H",
}


def medir(executavel, treino, teste):
    inicio = time.perf_counter()
    saida = subprocess.run(
        [str(executavel), str(treino), str(teste)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    ).stdout
    medidas = {"total": time.perf_counter() - inicio}
    for chave, texto in TEMPOS.items():
        linha = re.search(rf"^{re.escape(texto)}\s*:?\s+([\d.]+)\s+s$", saida, re.M)
        if linha:
            medidas[chave] = float(linha[1])
    if "treino" not in medidas:
        raise ValueError(f"{executavel} não imprimiu o tempo de treino")
    return medidas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executaveis", type=Path, nargs="+")
    parser.add_argument("--repeticoes", type=int, default=7)
    parser.add_argument("--treino", type=Path, default=Path("optdigits.tra"))
    parser.add_argument("--teste", type=Path, default=Path("optdigits.tes"))
    args = parser.parse_args()
    if args.repeticoes < 2:
        parser.error("use pelo menos duas repetições")

    executaveis = [caminho.resolve() for caminho in args.executaveis]
    treino, teste = args.treino.resolve(), args.teste.resolve()
    medidas = [[] for _ in executaveis]
    for executavel in executaveis:
        medir(executavel, treino, teste)
    for rodada in range(args.repeticoes):
        for posicao in range(len(executaveis)):
            indice = (rodada + posicao) % len(executaveis)
            medidas[indice].append(medir(executaveis[indice], treino, teste))

    print(f"{args.repeticoes} repetições após um aquecimento por executável.")
    for executavel, rodadas in zip(executaveis, medidas):
        print(f"\n{executavel.name} (segundos)")
        for metrica in rodadas[0]:
            valores = [rodada[metrica] for rodada in rodadas]
            mediana = statistics.median(valores)
            desvio = statistics.stdev(valores)
            amostras = ", ".join(f"{valor:.6f}" for valor in valores)
            print(f"{metrica:>11} | mediana {mediana:.6f} | DP {desvio:.6f} | {amostras}")


if __name__ == "__main__":
    main()

"""Compara as mesmas entradas e o mesmo binário com uma e quatro threads."""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def executar(executavel, arquivos, threads):
    ambiente = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE")
    inicio = time.perf_counter()
    saida = subprocess.run(
        [str(executavel), *(str(p) for p in arquivos)],
        env=ambiente, capture_output=True, encoding="utf-8", check=True, timeout=120,
    ).stdout
    tempos = {"processo": time.perf_counter() - inicio}
    linhas_tempo = r"^Tempo de (.+?)\s*:?\s+([0-9.]+) s$"
    for fase, valor in re.findall(linhas_tempo, saida, flags=re.MULTILINE):
        tempos[fase] = float(valor)
    numerica = re.sub(linhas_tempo + r"\n?", "", saida, flags=re.MULTILINE)
    return tempos, numerica


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("executavel", type=Path)
    cli.add_argument("--treino", type=Path, default=Path("optdigits.tra"))
    cli.add_argument("--teste", type=Path, default=Path("optdigits.tes"))
    cli.add_argument("--rodadas", type=int, default=9)
    args = cli.parse_args()
    if args.rodadas < 2:
        cli.error("use pelo menos duas rodadas")
    arquivos = (args.treino.resolve(strict=True), args.teste.resolve(strict=True))
    exe = args.executavel.resolve(strict=True)
    dados = {1: defaultdict(list), 4: defaultdict(list)}
    referencia = None
    for rodada in range(args.rodadas + 1):
        for threads in ((1, 4) if rodada % 2 == 0 else (4, 1)):
            tempos, numerica = executar(exe, arquivos, threads)
            if referencia is None:
                referencia = numerica
            if numerica != referencia:
                cli.exit(1, f"Saída divergente com {threads} threads na rodada {rodada}.\n")
            if rodada:
                for fase, valor in tempos.items():
                    dados[threads][fase].append(valor)

    print(f"{args.rodadas} rodadas intercaladas após um aquecimento por configuração")
    print(next(linha for linha in referencia.splitlines() if linha.startswith("Acuracia")))
    for fase, serial in dados[1].items():
        paralelo = dados[4][fase]
        for threads, amostras in ((1, serial), (4, paralelo)):
            valores = ", ".join(f"{v:.6f}" for v in amostras)
            print(f"{fase} com {threads} thread(s) | {statistics.median(amostras):.6f} s"
                  f" | DP {statistics.stdev(amostras):.6f} s | {valores}")
        if min(statistics.median(serial), statistics.median(paralelo)) > 0:
            print(f"{fase}: {statistics.median(serial) / statistics.median(paralelo):.3f}x")


if __name__ == "__main__":
    main()

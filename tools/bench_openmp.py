"""Compara o cálculo de inferência da versão CPU do Barenn com 1 e 4 threads."""

import argparse
import os
from pathlib import Path
import re
import statistics
import subprocess
from tempfile import TemporaryDirectory


def instrumentar(codigo, passagens):
    """Reaproveita a equipe OpenMP e mantém o corpo de inferência do original."""
    inicio_avaliar = codigo.index("void avaliar(")
    inicio = codigo.index("    const auto inicio = chrono::steady_clock::now();", inicio_avaliar)
    fim = codigo.index("    const chrono::duration<double> tempo", inicio)
    bloco = codigo[inicio:fim]
    pragma = "#pragma omp parallel for num_threads(threads) schedule(static)"
    if bloco.count(pragma) != 1:
        raise ValueError("Use o barenn.cpp da versão CPU com o laço de inferência OpenMP.")
    relogio, laco = bloco.split("\n", 1)
    laco = laco.replace(pragma, "#pragma omp for schedule(static)")
    laco = "".join("        " + linha if linha.strip() else linha
                   for linha in laco.splitlines(keepends=True))
    repeticao = (
        "    int equipe = 0;\n"
        f"{relogio}\n"
        "    #pragma omp parallel num_threads(threads)\n"
        "    {\n"
        "        if (omp_get_thread_num() == 0) {\n"
        "            equipe = omp_get_num_threads();\n"
        "        }\n"
        f"        for (int passagem = 0; passagem < {passagens}; ++passagem) {{\n"
        f"{laco}"
        "        }\n"
        "    }\n"
    )
    posicao_relatorio = codigo.index('    cout << "\\nAVALIACAO FINAL', fim)
    return (
        codigo[:inicio] + repeticao + codigo[fim:posicao_relatorio]
        + '    cout << "Equipe benchmark " << equipe << \'\\n\';\n'
        + codigo[posicao_relatorio:]
    )


def medir(exe, treino, teste, threads, passagens):
    ambiente = os.environ.copy()
    ambiente.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE")
    saida = subprocess.run(
        [str(exe), str(treino), str(teste)],
        env=ambiente, capture_output=True, encoding="utf-8", check=True, timeout=300,
    ).stdout
    equipe = re.search(r"^Equipe benchmark (\d+)$", saida, re.MULTILINE)
    tempo = re.search(r"^Tempo de inferencia ([0-9.]+) s$", saida, re.MULTILINE)
    acuracia = re.search(r"^Acuracia .+$", saida, re.MULTILINE)
    if not equipe or int(equipe[1]) != threads or not tempo or not acuracia:
        raise ValueError(f"Saída inesperada na medição com {threads} threads.\n{saida}")
    if float(tempo[1]) == 0:
        raise ValueError("Tempo abaixo da resolução da saída. Aumente --passagens.")
    return float(tempo[1]) * 1000 / passagens, acuracia[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fonte", type=Path, help="barenn.cpp da versão CPU")
    parser.add_argument("--treino", type=Path, default=Path("optdigits.tra"))
    parser.add_argument("--teste", type=Path, default=Path("optdigits.tes"))
    parser.add_argument("--passagens", type=int, default=100)
    parser.add_argument("--rodadas", type=int, default=8)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    if args.passagens < 1 or args.rodadas < 2:
        parser.error("Use pelo menos uma passagem e duas rodadas.")

    flags = ["-std=c++17", "-O2", "-g", "-fopenmp"]
    if os.name == "nt":
        flags.append("-static")
    try:
        treino = args.treino.resolve(strict=True)
        teste = args.teste.resolve(strict=True)
        codigo = instrumentar(args.fonte.read_text(encoding="utf-8"), args.passagens)
        amostras = {1: [], 4: []}
        referencia = None
        print(f"Compilação {' '.join([args.cxx, *flags])}", flush=True)
        print(f"{args.passagens} passagens por execução, 1 aquecimento e "
              f"{args.rodadas} rodadas por equipe.", flush=True)
        with TemporaryDirectory(prefix="barenn-openmp-") as pasta:
            fonte = Path(pasta) / "barenn.cpp"
            exe = Path(pasta) / ("barenn.exe" if os.name == "nt" else "barenn")
            fonte.write_text(codigo, encoding="utf-8")
            subprocess.run([args.cxx, *flags, str(fonte), "-o", str(exe)], check=True)
            for rodada in range(args.rodadas + 1):
                ordem = (1, 4) if rodada % 2 == 0 else (4, 1)
                for threads in ordem:
                    tempo, acuracia = medir(exe, treino, teste, threads, args.passagens)
                    if referencia is None:
                        referencia = acuracia
                    if acuracia != referencia:
                        raise ValueError(f"Acurácia divergente com {threads} threads. {acuracia}")
                    if rodada:
                        amostras[threads].append(tempo)
    except (OSError, ValueError, subprocess.SubprocessError) as erro:
        parser.exit(1, f"{erro}\n")

    print(referencia)
    medianas = {}
    for threads, valores in amostras.items():
        medianas[threads] = statistics.median(valores)
        print(f"{threads} thread(s), ms por passagem")
        print("  " + ", ".join(f"{tempo:.4f}" for tempo in valores))
        print(f"  mediana {medianas[threads]:.4f} | DP {statistics.stdev(valores):.4f}")
    speedup = medianas[1] / medianas[4]
    print(f"Speedup {speedup:.2f}x | eficiência {100 * speedup / 4:.1f}%")


if __name__ == "__main__":
    main()

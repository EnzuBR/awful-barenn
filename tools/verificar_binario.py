"""Compara os resultados das seis tags da APS 2 em Linux ou WSL."""

import argparse
from hashlib import sha256
import math
import os
from pathlib import Path
import platform
import shlex
import struct
import subprocess
from tempfile import TemporaryDirectory


TAGS_CPU = ("v0.1-baseline", "v0.2-cpu")
TAGS_CUDA = ("v0.3-cuda", "v0.4-cuda-shared", "v0.5-cuda-padding", "v1.0-aps2")
SECOES = (("parâmetros", "=2778f"), ("perdas", "=51d"), ("escores", "=17970f"))

GRAVAR_MODELO = r"""
        ofstream diagnostico(argv[3], ios::binary);
        if (!diagnostico) {
            throw runtime_error("Falha ao abrir a saída binária de verificação");
        }
        for (const auto* camada : {&l1, &l2, &l3}) {
PESOS
            diagnostico.write(reinterpret_cast<const char*>(camada->vieses().data()),
                              camada->vieses().size() * sizeof(float));
        }
        diagnostico.write(reinterpret_cast<const char*>(perdas.data()),
                          perdas.size() * sizeof(double));
        Barenn::avaliar(l1, l2, l3, teste, diagnostico);
        diagnostico.close();
        if (!diagnostico) {
            throw runtime_error("Falha ao gravar a saída binária de verificação");
        }"""

PESOS_LEGADO = """            for (const auto& linha : camada->pesos()) {
                diagnostico.write(reinterpret_cast<const char*>(linha.data()),
                                  linha.size() * sizeof(float));
            }"""

PESOS_CONTIGUOS = """            diagnostico.write(reinterpret_cast<const char*>(camada->pesos().data()),
                              camada->pesos().size() * sizeof(float));"""


def substituir(codigo, trecho, novo):
    if codigo.count(trecho) != 1:
        raise ValueError(f"trecho de instrumentação ausente ou repetido: {trecho!r}")
    return codigo.replace(trecho, novo, 1)


def instrumentar(codigo, legado):
    codigo = substituir(codigo, "if (argc != 1 && argc != 3)", "if (argc != 4)")
    codigo = codigo.replace("argc == 3 ?", "argc == 4 ?")
    inicio = codigo.index("void avaliar(")
    fim = codigo.index("int main(", inicio)
    avaliacao = codigo[inicio:fim]
    avaliacao = substituir(avaliacao, "const vector<Amostra>& teste) {",
                           "const vector<Amostra>& teste, ostream& diagnostico) {")
    if legado:
        construtor = "public:\n    DenseLayer("
        getters = """public:
    const auto& pesos() const { return weights; }
    const auto& vieses() const { return bias; }

    DenseLayer("""
        codigo = substituir(codigo[:inicio], construtor, getters) + codigo[inicio:]
        inicio = codigo.index("void avaliar(")
        fim = codigo.index("int main(", inicio)
        calculo = "        auto escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));"
        avaliacao = substituir(avaliacao, calculo, calculo + """
        diagnostico.write(reinterpret_cast<const char*>(escores.data()),
                          escores.size() * sizeof(float));""")
    else:
        resultados = "    vector<Resultado> resultados(teste.size());"
        avaliacao = substituir(avaliacao, resultados, resultados + """
    vector<array<float, classes>> escores_teste(teste.size());""")
        calculo = "        l3.Forward(a2.data(), escores.data());"
        avaliacao = substituir(avaliacao, calculo, calculo + "\n        escores_teste[i] = escores;")
        soma = "    // Soma na ordem das amostras para manter as métricas e os exemplos."
        avaliacao = substituir(avaliacao, soma, """    for (const auto& escores : escores_teste) {
        diagnostico.write(reinterpret_cast<const char*>(escores.data()),
                          escores.size() * sizeof(float));
    }

""" + soma)
    codigo = codigo[:inicio] + avaliacao + codigo[fim:]
    pesos = PESOS_LEGADO if legado else PESOS_CONTIGUOS
    return substituir(codigo, "        Barenn::avaliar(l1, l2, l3, teste);",
                      GRAVAR_MODELO.strip("\n").replace("PESOS", pesos))


def extrair(repo, tag, arquivo):
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"refs/tags/{tag}:{arquivo}"], timeout=300,
    )


def rodar(comando, **kwargs):
    print(f"$ {shlex.join(comando)}", flush=True)
    return subprocess.run(comando, check=True, timeout=300, **kwargs)


def executar(tag, pasta, repo, dados, args, cuda):
    pasta.mkdir()
    fonte = pasta / ("barenn.cu" if cuda else "barenn.cpp")
    codigo = extrair(repo, tag, "barenn.cpp").decode("utf-8").replace("\r\n", "\n")
    fonte.write_text(instrumentar(codigo, tag == TAGS_CPU[0]), encoding="utf-8")
    executavel = pasta / "barenn"
    if cuda:
        (pasta / "barenn_cuda.cuh").write_bytes(extrair(repo, tag, "barenn_cuda.cuh"))
        comando = [args.nvcc, "-ccbin", args.cxx, "-std=c++17", "-O2", f"-arch={args.arch}",
                   "--fmad=false", "-Xcompiler=-ffp-contract=off,-fopenmp", "-DBARENN_CUDA"]
    else:
        comando = [args.cxx, "-std=c++17", "-O2", "-ffp-contract=off"]
        if tag != TAGS_CPU[0]:
            comando.append("-fopenmp")
    rodar([*comando, str(fonte), "-o", str(executavel)])
    saida = pasta / "resultado.bin"
    rodar([str(executavel), *(str(arquivo) for arquivo in dados), str(saida)],
          env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"})
    binario = saida.read_bytes()
    print(f"  SHA-256 do resultado ({len(binario)} bytes) {sha256(binario).hexdigest()}")
    return binario


def ler_secoes(saida):
    tamanho = sum(struct.calcsize(formato) for _, formato in SECOES)
    if len(saida) != tamanho:
        raise ValueError(f"saída com {len(saida)} bytes, esperados {tamanho}")
    offset = 0
    for nome, formato in SECOES:
        fim = offset + struct.calcsize(formato)
        bloco = saida[offset:fim]
        numeros = struct.unpack(formato, bloco)
        if not all(map(math.isfinite, numeros)):
            raise ValueError(f"valores não finitos em {nome}")
        yield nome, bloco, numeros
        offset = fim


def comparar(referencia, candidato):
    identicos = True
    for (nome, bytes_ref, ref), (_, bytes_atual, atual) in zip(referencia, candidato):
        if bytes_ref != bytes_atual:
            diferenca = max(abs(a - b) for a, b in zip(ref, atual))
            primeiro = next(i for i, (a, b) in enumerate(zip(bytes_ref, bytes_atual)) if a != b)
            print(f"  {nome}: primeiro byte diferente no índice {primeiro}; "
                  f"maior diferença absoluta {diferenca:.9g}.")
            identicos = False
    return identicos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fonte = parser.add_mutually_exclusive_group()
    fonte.add_argument("--repo", type=Path, default=Path.cwd(), help="pasta do repositório")
    fonte.add_argument("--clonar", metavar="URL", help="clona o repositório em uma pasta temporária")
    parser.add_argument("--cxx", default="g++", help="compilador C++ usado também pelo nvcc")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"),
                        help="caminho do compilador CUDA (padrão: NVCC ou nvcc no PATH)")
    parser.add_argument("--arch", default="native", help="arquitetura CUDA (exemplo: sm_86)")
    parser.add_argument("--somente-cpu", action="store_true",
                        help="valida as tags de CPU e o fallback CPU da tag final")
    args = parser.parse_args()
    repo = args.repo.resolve()
    versoes = [(tag, False) for tag in TAGS_CPU]
    if not args.somente_cpu:
        versoes.extend((tag, True) for tag in TAGS_CUDA)
    versoes.append((TAGS_CUDA[-1], False))

    try:
        with TemporaryDirectory(prefix="barenn-verificacao-") as temporaria:
            pasta = Path(temporaria)
            if args.clonar:
                repo = pasta / "clone"
                rodar(["git", "clone", "--", args.clonar, str(repo)])
                rodar(["git", "-C", str(repo), "rev-parse", "HEAD"])
                if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"]):
                    raise ValueError("o clone acabou de ser criado, mas tem alterações")
            print(platform.platform(), flush=True)
            if Path("/etc/os-release").is_file():
                distro = platform.freedesktop_os_release().get("PRETTY_NAME") if hasattr(
                    platform, "freedesktop_os_release",
                ) else None
                if distro:
                    print(distro, flush=True)
            print("OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE", flush=True)
            rodar(["git", "--version"])
            rodar([args.cxx, "--version"])
            if not args.somente_cpu:
                rodar([args.nvcc, "--version"])
            # Todas as versões recebem os mesmos arquivos da tag de referência.
            dados = tuple(pasta / nome for nome in ("optdigits.tra", "optdigits.tes"))
            for arquivo in dados:
                dados_originais = extrair(repo, TAGS_CPU[0], arquivo.name)
                arquivo.write_bytes(dados_originais)
                print(f"{arquivo.name}: SHA-256 {sha256(dados_originais).hexdigest()}")
            referencia = None
            todas_iguais = True
            for tag, cuda in versoes:
                try:
                    backend = "CUDA" if cuda else "CPU"
                    commit = subprocess.check_output(
                        ["git", "-C", str(repo), "rev-parse", f"refs/tags/{tag}^{{commit}}"],
                        text=True,
                    ).strip()
                    print(f"\n{tag} ({backend}) | {commit}", flush=True)
                    atual = list(ler_secoes(executar(tag, pasta / f"{tag}-{backend}",
                                                     repo, dados, args, cuda)))
                    for nome, bloco, _ in atual:
                        print(f"  SHA-256 de {nome} {sha256(bloco).hexdigest()}")
                except (OSError, ValueError, subprocess.SubprocessError) as erro:
                    raise ValueError(f"{tag}: {erro}") from erro
                if referencia is None:
                    referencia = atual
                    print("  Referência gravada.")
                elif comparar(referencia, atual):
                    print("  Idêntica à referência: 2.778 parâmetros, 51 perdas e 17.970 escores.")
                else:
                    todas_iguais = False
            if not todas_iguais:
                parser.exit(1, "Falha: há diferenças em relação à referência.\n")
    except (OSError, ValueError, subprocess.SubprocessError) as erro:
        parser.exit(1, f"Falha: {erro}\n")


if __name__ == "__main__":
    main()

# Barenn no Franky

O job **30367** rodou em 16 de novembro de 2026, no `compute20` da fila `sunny_gpu`. Usei um nó, quatro CPUs e uma GPU. O nó tem um Intel Core i9-14900 e uma NVIDIA RTX A1000, com driver 570.124.06. O programa de detecção encontrou compute capability 8.6, e os dois builds CUDA usaram `sm_86`. O ambiente foi Rocky Linux 9.5, com GCC 14.2.0 e o módulo CUDA 12.8.1, que fornece NVCC 12.8.93.

## Compilação e saída

Compilei as quatro tags em checkouts limpos, com o mesmo GCC no host. Usei C++17, `-O2 -g -Wall -Wextra -ffp-contract=off -fopenmp`, quatro threads disponíveis e `OMP_DYNAMIC=FALSE`. Nos builds CUDA, usei também `-lineinfo --fmad=false -DBARENN_CUDA`.

| Versão | Tag | Commit |
| --- | --- | --- |
| Legado | `v0.1-baseline` | `049aebcf40a2d340ba98bc891ea517750bacb571` |
| CPU otimizada | `v0.2-cpu` | `76f62efcfb0aa375d4f2ce4b0a564857085c2adf` |
| CUDA inicial | `v0.3-cuda` | `9c6cce268179fd3329e7900863a5a99b5c974f23` |
| CUDA final | `v0.5-cuda-padding` | `f7da50c91be18204a36b84926ce5387b1c7fb78e` |

As quatro execuções acertaram 1.717 de 1.797 imagens e tiveram perda de teste de 0,0432. Os binários usados no benchmark passaram na comparação com o legado e no caso de imagens zeradas, com tolerância zero. A checagem em builds instrumentados também confirmou igualdade bit a bit nos 2.778 parâmetros, nas 51 perdas e nos 17.970 escores. Ela incluiu a tag intermediária com shared e a `v1.0-aps2`. [Log do job](slurm-30367.txt) · [Estado no SLURM](slurm-estado.txt).

## Tempos

Usei os arquivos completos `optdigits.tra` e `optdigits.tes`, com 3.823 e 1.797 imagens. Fiz um aquecimento por binário e 12 rodadas. A primeira versão de cada rodada muda em rotação, então cada versão aparece três vezes em cada posição. A tabela mostra mediana ± desvio padrão amostral em segundos. Ponta a ponta inclui a abertura do processo, a leitura, o treino e a avaliação. [Tempos por rodada](benchmark.txt).

| Versão | Treino (s) | Ponta a ponta (s) | Fase de kernels (s) |
| --- | ---: | ---: | ---: |
| Legado | 0,5546 ± 0,0516 | 0,5832 ± 0,0540 | — |
| CPU otimizada | 0,3666 ± 0,0379 | 0,3899 ± 0,0403 | — |
| CUDA inicial | 2,8846 ± 0,0326 | 3,3958 ± 0,0334 | 1,2382 ± 0,0100 |
| CUDA final | 2,3123 ± 0,0343 | 2,8238 ± 0,0359 | 0,6699 ± 0,0091 |

A CPU ganhou **1,51× no treino** e **1,50× de ponta a ponta** sobre o legado no Franky. No notebook, com i9-12900H e MinGW/GCC 16.2.0, o treino ganhou **1,77×**. A comparação entre os ambientes reúne diferenças de processador, sistema e compilador. [Rodadas no notebook](../benchmark-windows.txt).

O CUDA final ganhou **1,85× na fase de kernels** frente ao inicial no Franky, perto dos **1,95×** observados na RTX 3070 Ti Laptop com NVCC 12.5.40 e MSVC no Windows. Essa fase é medida entre eventos CUDA e pode incluir intervalos entre lançamentos. No Franky, o ganho no tempo de treino medido pelo host foi de **1,25×**. Cerca de 1,64 s separam a mediana do treino e a fase de kernels nas duas versões CUDA. Essa diferença engloba a preparação do device, as cópias e o cálculo das médias no host. Para este conjunto, a CPU otimizada teve o menor tempo de treino e de ponta a ponta.

## Nsight

O Nsight Systems 2024.6.2 registrou 1.530 kernels em cada versão. A soma dos tempos caiu de **1,2283 s** para **0,6624 s**, um ganho de **1,85×** nessa coleta. Em cada versão houve duas cópias H2D de cerca de 1,005 MB no total e duas D2H de cerca de 0,791 MB. As quatro cópias somaram cerca de 0,144 ms. No notebook, o mesmo perfil registrou 1.530 kernels por versão e cerca de 0,143 ms em cópias no CUDA final. [Systems inicial](nsys-cuda-inicial-stats.txt) · [Systems final](nsys-cuda-otimizado-stats.txt) · [Systems no notebook](../nsight-systems.csv).

O Nsight Compute 2025.1.1.0 retornou `ERR_NVGPUCTRPERM` nas duas versões. Os arquivos abaixo trazem os comandos, a saída completa e o código de retorno. Os contadores de ocupação, tráfego e conflitos de banco usados na análise continuam sendo os do notebook. Nos logs públicos, substituí o caminho da pasta do experimento e o UUID da GPU. [Compute inicial](ncu-cuda-inicial.txt) · [Compute final](ncu-cuda-otimizado.txt) · [nvidia-smi](nvidia-smi.txt) · [Contadores do notebook](../nsight-compute.csv).

## Como repetir

No nó de login, carregue o módulo CUDA e submeta o script com o caminho do repositório e uma pasta de saída nova. A partição do job foi `sunny_gpu`. Confira as partições disponíveis com `sinfo`.

```bash
git clone https://github.com/EnzuBR/awful-barenn.git
cd awful-barenn
module load cuda/12.8.1
sbatch tools/franky.slurm "$PWD" "$PWD/execucao-franky"
```

No nó usado neste teste, faltavam cabeçalhos do sistema e o executável do Git. Usei uma cópia dos cabeçalhos e do Git do nó de login em uma pasta do experimento. O script passa os cabeçalhos ao GCC com `-idirafter`, depois dos cabeçalhos do compilador. Se encontrar a mesma configuração, rode no nó de login:

```bash
apoio=$(mktemp -d "${SCRATCH:-$HOME}/barenn-apoio.XXXXXX")
mkdir "$apoio/headers" "$apoio/bin"
cp -a /usr/include/. "$apoio/headers/"
cp "$(command -v git)" "$apoio/bin/git"
ln -s git "$apoio/bin/git-upload-pack"
PATH="$apoio/bin:$PATH" GIT_EXEC_PATH="$apoio/bin" FRANKY_HEADERS="$apoio/headers" \
  sbatch tools/franky.slurm "$PWD" "$apoio/execucao"
```

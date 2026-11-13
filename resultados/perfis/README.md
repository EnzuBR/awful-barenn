# Perfis da APS 2

Os perfis CUDA usam os conjuntos completos de treino e teste, `optdigits.tra` e `optdigits.tes`. O treino percorre 3.823 imagens por época, durante 51 épocas. As três execuções acertaram 1.717 das 1.797 imagens de teste, com perda de 0,0432.

| Executável | Tag | Issue |
| --- | --- | --- |
| `barenn-global.exe` | [v0.3-cuda](https://github.com/EnzuBR/awful-barenn/tree/v0.3-cuda) | [#5](https://github.com/EnzuBR/awful-barenn/issues/5) |
| `barenn-shared.exe` | [v0.4-cuda-shared](https://github.com/EnzuBR/awful-barenn/tree/v0.4-cuda-shared) | [#7](https://github.com/EnzuBR/awful-barenn/issues/7) |
| `barenn-padded.exe` | [v0.5-cuda-padding](https://github.com/EnzuBR/awful-barenn/tree/v0.5-cuda-padding) | [#8](https://github.com/EnzuBR/awful-barenn/issues/8) |

Os arquivos de dados são os mesmos da [tag de referência](https://github.com/EnzuBR/awful-barenn/tree/v0.1-baseline). Os IDs Git dos blobs de treino e teste são `188f5d214829554b61c9ac3ed529fca918f7c910` e `d70c84c9916ddf9b8b9f220a9177a6fc119fada7`.

## Compilação e coleta no Windows

Usei Windows 11, Intel Core i9-12900H, NVIDIA GeForce RTX 3070 Ti Laptop (driver 555.97), NVCC 12.5.40 e MSVC 19.29. Compile cada tag em sua pasta, em um prompt de desenvolvimento x64 do Visual Studio. Troque `global` por `shared` ou `padded` para as outras tags.

```powershell
nvcc -std=c++17 -arch=sm_86 -O2 -lineinfo --fmad=false -Xcompiler="/utf-8 /fp:precise /openmp" -DBARENN_CUDA -x cu barenn.cpp -o barenn-global.exe
```

Execute na raiz com os dois conjuntos de dados. Usei quatro threads na avaliação em CPU.

### Nsight Systems 2024.5.1.113

```powershell
$env:OMP_NUM_THREADS = '4'
$env:CUDA_INJECTION_SHM_ALLOWED = 'TRUE'
$versao = 'global'
nsys profile --sample=none --cpuctxsw=none --trace=cuda --force-overwrite=true --output=$versao ".\barenn-$versao.exe"
nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum "$versao.nsys-rep"
```

Cada relatório contém o treino completo, com 1.530 lançamentos e quatro cópias entre host e device. Os arquivos `nsys-*.txt` preservam as tabelas do `nsys stats`. Os CSVs por lançamento e por cópia foram extraídos das tabelas CUDA da exportação SQLite do Nsight Systems. O zero dos tempos corresponde ao início da primeira cópia H2D. Some `duracao_ns` para obter o tempo de cada tipo de atividade. O fim da última cópia D2H marca o intervalo completo no device. `gpu.csv` traz as propriedades técnicas da GPU selecionada, incluindo os 384,064 GB/s nominais que o Nsight registrou.

### Nsight Compute 2024.2.0, build 34181891

```powershell
$env:OMP_NUM_THREADS = '4'
$versao = 'global'
$metricas = 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,dram__bytes.sum,lts__t_bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warps_eligible.avg.per_cycle_active'
ncu --target-processes all --launch-skip 10 --launch-count 1 --section Occupancy --metrics $metricas --page details --csv --force-overwrite --export "ncu-$versao" ".\barenn-$versao.exe"
```

Use um terminal elevado se o driver restringir o acesso aos contadores. O profiler pula os dez primeiros lançamentos e mede o 11º, de índice 10. Ele processa as amostras de número 1.281 a 1.408 da primeira época, em uma grade de um bloco de 64 threads. O identificador 0 na tabela do Nsight Compute é a primeira coleta do relatório.

Os logs `ncu-*.txt` preservam a saída e a tabela CSV da coleta. A instrumentação do Nsight Compute altera a duração da execução. A comparação de desempenho usa as rodadas sem profiler e o Nsight Systems para separar kernels e cópias. Nos logs de GPU, `<FERRAMENTAS>` substitui o caminho local de instalação e trabalho.

Os arquivos `.nsys-rep` e `.ncu-rep` ficam locais porque guardam metadados da máquina e caminhos de arquivos. Os comandos acima geram esses relatórios para abrir no Nsight. As tabelas exportadas e os registros de todos os lançamentos e cópias estão nesta pasta.

### O que limita o kernel

Cada amostra faz cerca de 15.148 operações FP32 úteis. São 5.440 no forward, 1.344 nos gradientes de entrada usados, 8.160 nas atualizações de pesos, 116 nas atualizações de vieses, 48 na derivada da ReLU e 40 na perda. No lote de 128 amostras, isso dá 1.938.944 operações.

| Versão | Intensidade DRAM (FLOP/B) | Vazão no lançamento 11 (GFLOP/s) | Vazão no treino (GFLOP/s) |
| --- | ---: | ---: | ---: |
| Global | 20,0 | 2,12 | 2,12 |
| Shared | 32,8 | 1,94 | 2,90 |
| Shared com padding | 31,8 | 2,52 | 3,69 |

A intensidade usa os bytes DRAM do lançamento 11 no Nsight Compute. As duas vazões usam a duração do mesmo lançamento e a soma dos kernels do treino no Nsight Systems, respectivamente. São coletas separadas. O lançamento 11 da versão shared ficou mais lento na coleta do Systems. Para avaliar o ganho do treino, use a soma dos lançamentos e as rodadas sem profiler.

Com os 384,064 GB/s declarados pelo Nsight, o teto de DRAM do chip nessas intensidades vai de 7,70 a 12,59 TFLOP/s. Para FP32 sem FMA, 128 resultados por ciclo por SM e o clock nominal de 1,035 GHz dão cerca de 6,09 TFLOP/s para os 46 SMs e 132,5 GFLOP/s para um SM. O treino usa um bloco e dois warps por lançamento. Esse teto de um SM ainda supõe emissão contínua e fica bem acima do observado. Entre shared e padding, a ocupação atingida ficou em 4,17% e os bytes DRAM mudaram pouco. A queda nos conflitos de banco explica melhor o ganho nessa etapa. [Propriedades da GPU](gpu.csv) · [Vazão de instruções no guia CUDA](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#arithmetic-instructions).

## Callgrind 3.22.0

Usei Ubuntu 24.04 no WSL2 e GCC 13.3.0. O Callgrind perfila o processo todo e simula o comportamento do cache. As faltas de cache do relatório são simuladas. O tempo de relógio do processo sob o Callgrind inclui o custo da instrumentação.

```bash
# Na raiz de cada versão. Use flags OpenMP apenas na tag v0.2-cpu.
g++ -std=c++17 -O2 -g -Wall -Wextra -ffp-contract=off barenn.cpp -o barenn
# Para v0.2-cpu:
# g++ -std=c++17 -O2 -g -Wall -Wextra -ffp-contract=off -fopenmp barenn.cpp -o barenn

OMP_NUM_THREADS=4 valgrind --tool=callgrind --cache-sim=yes \
  --callgrind-out-file=callgrind.out ./barenn
callgrind_annotate --inclusive=no --auto=yes --threshold=92 callgrind.out
```

A simulação usou linhas de 64 B, L1 de dados de 48 KiB e último nível de 24 MiB. O [CSV da primeira coleta](../callgrind.csv) corresponde às Issues [#1](https://github.com/EnzuBR/awful-barenn/issues/1) e [#2](https://github.com/EnzuBR/awful-barenn/issues/2). Repeti a coleta a partir das mesmas tags para guardar os arquivos `.out`, a saída do Valgrind e a anotação de fonte. As duas coletas tiveram a mesma saída numérica.

| Versão | Instruções na nova coleta | Faltas L1 de dados | Faltas no último nível de dados |
| --- | ---: | ---: | ---: |
| `v0.1-baseline` | 14.004.012.760 | 944.772 | 36.156 |
| `v0.2-cpu` | 10.868.372.995 | 997.312 | 36.878 |

Na repetição, o backward do legado continuou com 67,36% das instruções. Os perfis da pasta abaixo estão comprimidos como `.out.gz`. Para abrir um deles no Callgrind, descompacte com `gzip -dk arquivo.out.gz`. O arquivo `.out` na raiz do projeto pertence a uma versão anterior ao baseline. [Arquivos da nova coleta](callgrind/).

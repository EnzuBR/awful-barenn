# APS 2 · Barenn

O Barenn treina uma rede densa para classificar dígitos. Mantive a rede, as 51 épocas e o SGD na ordem do arquivo. Otimizei o treino em CPU e a avaliação com OpenMP. Em CUDA, passei os pesos para memória compartilhada e ajustei o layout para reduzir conflitos de banco.

| Versão | Treino (s) | Ponta a ponta (s) | Fase de kernels (s) | Issue |
| --- | ---: | ---: | ---: | --- |
| [Baseline legado](https://github.com/EnzuBR/awful-barenn/tree/v0.1-baseline) | 0,7247 ± 0,0129 | 1,5527 ± 0,0279 | — | [#1](https://github.com/EnzuBR/awful-barenn/issues/1) |
| [CPU otimizada](https://github.com/EnzuBR/awful-barenn/tree/v0.2-cpu) | 0,4100 ± 0,0167 | 1,2408 ± 0,0413 | — | [#2](https://github.com/EnzuBR/awful-barenn/issues/2) |
| [CUDA inicial](https://github.com/EnzuBR/awful-barenn/tree/v0.3-cuda) | 1,5481 ± 0,0525 | 1,7399 ± 0,0593 | 1,4576 ± 0,0437 | [#5](https://github.com/EnzuBR/awful-barenn/issues/5) |
| [CUDA com memória compartilhada](https://github.com/EnzuBR/awful-barenn/tree/v0.4-cuda-shared) | 1,0397 ± 0,0393 | 1,2476 ± 0,0360 | 0,9671 ± 0,0330 | [#7](https://github.com/EnzuBR/awful-barenn/issues/7) |
| [CUDA otimizado com padding](https://github.com/EnzuBR/awful-barenn/tree/v0.5-cuda-padding) | 0,8196 ± 0,0038 | 1,0113 ± 0,0097 | 0,7463 ± 0,0001 | [#8](https://github.com/EnzuBR/awful-barenn/issues/8) |

Mediana ± desvio padrão amostral de nove rodadas alternadas, após um aquecimento por executável. Usei o mesmo i9-12900H, RTX 3070 Ti Laptop, Windows 11 e os arquivos completos `optdigits.tra` e `optdigits.tes`. O tempo de treino em CUDA inclui a preparação do device e as transferências. A fase de kernels é o intervalo medido por eventos CUDA. O tempo de ponta a ponta inclui leitura, treino e avaliação. [Medições por rodada](resultados/benchmark-windows.txt).

A CPU otimizada treinou **1,77× mais rápido** que o legado. O CUDA final reduziu a fase de kernels em **1,95×** frente ao CUDA inicial. Na tabela acima, o menor tempo total foi do executável CUDA, que usou MSVC no host. A CPU usou MinGW. A comparação com o mesmo compilador está abaixo.

## CPU e CUDA com o mesmo compilador

Compilei os dois modos com NVCC 12.5.40 e MSVC 19.29 no host. Alterei apenas a macro `BARENN_CUDA`. Usei os mesmos dados e quatro threads na avaliação, com 15 rodadas alternadas e um aquecimento por modo.

| Modo | Leitura (ms) | Treino (ms) | Avaliação (ms) | Ponta a ponta (ms) |
| --- | ---: | ---: | ---: | ---: |
| CPU | 167,95 ± 9,61 | 798,30 ± 49,08 | 1,14 ± 0,56 | 977,44 ± 57,08 |
| CUDA | 176,41 ± 24,92 | 909,50 ± 53,93 | 1,30 ± 0,34 | 1.120,60 ± 62,58 |

A diferença de leitura ficou dentro da dispersão. Com o host igual, a CPU teve menor mediana de treino e de ponta a ponta. O SGD usa os pesos atualizados na amostra anterior. Um único bloco fica ativo por lançamento, e as barreiras entre operações das camadas limitam o trabalho que a GPU consegue sobrepor. [Issue #13](https://github.com/EnzuBR/awful-barenn/issues/13) · [Tempos por rodada](resultados/mesmo-compilador-windows.txt).

## Uma passagem com OpenMP

Com o mesmo executável CPU, quatro threads fizeram a inferência em **1,450 ms** na mediana, ante **2,703 ms** com uma thread. Foram 15 rodadas alternadas, com ganho em 13 delas e a mesma saída numérica em todas. O tempo de ponta a ponta ficou perto de 0,96 s nas duas configurações. A inferência responde por uma fração pequena do total.

Também testei ler treino e teste em paralelo. No MSVC, a mediana da leitura subiu de 159 para 205 ms. Mantive a leitura serial e o OpenMP na avaliação. [Issue #14](https://github.com/EnzuBR/awful-barenn/issues/14) · [Rodadas da inferência](resultados/threads-msvc.txt) · [Ensaio de leitura](resultados/leitura-paralela-msvc.txt).

As cinco versões obtiveram **1.717/1.797 acertos (95,5%)** e perda de teste **0,0432**. A verificação binária em Ubuntu/WSL2, com GCC 13.3 e NVCC 12.5, comparou parâmetros finais, perdas por época e escores de teste. Os resultados coincidiram bit a bit.

## Como executar

Requer C++17. Para CUDA, use também o Toolkit e uma GPU NVIDIA. Os comandos usam os dados da raiz do repositório. O executável aceita dois caminhos opcionais, de treino e de teste, nessa ordem.

No Linux ou WSL2, com GCC:

```bash
g++ -std=c++17 -O2 -g -Wall -Wextra -ffp-contract=off -fopenmp barenn.cpp -o barenn-cpu
OMP_NUM_THREADS=4 ./barenn-cpu

nvcc -std=c++17 -arch=sm_86 -O2 -lineinfo --fmad=false \
  -Xcompiler=-ffp-contract=off,-fopenmp -DBARENN_CUDA -x cu barenn.cpp -o barenn-cuda
OMP_NUM_THREADS=4 ./barenn-cuda
```

Ajuste `sm_86` à sua GPU. Para CPU sem OpenMP, omita `-fopenmp`. No Windows, os builds da tabela usaram MinGW GCC 16.2 para CPU e NVCC 12.5.40 com MSVC 19.29 para CUDA. Acrescente `-static` ao comando MinGW e use `-Xcompiler="/utf-8 /fp:precise /openmp"` no NVCC em um prompt de desenvolvimento do Visual Studio. O legado usa o mesmo comando GCC sem `-fopenmp`.

## Como conferir os resultados

Em Linux/WSL2, entre na raiz de cada tag e use a linha correspondente. O comando `git switch --detach <tag>` seleciona a versão em um clone de trabalho.

| Tag | Como compilar |
| --- | --- |
| `v0.1-baseline` | Comando GCC acima sem `-fopenmp`. |
| `v0.2-cpu` | Comando GCC acima com `-fopenmp`. |
| `v0.3-cuda`, `v0.4-cuda-shared`, `v0.5-cuda-padding` | Comando NVCC acima com `-DBARENN_CUDA`. |
| `v1.0-aps2` e marcos seguintes | GCC para CPU ou NVCC com `-DBARENN_CUDA` para CUDA. |

Use os scripts da `main` com os caminhos dos executáveis.

```bash
OMP_NUM_THREADS=4 python3 tools/benchmark.py \
  ./barenn-legado ./barenn-cpu ./barenn-global ./barenn-shared ./barenn-cuda --repeticoes 9
python3 tools/verificar.py ./barenn-legado ./barenn-cuda --atol 0
python3 tools/verificar_binario.py --clonar https://github.com/EnzuBR/awful-barenn.git --arch sm_86
git show v0.2-cpu:barenn.cpp > /tmp/barenn-v0.2.cpp
python3 tools/bench_openmp.py /tmp/barenn-v0.2.cpp
```

O primeiro verificador usa o conjunto completo e um caso analítico com imagens zeradas. O verificador binário baixa um clone novo e compila as seis tags de `v0.1-baseline` a `v1.0-aps2` em uma pasta temporária. Ele roda a `v1.0-aps2` em CPU e CUDA e compara pesos, vieses, perdas e escores. Use `--nvcc` se o compilador CUDA estiver fora do PATH. [Log da comparação binária](resultados/clone-limpo-wsl.txt) · [Builds e execuções dos programas das tags](resultados/tags-sem-instrumentacao.txt).

O microbenchmark OpenMP mede 100 passagens pelo conjunto de teste com uma equipe por processo. [Rodadas do OpenMP](resultados/openmp-windows.txt).

Os perfis trazem os comandos, as versões das ferramentas e os arquivos de saída. [Como repetir as coletas](resultados/perfis/README.md) · [Issue #15](https://github.com/EnzuBR/awful-barenn/issues/15).

## Enunciado original

# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.

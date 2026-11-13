# Ensaio de leitura paralela

O patch aplica o protótipo da Issue #14 à tag `v1.1-mesmo-compilador`. Com uma thread, o programa lê treino e teste em sequência. Com quatro threads disponíveis, usa duas na leitura, uma por arquivo, e quatro na avaliação. O benchmark compara a saída numérica em todas as rodadas.

No PowerShell com o ambiente x64 do Visual Studio, rode a partir da raiz do repositório:

```powershell
$patch = (Resolve-Path resultados/leitura-paralela.patch).Path
$benchmark = (Resolve-Path tools/benchmark_threads.py).Path
git worktree add --detach ../barenn-leitura v1.1-mesmo-compilador
Push-Location ../barenn-leitura
git apply --check $patch
git apply $patch
nvcc -std=c++17 -arch=sm_86 -O2 -lineinfo --fmad=false -Xcompiler="/utf-8 /fp:precise /openmp" -x cu barenn.cpp -o cpu-leitura.exe
python $benchmark ./cpu-leitura.exe --rodadas 15
Pop-Location
```

Usei NVCC 12.5.40, MSVC 19.29 e os arquivos completos. A mediana da leitura foi de 159,10 ms em sequência e 205,07 ms em paralelo. O tempo total passou de 0,941 s para 0,983 s. A versão de entrega mantém a leitura serial. [Patch](leitura-paralela.patch) · [Tempos por rodada](leitura-paralela-msvc.txt) · [Issue #14](https://github.com/EnzuBR/awful-barenn/issues/14).

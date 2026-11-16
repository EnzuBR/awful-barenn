#!/bin/bash
#SBATCH --job-name=barenn-franky
#SBATCH --partition=sunny_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=barenn-franky-%j.out

set -euo pipefail
export LC_ALL=C PYTHONIOENCODING=UTF-8
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE

if [[ $# != 2 ]]; then
    echo "Uso: sbatch [--partition=fila] tools/franky.slurm REPO NOVA_PASTA" >&2
    exit 2
fi
repo=$(realpath -- "$1")
saida=$(realpath -m -- "$2")
mkdir -- "$saida"

registrar_comando() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
}

registrar() {
    local log=$1 status=0
    shift
    registrar_comando "$@" > "$saida/$log"
    "$@" >> "$saida/$log" 2>&1 || status=$?
    printf '\nCódigo de saída: %s\n' "$status" >> "$saida/$log"
    cat "$saida/$log"
    return "$status"
}

{
    date -Is
    hostname
    uname -a
    cat /etc/os-release
    lscpu
    grep '^Cpus_allowed_list:' /proc/self/status
    printf 'SLURM_JOB_ID=%s\nSLURM_JOB_PARTITION=%s\nSLURM_JOB_NODELIST=%s\n' \
        "${SLURM_JOB_ID:-}" "${SLURM_JOB_PARTITION:-}" "${SLURM_JOB_NODELIST:-}"
    printf 'SLURM_CPUS_PER_TASK=%s\nSLURM_JOB_GPUS=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
        "${SLURM_CPUS_PER_TASK:-}" "${SLURM_JOB_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"
    printf 'OMP_NUM_THREADS=%s\nOMP_DYNAMIC=%s\n' "$OMP_NUM_THREADS" "$OMP_DYNAMIC"
} | tee "$saida/ambiente.txt"

if ! command -v nvcc >/dev/null && ! type module >/dev/null 2>&1; then
    if [[ -r /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -r /etc/profile.d/lmod.sh ]]; then
        source /etc/profile.d/lmod.sh
    fi
fi
if [[ -n ${CUDA_MODULE:-} ]] || ! command -v nvcc >/dev/null; then
    registrar modulo.txt module load "${CUDA_MODULE:-cuda/12.8.1}"
fi
if type module >/dev/null 2>&1; then
    registrar modulos.txt module list
fi
gcc=$(command -v "${CXX:-g++}")
# Alguns nós de GPU não têm os cabeçalhos de desenvolvimento do Rocky Linux.
if [[ -n ${FRANKY_HEADERS:-} ]]; then
    [[ -r "$FRANKY_HEADERS/features.h" ]]
    # O fallback vem depois dos includes do GCC, inclusive para include_next.
    {
        printf '#!/bin/bash\nexec %q -idirafter %q "$@"\n' "$gcc" "$FRANKY_HEADERS"
    } > "$saida/g++"
    chmod +x "$saida/g++"
    cat "$saida/g++" | tee -a "$saida/ambiente.txt"
    gcc="$saida/g++"
fi
nvcc=$(command -v nvcc)
registrar gcc.txt "$gcc" --version
registrar nvcc.txt "$nvcc" --version
registrar python.txt python3 --version
registrar git.txt git --version
registrar nvidia-smi.txt nvidia-smi
registrar gpu.txt nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version --format=csv
registrar nsys-versao.txt nsys --version || true
registrar ncu-versao.txt ncu --version || true

# Consulta o dispositivo CUDA entregue pelo Slurm na ordem local do processo.
cat > "$saida/dispositivo.cu" <<'EOF'
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int total = 0;
    cudaError_t resultado = cudaGetDeviceCount(&total);
    if (resultado != cudaSuccess || total != 1) {
        std::fprintf(stderr, "Dispositivos CUDA visíveis: %d (%s)\n",
                     total, cudaGetErrorString(resultado));
        return 1;
    }
    cudaDeviceProp gpu{};
    resultado = cudaGetDeviceProperties(&gpu, 0);
    if (resultado != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(resultado));
        return 1;
    }
    char pci[32]{};
    resultado = cudaDeviceGetPCIBusId(pci, sizeof(pci), 0);
    if (resultado != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(resultado));
        return 1;
    }
    std::printf("GPU CUDA 0: %s, PCI %s\nsm_%d%d\n", gpu.name, pci, gpu.major, gpu.minor);
}
EOF
registrar compilar-dispositivo.txt "$nvcc" -ccbin "$gcc" "$saida/dispositivo.cu" -o "$saida/dispositivo"
registrar dispositivo.txt "$saida/dispositivo"
arch=$(grep -E '^sm_[0-9]+$' "$saida/dispositivo.txt")

tags=(v0.1-baseline v0.2-cpu v0.3-cuda v0.5-cuda-padding)
nomes=(legado cpu cuda-inicial cuda-otimizado)
executaveis=()
flags_gcc=(-std=c++17 -O2 -g -Wall -Wextra -ffp-contract=off -fopenmp)
flags_cuda=(-std=c++17 -O2 -lineinfo "--gpu-architecture=$arch" --fmad=false
            -Xcompiler=-g,-Wall,-Wextra,-ffp-contract=off,-fopenmp -DBARENN_CUDA -x cu)

clone="$saida/clone"
registrar clonar.txt git clone --no-hardlinks --no-checkout -- "$repo" "$clone"
for i in "${!tags[@]}"; do
    tag=${tags[i]}
    nome=${nomes[i]}
    pasta="$saida/$tag"
    commit=$(git -C "$clone" rev-parse "refs/tags/$tag^{commit}")
    printf '%s %s\n' "$tag" "$commit" | tee -a "$saida/tags.txt"
    registrar "checkout-$nome.txt" git -C "$clone" worktree add --detach "$pasta" "$commit"
    registrar "estado-$nome.txt" git -C "$pasta" status --porcelain
    if [[ -n $(git -C "$pasta" status --porcelain) ]]; then
        echo "Checkout de $tag tem alterações." >&2
        exit 1
    fi
    registrar "$nome-sha256.txt" sha256sum "$pasta/optdigits.tra" "$pasta/optdigits.tes" "$pasta/barenn.cpp"
    exe="$pasta/barenn-$nome"
    executaveis+=("$exe")
    if (( i < 2 )); then
        registrar "compilar-$nome.txt" "$gcc" "${flags_gcc[@]}" "$pasta/barenn.cpp" -o "$exe"
    else
        registrar "$nome-cuda-sha256.txt" sha256sum "$pasta/barenn_cuda.cuh"
        registrar "compilar-$nome.txt" "$nvcc" -ccbin "$gcc" "${flags_cuda[@]}" "$pasta/barenn.cpp" -o "$exe"
    fi
done

base="$saida/${tags[0]}"
treino="$base/optdigits.tra"
teste="$base/optdigits.tes"
for i in "${!tags[@]}"; do
    pasta="$saida/${tags[i]}"
    cmp -- "$treino" "$pasta/optdigits.tra"
    cmp -- "$teste" "$pasta/optdigits.tes"
    registrar "saida-${nomes[i]}.txt" "${executaveis[i]}" "$treino" "$teste"
done
for i in 1 2 3; do
    registrar "validar-${nomes[i]}.txt" python3 "$repo/tools/verificar.py" \
        "${executaveis[0]}" "${executaveis[i]}" --repo "$base" --atol 0
done
# A checagem binária também inclui shared e v1.0, com cópias instrumentadas.
registrar validar-binario.txt python3 "$repo/tools/verificar_binario.py" \
    --repo "$clone" --cxx "$gcc" --nvcc "$nvcc" --arch "$arch"
registrar benchmark.txt python3 "$repo/tools/benchmark.py" "${executaveis[@]}" \
    --treino "$treino" --teste "$teste" --repeticoes 12

python3 - "$saida/benchmark.txt" <<'PY' | tee "$saida/speedups.txt"
from pathlib import Path
import re
import sys

dados = {}
for linha in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if linha.endswith(" (segundos)"):
        nome = linha.split()[0]
        dados[nome] = {}
    medida = re.match(r"\s*(\w+) \| mediana ([0-9.]+)", linha)
    if medida:
        dados[nome][medida[1]] = float(medida[2])
for nome, tempos in dados.items():
    print(
        f"{nome} vs legado: treino {dados['barenn-legado']['treino'] / tempos['treino']:.3f}x, "
        f"ponta a ponta {dados['barenn-legado']['total'] / tempos['total']:.3f}x"
    )
inicial, otimizado = (dados["barenn-" + nome] for nome in ("cuda-inicial", "cuda-otimizado"))
print(f"CUDA otimizado vs inicial: treino {inicial['treino'] / otimizado['treino']:.3f}x, "
      f"ponta a ponta {inicial['total'] / otimizado['total']:.3f}x, "
      f"fase de kernels {inicial['gpu_kernel'] / otimizado['gpu_kernel']:.3f}x")
PY

metricas=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,dram__bytes.sum,lts__t_bytes.sum
metricas+=,sm__warps_active.avg.pct_of_peak_sustained_active
metricas+=,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warps_eligible.avg.per_cycle_active
for i in 2 3; do
    nome=${nomes[i]}
    exe=${executaveis[i]}
    sistemas="$saida/nsys-$nome"
    if registrar "nsys-$nome.txt" nsys profile --sample=none --cpuctxsw=none \
        --trace=cuda --output="$sistemas" "$exe" "$treino" "$teste"; then
        registrar "nsys-$nome-stats.txt" nsys stats \
            --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
            "$sistemas.nsys-rep" || true
    fi
    registrar "ncu-$nome.txt" ncu --target-processes all \
        --launch-skip 10 --launch-count 1 --section Occupancy --metrics "$metricas" \
        --page details --csv --export "$saida/ncu-$nome" "$exe" "$treino" "$teste" || true
done
registrar nvidia-smi-final.txt nvidia-smi

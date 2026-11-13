#pragma once

#include <cuda_runtime.h>

namespace Barenn {
namespace gpu {

inline void verificar(cudaError_t resultado, const char* operacao) {
    if (resultado != cudaSuccess) {
        throw runtime_error(string(operacao) + ": " + cudaGetErrorString(resultado));
    }
}

template <typename T>
class Buffer {
public:
    explicit Buffer(size_t quantidade) : bytes(quantidade * sizeof(T)) {
        verificar(cudaMalloc(&ponteiro, bytes), "cudaMalloc");
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { cudaFree(ponteiro); }

    T* dados() const { return ponteiro; }

    void enviar(const T* host) const {
        verificar(cudaMemcpy(ponteiro, host, bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy host para device");
    }

    void receber(T* host) const {
        verificar(cudaMemcpy(host, ponteiro, bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy device para host");
    }

private:
    T* ponteiro = nullptr;
    size_t bytes;
};

class Evento {
public:
    Evento() { verificar(cudaEventCreate(&evento), "cudaEventCreate"); }
    Evento(const Evento&) = delete;
    Evento& operator=(const Evento&) = delete;
    ~Evento() { cudaEventDestroy(evento); }

    void marcar() const { verificar(cudaEventRecord(evento), "cudaEventRecord"); }

    float desde(const Evento& inicio) const {
        verificar(cudaEventSynchronize(evento), "cudaEventSynchronize");
        float milissegundos = 0;
        verificar(cudaEventElapsedTime(&milissegundos, inicio.evento, evento),
                  "cudaEventElapsedTime");
        return milissegundos;
    }

private:
    cudaEvent_t evento = nullptr;
};

template <int Entradas, int Saidas, int Passo = Entradas>
struct Camada {
    float pesos[Passo * Saidas];
    float vieses[Saidas];
};

template <int Padding = 0>
struct Modelo {
    Camada<64, 32, 64 + Padding> l1;
    Camada<32, 16, 32 + Padding> l2;
    Camada<16, 10, 16 + Padding> l3;
};

struct Exemplo {
    float imagem[64];
    int rotulo;
};

struct Intermediarios {
    float saida1[32];
    float saida2[16];
    float escores[10];
    float gradiente1[32];
    float gradiente2[16];
    float gradiente3[10];
    float delta[32];
};

template <bool Final, int Entradas, int Saidas, int Passo>
__device__ void forward(const Camada<Entradas, Saidas, Passo>& camada,
                        const float* entrada, float* saida) {
    int i = threadIdx.x;
    if (i < Saidas) {
        float soma = camada.vieses[i];
        for (int j = 0; j < Entradas; ++j) {
            soma += entrada[j] * camada.pesos[i * Passo + j];
        }
        saida[i] = Final || soma > 0.0f ? soma : 0.0f;
    }
    __syncthreads();
}

template <bool Final, bool CalcularEntrada, int Entradas, int Saidas, int Passo>
__device__ void backward(Camada<Entradas, Saidas, Passo>& camada, const float* entrada,
                         const float* saida, const float* grad_saida,
                         float* grad_entrada, float* delta, float taxa) {
    int tid = threadIdx.x;
    if (tid < Saidas) {
        delta[tid] = Final ? grad_saida[tid]
                           : grad_saida[tid] * (saida[tid] > 0.0f ? 1.0f : 0.0f);
    }
    __syncthreads();

    if constexpr (CalcularEntrada) {
        if (tid < Entradas) {
            float soma = 0.0f;
            for (int j = 0; j < Saidas; ++j) {
                soma += delta[j] * camada.pesos[j * Passo + tid];
            }
            grad_entrada[tid] = soma;
        }
        // Nenhuma thread pode atualizar um peso que outra ainda está lendo.
        __syncthreads();
    }

    for (int i = tid; i < Entradas * Saidas; i += blockDim.x) {
        const int linha = i / Entradas;
        const int coluna = i % Entradas;
        camada.pesos[linha * Passo + coluna] -= taxa * (delta[linha] * entrada[coluna]);
    }
    if (tid < Saidas) {
        camada.vieses[tid] -= taxa * delta[tid];
    }
    __syncthreads();
}

template <int Entradas, int Saidas, int PassoOrigem, int PassoDestino>
__device__ void copiar_camada(const Camada<Entradas, Saidas, PassoOrigem>& origem,
                              Camada<Entradas, Saidas, PassoDestino>& destino) {
    int tid = threadIdx.x;
    for (int i = tid; i < Entradas * Saidas; i += blockDim.x) {
        const int linha = i / Entradas;
        const int coluna = i % Entradas;
        destino.pesos[linha * PassoDestino + coluna] = origem.pesos[linha * PassoOrigem + coluna];
    }
    if (tid < Saidas) {
        destino.vieses[tid] = origem.vieses[tid];
    }
}

static __global__ void treinar_lote(Modelo<>* modelo,
                                    const Exemplo* exemplos, float* perdas,
                                    int quantidade, float taxa) {
    // Cada linha ocupa um banco diferente quando o forward lê uma coluna.
    __shared__ Modelo<1> local;
    __shared__ Intermediarios estado;
    copiar_camada<64, 32>(modelo->l1, local.l1);
    copiar_camada<32, 16>(modelo->l2, local.l2);
    copiar_camada<16, 10>(modelo->l3, local.l3);
    __syncthreads();
    // As amostras seguem em série. As operações de cada camada usam o bloco.
    for (int n = 0; n < quantidade; ++n) {
        const Exemplo& exemplo = exemplos[n];
        forward<false>(local.l1, exemplo.imagem, estado.saida1);
        forward<false>(local.l2, estado.saida1, estado.saida2);
        forward<true>(local.l3, estado.saida2, estado.escores);

        if (threadIdx.x == 0) {
            float perda = 0.0f;
            for (int i = 0; i < 10; ++i) {
                float erro = estado.escores[i] - (i == exemplo.rotulo ? 1.0f : 0.0f);
                perda += 0.5f * erro * erro;
                estado.gradiente3[i] = erro;
            }
            perdas[n] = perda;
        }
        __syncthreads();

        backward<true, true>(local.l3, estado.saida2, estado.escores,
                             estado.gradiente3, estado.gradiente2, estado.delta, taxa);
        backward<false, true>(local.l2, estado.saida1, estado.saida2,
                              estado.gradiente2, estado.gradiente1, estado.delta, taxa);
        backward<false, false>(local.l1, exemplo.imagem, estado.saida1,
                                estado.gradiente1, nullptr, estado.delta, taxa);
    }
    copiar_camada<64, 32>(local.l1, modelo->l1);
    copiar_camada<32, 16>(local.l2, modelo->l2);
    copiar_camada<16, 10>(local.l3, modelo->l3);
}

template <int Entradas, int Saidas>
void copiar_para_modelo(const DenseLayer& host, Camada<Entradas, Saidas>& camada) {
    copy_n(host.pesos().data(), Entradas * Saidas, camada.pesos);
    copy_n(host.vieses().data(), Saidas, camada.vieses);
}

template <int Entradas, int Saidas>
void copiar_para_host(const Camada<Entradas, Saidas>& camada, DenseLayer& host) {
    copy_n(camada.pesos, Entradas * Saidas, host.pesos().data());
    copy_n(camada.vieses, Saidas, host.vieses().data());
}

} // namespace gpu

template <size_t Epocas>
void treinar_cuda(const vector<Amostra>& treino, DenseLayer& l1, DenseLayer& l2,
                  DenseLayer& l3, array<double, Epocas>& perdas, float taxa) {
    gpu::Modelo<> modelo;
    gpu::copiar_para_modelo(l1, modelo.l1);
    gpu::copiar_para_modelo(l2, modelo.l2);
    gpu::copiar_para_modelo(l3, modelo.l3);

    vector<gpu::Exemplo> exemplos(treino.size());
    for (size_t i = 0; i < treino.size(); ++i) {
        copy_n(treino[i].imagem.data(), 64, exemplos[i].imagem);
        exemplos[i].rotulo = treino[i].rotulo;
    }
    vector<float> perdas_amostras(Epocas * treino.size());

    gpu::Buffer<gpu::Modelo<>> d_modelo(1);
    gpu::Buffer<gpu::Exemplo> d_exemplos(exemplos.size());
    gpu::Buffer<float> d_perdas(perdas_amostras.size());
    gpu::Evento inicio, depois_envio, depois_kernels, depois_retorno;

    inicio.marcar();
    d_modelo.enviar(&modelo);
    d_exemplos.enviar(exemplos.data());
    depois_envio.marcar();

    constexpr int threads = 64;
    constexpr int amostras_por_lote = 128;
    for (size_t epoca = 0; epoca < Epocas; ++epoca) {
        for (size_t i = 0; i < exemplos.size(); i += amostras_por_lote) {
            int quantidade = static_cast<int>(min<size_t>(amostras_por_lote, exemplos.size() - i));
            gpu::treinar_lote<<<1, threads>>>(
                d_modelo.dados(), d_exemplos.dados() + i,
                d_perdas.dados() + epoca * exemplos.size() + i, quantidade, taxa);
            gpu::verificar(cudaGetLastError(), "treinar_lote");
        }
    }
    depois_kernels.marcar();
    d_modelo.receber(&modelo);
    d_perdas.receber(perdas_amostras.data());
    depois_retorno.marcar();

    copiar_para_host(modelo.l1, l1);
    copiar_para_host(modelo.l2, l2);
    copiar_para_host(modelo.l3, l3);

    for (size_t epoca = 0; epoca < Epocas; ++epoca) {
        double total = 0.0;
        for (size_t i = 0; i < exemplos.size(); ++i) {
            total += perdas_amostras[epoca * exemplos.size() + i];
        }
        perdas[epoca] = total / exemplos.size();
    }

    cout << fixed << setprecision(6)
         << "CUDA H2D " << depois_envio.desde(inicio) / 1000.0 << " s\n"
         << "CUDA kernels " << depois_kernels.desde(depois_envio) / 1000.0 << " s\n"
         << "CUDA D2H " << depois_retorno.desde(depois_kernels) / 1000.0 << " s\n";
}

} // namespace Barenn

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;
constexpr int oculta1 = 32;
constexpr int oculta2 = 16;

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // A linha i contém os pesos de todas as entradas do neurônio i.
    vector<float> weights;
    vector<float> bias;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(output_size * input_size), bias(output_size, 0) {
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (float& weight : weights) {
            weight = distribution(gen);
        }
    }

    const vector<float>& pesos() const { return weights; }
    vector<float>& pesos() { return weights; }
    const vector<float>& vieses() const { return bias; }
    vector<float>& vieses() { return bias; }

    void Forward(const float* input, float* out, float* z = nullptr) const {
        for (int i = 0; i < out_features; ++i) {
            const float* row = weights.data() + i * in_features;
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * row[j];
            }
            if (z) {
                z[i] = sum;
            }
            out[i] = final ? sum : max(0.0f, sum);
        }
    }

    void Backward(const float* input, const float* z, const float* grad_out,
                  float learning_rate, float* grad_in) {
        if (grad_in) {
            fill_n(grad_in, in_features, 0.0f);
        }
        for (int i = 0; i < out_features; ++i) {
            const float delta = final ? grad_out[i]
                                      : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
            float* row = weights.data() + i * in_features;
            // Percorre uma linha contígua e usa os pesos antes de atualizá-los.
            if (grad_in) {
                for (int j = 0; j < in_features; ++j) {
                    grad_in[j] += delta * row[j];
                }
            }
            for (int j = 0; j < in_features; ++j) {
                row[j] -= learning_rate * (delta * input[j]);
            }
            bias[i] -= learning_rate * delta;
        }
    }
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

void avaliar(const DenseLayer& l1, const DenseLayer& l2, const DenseLayer& l3,
             const vector<Amostra>& teste) {
    struct Resultado {
        int previsto;
        float erro;
    };
    vector<Resultado> resultados(teste.size());

#ifdef _OPENMP
    const int threads = getenv("OMP_NUM_THREADS") ? omp_get_max_threads()
                                                 : min(4, omp_get_max_threads());
#endif
    const auto inicio = chrono::steady_clock::now();
#ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(teste.size()); ++i) {
        const auto& amostra = teste[i];
        array<float, oculta1> a1;
        array<float, oculta2> a2;
        array<float, classes> escores;
        l1.Forward(amostra.imagem.data(), a1.data());
        l2.Forward(a1.data(), a2.data());
        l3.Forward(a2.data(), escores.data());
        resultados[i] = {
            static_cast<int>(max_element(escores.begin(), escores.end()) - escores.begin()),
            perda(escores.data(), amostra.rotulo)
        };
    }
    const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

    // Soma na ordem das amostras para manter as métricas e os exemplos.
    array<array<size_t, classes>, classes> confusoes{};
    vector<pair<int, int>> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;
    for (size_t i = 0; i < teste.size(); ++i) {
        const int real = teste[i].rotulo;
        const auto [previsto, erro] = resultados[i];
        const bool acertou = previsto == real;
        acertos += acertou;
        ++confusoes[real][previsto];
        perda_total += erro;

        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            exemplos.emplace_back(real, previsto);
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n"
         << "Tempo de inferencia " << setprecision(6) << tempo.count() << " s\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& [real, previsto] : exemplos) {
        cout << right << setw(8) << real
             << "  " << setw(8) << previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

#ifdef BARENN_CUDA
#include "barenn_cuda.cuh"
#endif

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, Barenn::oculta1);
        Barenn::DenseLayer l2(Barenn::oculta1, Barenn::oculta2);
        Barenn::DenseLayer l3(Barenn::oculta2, Barenn::classes, true);
        array<double, epocas> perdas{};

#ifndef BARENN_CUDA
        array<float, Barenn::oculta1> a1, z1, grad2;
        array<float, Barenn::oculta2> a2, z2, grad3;
        array<float, Barenn::classes> escores, gradiente;
#endif
        const auto inicio = chrono::steady_clock::now();
#ifdef BARENN_CUDA
        Barenn::treinar_cuda(treino, l1, l2, l3, perdas, taxa);
#else
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (const auto& amostra : treino) {
                const float* imagem = amostra.imagem.data();
                l1.Forward(imagem, a1.data(), z1.data());
                l2.Forward(a1.data(), a2.data(), z2.data());
                l3.Forward(a2.data(), escores.data());
                perda_total += Barenn::perda(escores.data(), amostra.rotulo, gradiente.data());

                l3.Backward(a2.data(), nullptr, gradiente.data(), taxa, grad3.data());
                l2.Backward(a1.data(), z2.data(), grad3.data(), taxa, grad2.data());
                // O treinamento não usa o gradiente dos pixels da imagem.
                l1.Backward(imagem, z1.data(), grad2.data(), taxa, nullptr);
            }
            perdas[epoca] = perda_total / treino.size();
        }
#endif
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}

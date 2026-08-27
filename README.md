# Projeto LLM — Benchmark de Quantização de Modelos de Linguagem

Trabalho prático desenvolvido como avaliação da disciplina.

---

## Objetivo

Este projeto tem como objetivo demonstrar na prática o impacto da **quantização de pesos** no desempenho de um **Large Language Model (LLM)**, comparando o modelo carregado em sua precisão original (**FP16 — 16 bits por parâmetro**) com uma versão quantizada em **INT4 (4 bits por parâmetro)**.

A quantização é uma técnica amplamente utilizada em produção para viabilizar a execução de modelos de grande escala em hardware com memória de vídeo (VRAM) limitada, sem perda significativa de qualidade nas respostas.

---

## Modelo Utilizado

| Propriedade     | Valor                      |
|-----------------|----------------------------|
| **Modelo**      | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) |
| **Parâmetros**  | 7 bilhões                  |
| **Tamanho FP16**| ~14–15 GB                  |
| **Origem**      | Hugging Face               |

---

## Tecnologias e Bibliotecas

| Biblioteca        | Finalidade                                                    |
|-------------------|---------------------------------------------------------------|
| `transformers`    | Carregamento do modelo e tokenizador via Hugging Face        |
| `bitsandbytes`    | Quantização do modelo para INT4 (`BitsAndBytesConfig`)       |
| `torch` (PyTorch) | Backend de computação, gerenciamento de GPU e geração de tokens |

**Ambiente:** Python 3.14 · Jupyter Notebook · GPU NVIDIA (CUDA)

---

## Metodologia

O experimento foi conduzido em três etapas:

### Etapa 1 — Baseline em FP16
O modelo completo é carregado com `torch_dtype=torch.float16` e `device_map="auto"`, que distribui os pesos automaticamente entre a VRAM da GPU e a RAM do sistema. Como o modelo supera a capacidade da VRAM disponível (~8 GB), parte dos pesos fica na RAM, criando um **gargalo de I/O** que degrada severamente o desempenho.

> Aviso: Durante o benchmark em FP16, ocorreram erros de `Out of Memory (OOM)` na GPU, confirmando que o hardware disponível não comporta o modelo completo nessa precisão.

### Etapa 2 — Limpeza de Memória
Após o experimento em FP16, o modelo é explicitamente deletado da memória com `del model_fp16`, seguido de `gc.collect()` e `torch.cuda.empty_cache()`, liberando a VRAM para o próximo teste.

### Etapa 3 — Modelo Quantizado em INT4
O modelo é recarregado com `BitsAndBytesConfig(load_in_4bit=True)`. Os pesos são comprimidos de 16 bits para 4 bits, reduzindo o consumo de memória em aproximadamente **4x**. As operações matemáticas intermediárias são mantidas em FP16 (`bnb_4bit_compute_dtype=torch.float16`) para preservar a qualidade das respostas.

---

## Função de Benchmark

```python
def benchmark(model, prompt, n=5):
    tok = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    for _ in range(n):
        model.generate(**tok, max_new_tokens=50)
    
    tempo_medio_ms = (time.time() - t0) * 1000 / n
    tokens_por_segundo = (50 * n) / (time.time() - t0)
    
    return tempo_medio_ms, tokens_por_segundo
```

A função executa `n` rodadas de geração de texto (cada uma com **50 novos tokens**) e calcula:
- **Latência média** (ms por rodada de inferência)
- **Throughput** (tokens gerados por segundo)

**Prompt de teste utilizado:** `"Qual o resultado da derivada de 5x²?"`

---

## Resultados

### Desempenho — INT4

| Métrica              | Valor                   |
|----------------------|-------------------------|
| **Latência média**   | 1891,43 ms / inferência |
| **Throughput**       | 26,44 tokens/segundo    |

> O baseline em FP16 não produziu resultados válidos devido ao OOM, confirmando que a quantização é **essencial** para executar modelos de 7B em GPUs com menos de 16 GB de VRAM.

### Qualidade — Avaliação Qualitativa

**Prompt:** `Qual o resultado da derivada de 5x²?`

**Resposta gerada pelo modelo INT4:**

> "A regra de derivação para uma função do tipo f(x) = ax^n é dada por f'(x) = a · n · x^(n-1). No caso da função f(x) = 5x², temos a = 5 e n = 2. Portanto, a derivada da função é: f'(x) = 5 · 2 · x^(2-1) = **10x**."

A resposta demonstra que o modelo quantizado em INT4 mantém raciocínio matemático correto, indicando que a perda de qualidade introduzida pela compressão dos pesos é mínima neste cenário.

---

## Estrutura do Projeto

```
projeto-llm/
└── ProjetoLLM.ipynb   # Notebook principal com todo o código e resultados
```

---

## Como Reproduzir

Requisito mínimo de hardware: GPU NVIDIA com ao menos 8 GB de VRAM para o modo INT4.

1. Clone o repositório e crie um ambiente virtual:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```

2. Instale as dependências:
   ```bash
   pip install transformers bitsandbytes torch accelerate
   ```

3. Execute o notebook `ProjetoLLM.ipynb` célula por célula, na ordem apresentada.

4. Na Célula 2, o download do modelo (~14–15 GB) será feito automaticamente via Hugging Face na primeira execução.

---

## Conclusão

O experimento evidencia um dos desafios centrais da inferência de LLMs em hardware de consumo: **modelos com bilhões de parâmetros simplesmente não cabem na VRAM disponível em precisão plena (FP16/BF16)**. A quantização em INT4, viabilizada pela biblioteca `bitsandbytes`, resolve este problema comprimindo os pesos em 4x e permitindo que o modelo rode inteiramente na GPU, alcançando **26,44 tokens/segundo** com qualidade de resposta preservada — como demonstrado pela resolução correta do exercício de derivada.

---

*Projeto desenvolvido para avaliação acadêmica.*

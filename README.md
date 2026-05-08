# Laboratório 09 — Pipeline RAG Avançado: HNSW + HyDE + Cross-Encoder

## Sobre o Projeto

Este laboratório implementa um pipeline de Retrieval-Augmented Generation (RAG) voltado para busca em manuais médicos. O sistema recebe uma query coloquial de um paciente e aplica três camadas de inteligência antes de entregar os documentos ao modelo gerador.

---

## Como Rodar

1. Instale as dependências:

```bash
pip install faiss-cpu sentence-transformers openai numpy
```

2. Configure a chave do OpenRouter:

```bash
export OPENROUTER_API_KEY="sua-chave-aqui"
```

3. Execute o script:

```bash
python lab9.py
```

---

## Por que OpenRouter em vez da API da OpenAI?

A API oficial da OpenAI é paga e exige cadastro de cartão de crédito. O OpenRouter oferece acesso gratuito a vários modelos de linguagem sem custo. O cliente utilizado é o mesmo SDK da OpenAI, apenas com a `base_url` apontando para o servidor do OpenRouter. O modelo configurado é `openrouter/free`, que seleciona automaticamente um modelo gratuito disponível no momento da requisição.

### Como trocar para a API do GPT

Substitua o bloco de configuração no topo do arquivo:

**De:**

```python
MINHA_CHAVE = os.environ.get("OPENROUTER_API_KEY", "")
cliente_llm = OpenAI(api_key=MINHA_CHAVE, base_url="https://openrouter.ai/api/v1")
```

**Para:**

```python
MINHA_CHAVE = os.environ.get("OPENAI_API_KEY", "")
cliente_llm = OpenAI(api_key=MINHA_CHAVE)
```

E na função `gerar_documento_hipotetico`, trocar o model para:

```python
model="gpt-3.5-turbo"
```

---

## Passo a Passo da Implementação

### Passo 1 — Base de Dados e Indexação HNSW

Foram criados 25 fragmentos de manuais médicos fictícios cobrindo cinco especialidades: Neurologia, Cardiologia, Pneumologia, Gastroenterologia e Ortopedia/Reumatologia. Cada fragmento usa terminologia clínica real para simular um manual técnico.

Os fragmentos foram convertidos em vetores densos usando o modelo `paraphrase-multilingual-MiniLM-L12-v2` da biblioteca sentence-transformers, que suporta português nativamente. Os vetores foram normalizados para permitir o uso de produto interno como equivalente ao cosseno.

Em seguida, foi construído um índice HNSW com FAISS usando os hiperparâmetros M=32 e ef_construction=200.

Saída dessa etapa:

```
Base carregada: 25 fragmentos
Embeddings gerados em 0.96s | dimensao: 384
Indice HNSW pronto | docs: 25 | M: 32 | ef_construction: 200
```

### Passo 2 — HyDE: Transformação da Query

A função `gerar_documento_hipotetico` recebe a query coloquial do paciente e envia ao LLM com uma instrução para reescrever os sintomas em linguagem técnica clínica. O texto gerado é vetorizado e usado como âncora de busca no lugar da query original.

Exemplo para a query `"dor de cabeça latejante com a luz incomodando muito e enjoo"`:

```
Documento hipotetico (HyDE):
------------------------------------------------------------
Paciente apresenta cefaleia pulsátil de intensidade moderada a alta, com piora
em ambientes iluminados (fotofobia) e acompanhada de náusea, sem relato de
vômito ou sinais de alarme neurológico.
------------------------------------------------------------
```

### Passo 3 — Busca no Índice HNSW

Com o vetor HyDE em mãos, é feita uma busca de similaridade de cosseno no índice HNSW, recuperando os 10 documentos mais próximos. Essa é a etapa do funil largo: rápida, mas ainda com algum ruído.

Saída para a query de cefaleia:

```
Top-10 recuperados via HNSW
#01 | cosseno: 0.7175 — hipertensão intracraniana idiopática...
#02 | cosseno: 0.7118 — cefaleia em salvas...
#03 | cosseno: 0.6781 — cefaleia pulsátil associada a fotofobia...
```

### Passo 4 — Reranking com Cross-Encoder

Os 10 candidatos são passados pelo modelo `cross-encoder/ms-marco-MiniLM-L-6-v2`, que avalia cada par (query, documento) em conjunto. Os documentos são reordenados e os 3 melhores selecionados para injeção no contexto do LLM gerador.

Saída para a query de coração disparado:

```
Top-3 final apos Cross-Encoder
Query: 'meu coracao fica disparado e irregular, fico com falta de ar'

Posicao 1
  cross-encoder : -3.4667
  cosseno hnsw  : 0.5905
  texto: Palpitações irregulares com frequência ventricular variável ao ECG
         sugerem fibrilação atrial, arritmia mais prevalente na prática clínica.
```

### Comparativo HyDE vs Busca Direta

```
Sem HyDE:
  [0.6598] Pacientes com cefaleia em salvas...
  [0.6338] Cefaleia pulsátil associada a fotofobia...
  [0.6237] Hipertensão intracraniana idiopática...

Com HyDE:
  [0.8233] Cefaleia pulsátil associada a fotofobia...
  [0.7126] Pacientes com cefaleia em salvas...
  [0.6493] Hipertensão intracraniana idiopática...
```

---

## Hiperparâmetros HNSW: M e ef_construction vs KNN Exato

O KNN exato carrega todos os vetores na RAM e percorre todos eles a cada busca, custando O(n) operações. Para 1 milhão de documentos com vetores de 768 dimensões, só os vetores já ocupam cerca de 3 GB de RAM.

O HNSW organiza os vetores em um grafo de múltiplas camadas atingindo O(log n) sem calcular distância com todos os documentos.

**M** define quantas conexões cada nó mantém no grafo. Valores maiores aumentam o recall mas também o consumo de RAM e o tempo de construção do índice. Com M=32, cada nó armazena aproximadamente 32 arestas, adicionando cerca de 256 bytes por nó além do próprio vetor.

**ef_construction** controla a fila de candidatos durante a construção. Valores altos resultam em grafo mais preciso, mas o custo é apenas na construção, não em produção.

**efSearch** (64) controla o trade-off entre velocidade e recall na busca, sem impacto na memória.

---

## Erros Encontrados Durante o Desenvolvimento

### Erro 1 — OpenAIError: chave não encontrada

```
OpenAIError: The api_key client option must be set either by passing api_key
to the client or by setting the OPENAI_API_KEY environment variable
```

**Causa:** a chave foi configurada com `os.environ.setdefault("lab9", ...)` em vez de `"OPENAI_API_KEY"`. Além disso, a chave era do OpenRouter, exigindo também o `base_url`.

**Solução:** passar a chave diretamente no cliente com `api_key=MINHA_CHAVE` e adicionar `base_url="https://openrouter.ai/api/v1"`.

---

### Erro 2 — NotFoundError 404: modelo não encontrado

```
NotFoundError: Error code: 404 - No endpoints found for mistralai/mistral-7b-instruct:free.
```

**Causa:** o modelo foi removido ou ficou indisponível no OpenRouter. O mesmo aconteceu com `google/gemma-3-4b-it:free`.

**Solução:** trocar para `openrouter/free`, roteador automático que seleciona qualquer modelo gratuito disponível no momento.

---

### Erro 3 — AttributeError: NoneType no conteúdo da resposta

```
AttributeError: 'NoneType' object has no attribute 'strip'
```

**Causa:** o roteador `openrouter/free` selecionou um modelo de reasoning que retorna a resposta no campo `reasoning` em vez de `content`, deixando `content=None`.

**Solução:**

```python
mensagem = resposta.choices[0].message
conteudo = mensagem.content or ""
if not conteudo:
    conteudo = getattr(mensagem, "reasoning", "") or ""
doc_hipotetico = conteudo.strip()
```

---

### Erro 4 — TypeError: argumento verbose inesperado

```
TypeError: vetorizar_hyde() got an unexpected keyword argument 'verbose'
```

**Causa:** a função `vetorizar_hyde` não tinha o parâmetro `verbose` na assinatura.

**Solução:** adicionar `verbose=True` na assinatura e repassar para `gerar_documento_hipotetico`.

---

## Sobre os Scores do Cross-Encoder

Os scores são valores brutos, não normalizados. É normal que apareçam negativos. O que importa é a ordem relativa. Na query do coração, o primeiro colocado ficou em -3.46 enquanto os demais ficaram abaixo de -8, confirmando que o reranking funcionou corretamente.

---

## Declaração de Integridade Acadêmica

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por **Gabriel Linard**.

O uso de ferramentas de IA generativa se restringiu a brainstorming da estrutura do pipeline, geração dos 25 fragmentos de manuais médicos fictícios e templates iniciais de código, todos revisados e validados criticamente pelo autor antes da entrega.
import time
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import os

MINHA_CHAVE = os.getenv("OPENROUTER_API_KEY")

cliente_llm = OpenAI(
    api_key=MINHA_CHAVE,
    base_url="https://openrouter.ai/api/v1",
) 

fragmentos_medicos = [
    "Cefaleia pulsátil associada a fotofobia e fonofobia é critério diagnóstico para migrânea sem aura, conforme a Classificação Internacional das Cefaleias (ICHD-3).",
    "Pacientes com cefaleia em salvas relatam dor unilateral periorbital intensa, geralmente acompanhada de lacrimejamento ipsilateral e congestão nasal.",
    "A hipertensão intracraniana idiopática manifesta-se por cefaleia difusa progressiva, diplopia e papiledema ao exame fundoscópico.",
    "Déficit neurológico focal de instalação aguda, incluindo hemiplegia, afasia ou amaurose fugaz, deve ser investigado como acidente vascular cerebral isquêmico.",
    "A síndrome de Guillain-Barré apresenta fraqueza muscular ascendente, arreflexia e dissociação albumino-citológica no líquor.",
    "Dor precordial opressiva irradiada para o membro superior esquerdo, associada a sudorese fria e náuseas, é apresentação clássica de síndrome coronariana aguda.",
    "Dispneia paroxística noturna e ortopneia são sintomas indicativos de insuficiência cardíaca congestiva com congestão pulmonar.",
    "Palpitações irregulares com frequência ventricular variável ao ECG sugerem fibrilação atrial, arritmia mais prevalente na prática clínica.",
    "O intervalo QT prolongado no eletrocardiograma eleva o risco de taquicardia ventricular polimórfica do tipo Torsades de Pointes.",
    "Edema maleolar bilateral de progressão ascendente, associado a turgência jugular, é sinal de insuficiência cardíaca direita descompensada.",
    "Dispneia progressiva aos esforços, tosse produtiva crônica e histórico tabágico superior a 10 maços-ano configuram critérios clínicos para DPOC.",
    "Sibilância expiratória difusa com melhora após broncodilatador é achado característico da asma brônquica em exacerbação aguda.",
    "Derrame pleural unilateral com líquido exsudativo e células neoplásicas ao citopatológico é sugestivo de pleurite carcinomatosa.",
    "Consolidação pulmonar segmentar com broncograma aéreo e febre elevada orienta para diagnóstico de pneumonia bacteriana lobar.",
    "Hemoptise volumosa recorrente em paciente imunossuprimido exige exclusão de aspergilose pulmonar invasiva por tomografia de alta resolução.",
    "Epigastralgia em queimação pós-prandial com melhora a antiácidos é sintoma típico de doença do refluxo gastroesofágico (DRGE).",
    "Diarreia sanguinolenta associada a tenesmo e febre baixa pode indicar retocolite ulcerativa em atividade, necessitando colonoscopia diagnóstica.",
    "Icterícia colestática com hiperbilirrubinemia direta, colúria e acolia fecal orienta para obstrução biliar extra-hepática.",
    "Dor em fossa ilíaca direita de início periumbilical migratória, com sinal de Blumberg positivo, é quadro clínico clássico de apendicite aguda.",
    "Ascite de grande monta com gradiente de albumina soro-ascite superior a 1,1 g/dL é critério diagnóstico de hipertensão portal.",
    "Artralgia migratória assimétrica em grandes articulações, associada a eritema marginado e febre, é critério maior de Jones para febre reumática.",
    "Rigidez matinal articular com duração superior a uma hora, sinovite simétrica em metacarpofalângicas e fator reumatoide positivo indicam artrite reumatoide.",
    "Dor lombar de caráter inflamatório, com melhora ao movimento e piora em repouso, associada a sacroileíte bilateral é critério para espondilite anquilosante.",
    "Monoartrite aguda em primeira metatarsofalângica, com cristais de urato monossódico na artrocentese, confirma diagnóstico de gota gotosa.",
    "Fratura por estresse em atletas de alto impacto manifesta-se como dor localizada progressiva com edema focal, confirmada por cintilografia óssea.",
]

print(f"Base carregada: {len(fragmentos_medicos)} fragmentos\n")

modelo_embedding = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

t0 = time.time()
vetores_base = modelo_embedding.encode(
    fragmentos_medicos,
    show_progress_bar=True,
    normalize_embeddings=True
).astype("float32")
print(f"Embeddings gerados em {time.time() - t0:.2f}s | dimensao: {vetores_base.shape[1]}\n")

dimensao = vetores_base.shape[1]
M = 32
ef_construction = 200

indice_hnsw = faiss.IndexHNSWFlat(dimensao, M, faiss.METRIC_INNER_PRODUCT)
indice_hnsw.hnsw.efConstruction = ef_construction
indice_hnsw.hnsw.efSearch = 64
indice_hnsw.add(vetores_base)

print(f"Indice HNSW pronto | docs: {indice_hnsw.ntotal} | M: {M} | ef_construction: {ef_construction}\n")

def gerar_documento_hipotetico(pergunta, verbose=True):
    instrucao = (
        "Você é um especialista em medicina clínica. "
        "Ao receber sintomas em linguagem coloquial, escreva um parágrafo curto "
        "no estilo de um manual médico usando terminologia clínica adequada. "
        "Não dê diagnóstico definitivo, apenas documente o caso tecnicamente. "
        "Responda apenas com o parágrafo."
    )

    resposta = cliente_llm.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "system", "content": instrucao},
            {"role": "user", "content": pergunta},
        ],
        temperature=0.3,
        max_tokens=180,
    )
    mensagem = resposta.choices[0].message
    conteudo = mensagem.content or ""
    if not conteudo:
        conteudo = getattr(mensagem, "reasoning", "") or ""
    doc_hipotetico = conteudo.strip()

    if verbose:
        print("Documento hipotetico (HyDE):")
        print("-" * 60)
        print(doc_hipotetico)
        print("-" * 60 + "\n")

    return doc_hipotetico


def vetorizar_hyde(pergunta, verbose=True):
    doc_hip = gerar_documento_hipotetico(pergunta, verbose=verbose)
    vetor = modelo_embedding.encode(
        [doc_hip], normalize_embeddings=True
    ).astype("float32")
    return vetor, doc_hip

def buscar_no_hnsw(vetor_consulta, top_k=10):
    scores, indices = indice_hnsw.search(vetor_consulta, top_k)

    resultados = []
    for pos, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx != -1:
            resultados.append({
                "posicao": pos + 1,
                "score_cosseno": float(score),
                "texto": fragmentos_medicos[idx],
            })
    return resultados

def imprimir_candidatos(candidatos):
    print("=" * 65)
    print(f"  Top-{len(candidatos)} recuperados via HNSW")
    print("=" * 65)
    for doc in candidatos:
        print(f"\n#{doc['posicao']:02d} | cosseno: {doc['score_cosseno']:.4f}")
        print(f"    {doc['texto'][:110]}...")
    print()

print("Carregando Cross-Encoder...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-Encoder pronto\n")


def reranquear(query_original, candidatos, top_final=3):
    pares = [[query_original, doc["texto"]] for doc in candidatos]
    scores_profundos = cross_encoder.predict(pares)

    for doc, score in zip(candidatos, scores_profundos):
        doc["score_cross_encoder"] = float(score)

    reordenados = sorted(candidatos, key=lambda x: x["score_cross_encoder"], reverse=True)
    return reordenados[:top_final]


def imprimir_top3(top3, query):
    print("=" * 65)
    print(f"Top-3 final apos Cross-Encoder")
    print(f"Query: '{query}'")
    print("=" * 65)
    for i, doc in enumerate(top3, 1):
        print(f"\nPosicao {i}")
        print(f"cross-encoder : {doc['score_cross_encoder']:.4f}")
        print(f"cosseno hnsw  : {doc['score_cosseno']:.4f}")
        print(f"texto         : {doc['texto']}")
    print("\nEsses 3 fragmentos seriam injetados no contexto do LLM gerador.\n")


def pipeline_rag_medico(pergunta, k_busca=10, k_final=3):
    print("\n" + "=" * 65)
    print(f"  Query: '{pergunta}'")
    print("=" * 65 + "\n")

    print("[1/3] Gerando documento hipotetico (HyDE)...")
    vetor_consulta, doc_hip = vetorizar_hyde(pergunta)

    print(f"[2/3] Buscando top-{k_busca} no indice HNSW...")
    candidatos = buscar_no_hnsw(vetor_consulta, top_k=k_busca)
    imprimir_candidatos(candidatos)

    print(f"[3/3] Rerankeando com Cross-Encoder, selecionando top-{k_final}...")
    top_final = reranquear(pergunta, candidatos, top_final=k_final)
    imprimir_top3(top_final, pergunta)

    return {
        "query_original": pergunta,
        "documento_hipotetico": doc_hip,
        "candidatos_hnsw": candidatos,
        "top_final": top_final,
    }


resultado1 = pipeline_rag_medico(
    "dor de cabeça latejante com a luz incomodando muito e enjoo"
)

resultado2 = pipeline_rag_medico(
    "meu coracao fica disparado e irregular, fico com falta de ar"
)

resultado3 = pipeline_rag_medico(
    "dor no estomago que queima bastante depois de comer"
)

print("\n" + "=" * 65)
print("  Comparativo: busca direta vs HyDE")
print("=" * 65)

query_exp = "dor de cabeça latejante com a luz incomodando muito e enjoo"

vetor_direto = modelo_embedding.encode(
    [query_exp], normalize_embeddings=True
).astype("float32")
sem_hyde = buscar_no_hnsw(vetor_direto, top_k=3)

print("\nSem HyDE:")
for d in sem_hyde:
    print(f"  [{d['score_cosseno']:.4f}] {d['texto'][:90]}...")

print("\nCom HyDE:")
vetor_h, _ = vetorizar_hyde(query_exp, verbose=False)
com_hyde = buscar_no_hnsw(vetor_h, top_k=3)
for d in com_hyde:
    print(f"  [{d['score_cosseno']:.4f}] {d['texto'][:90]}...")
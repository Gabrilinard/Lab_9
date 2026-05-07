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
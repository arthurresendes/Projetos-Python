import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from google_play_scraper import Sort, reviews, app
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict, Counter
import re


# 1 --- Configs Iniciais

RANDOM_SEED = 42 # Numero magico na CD, garante reprodutilidade
np.random.seed(RANDOM_SEED) # Controla aleatoriamente os números gerados
torch.manual_seed(RANDOM_SEED) # Controla a aleatoriedade , modelo começa com os mesmo peso

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Para saber se usa CPU ou GPU
print(f"Dispositivo sendo usado: {device}")

sns.set(style='whitegrid', palette='muted', font_scale=1.2) # Estilo de graficos padrão


# 2 --- Coletando dados

jogos_ids = [
    "com.supercell.clashroyale",
    "com.dts.freefireth",
    "com.supercell.brawlstars",
    "com.tencent.ig",
    "br.pipacombate.maiworm",
    "com.duolingo",
    "com.kiloo.subwaysurf",
    "air.com.hypah.io.slither"
]

print("\nColetando informações dos aplicativos")


aplic_infos = []
for aplic in tqdm(jogos_ids, desc="Apps"):
    try:
        info = app(aplic, lang="pt", country="br")
        info.pop('comments', None)
        aplic_infos.append(info)
    except Exception as e:
        print(f"Erro ao coletar {aplic}: {e}")
        continue

dataframe_jogos = pd.DataFrame(aplic_infos)
print(f"{len(dataframe_jogos)} aplicativos coletados")
print(dataframe_jogos[['title', 'score', 'installs']].head(3))


# 3 - Coletando reviews
'''
Dados brutos: Textos que usuários escreveram
Metadados: Notas (1-5 estrelas), data, relevância
Diversidade: Reviews recentes + relevantes + todas as nota
'''

aplic_reviews = []
print("\nColetando reviews dos usuários")

for aplic_id in tqdm(jogos_ids, desc="Reviews"):
    # Passar por notas de 1 a 5
    for pontuacao in range(1,6):
        # Mais relevantes e mais recentes para gerar mais diversidade
        for ordem_ordenada in [Sort.MOST_RELEVANT, Sort.NEWEST]:
            try:
                import time
                time.sleep(1)
                count = 200 if pontuacao == 3 else 100 # Reviews neutros são mais raros, mas importantes para o modelo aprender, ou seja sempre 3 será mais valioso na hora de treinar
                
                # rvs -> Lista dos reviews e _ ignora o token de continuação (paginação) e reviews vem da playstore passando os devidos parametros (apli-aplicativo, linguagem , pais , sort(ordenação), contagem de pontuação , e a filtragem)
                rvs, _ = reviews(
                    aplic_id,
                    lang='pt',
                    country='br',
                    sort=ordem_ordenada,
                    count=count,
                    filter_score_with=pontuacao
                )
                
                for r in rvs:
                    r['sortOrder'] = 'mais_relevante' if ordem_ordenada == Sort.MOST_RELEVANT else 'mais_recente'
                    r['appId'] = aplic_id
                aplic_reviews.extend(rvs)
            except Exception as e:
                print(f"Erro {aplic_id} com avaliação {pontuacao}: {e}")
                continue

df = pd.DataFrame(aplic_reviews)
print(f"Total de reviews coletados: {len(df)}")


# 4 - Analise de sentimento (Dash)

plt.figure(figsize=(10,5)) # # Cria figura de 10x5 polegadas
sns.countplot(x='score', data=df, palette='viridis') #Pega a coluna score do df viridis esquema verde e azulado
plt.xlabel('Notas do Reviews') # Label x
plt.ylabel('Quantidade') # Label y
plt.title('Distribuição das Notas dos Usuários')
plt.tight_layout() # Ajusta espaçamento automático
plt.savefig('distribuicao_notas.png', dpi=300, bbox_inches='tight')
plt.show()


# 5 - Mapear os sentimentos

def mapeando_sentimentos(avaliacao):
    avaliacao = int(avaliacao)
    if avaliacao <=2:
        return 0
    elif avaliacao == 3:
        return 1
    else:
        return 2

df['sentiment'] = df['score'].apply(mapeando_sentimentos)
nomes_classes = ['Negativo', 'Neutro', 'Positivo']

plt.figure(figsize=(8,5))
ax = sns.countplot(x='sentiment', data=df, palette='Set2')
ax.set_xticks(range(len(nomes_classes)))
ax.set_xticklabels(nomes_classes)
plt.xlabel("Sentimento")
plt.ylabel("Quantidade")
plt.title("Distribuição dos Sentimentos")

# ax.patches são as 3 barras do graficos enumeradas, faz os calculos para aparecer o numero na barra
for i , p in enumerate(ax.patches):
    altura = p.get_height()
    ax.text(p.get_x() +  p.get_width()/2., altura + 50,f'{int(altura)}', ha="center", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('distribuicao_sentimentos.png', dpi=300, bbox_inches='tight')
plt.show()

df.to_csv('reviews.csv', index=False)
print("Dados salvos em 'reviews.csv'")

# 6 - PRÉ processamento de texto

'''
Limpagem de texto preparando para o modelo
Adicionando nova coluna no df onde tem essa limpeza que é aplicado no content(comentarios) e reviews com menos de 10 caracteres são removidas por serem curtas
'''

def limpagem(texto):
    # Protege contra valores nulos/NaN
    if pd.isna(texto):
        return ""
    texto = str(texto).lower() # Deixando minusculo
    texto = re.sub(r'http\S+|www\S+', '', texto)  # Remove URLs
    texto = re.sub(r'@\w+', '', texto)  # Remove menções
    texto = re.sub(r'[^\w\s]', ' ', texto)  # Remove pontuação
    texto = re.sub(r'\s+', ' ', texto).strip()  # Remove espaços extras
    return texto

df['content_clean'] = df['content'].apply(limpagem)
df = df[df['content_clean'].str.len() > 10] # Remove comentarios/reviews curtos
print(f"Reviews após limpeza: {len(df)}")

# 7 -- Construindo vocabulario

TAMANHO_MAX_VOCAB = 5000
MAX_LEN = 50 # Comprimento maximo de sequencia

def construir_vocabulario(texts, max_size):
    contagem_palavras = Counter()
    for text in texts:
        contagem_palavras.update(text.split())
        
    vocabulario = {'<PAD>': 0, '<UNK>': 1}
    for palavra , _ in contagem_palavras.most_common(max_size - 2):
        vocabulario[palavra] = len(vocabulario)
    return vocabulario

print("Construindo vocabulário.")
vocabulario = construir_vocabulario(df['content_clean'], TAMANHO_MAX_VOCAB)
print(f"Vocabulário criado com {len(vocabulario)} palavras")

def sequencia_texto(text,vocabulario,max_len):
    palavras = text.split()[:max_len]
    sequencia = [vocabulario.get(palavra, vocabulario['<UNK>']) for palavra in palavras]
    
    sequencia += [vocabulario['<PAD>']] * (max_len - len(sequencia))
    return sequencia
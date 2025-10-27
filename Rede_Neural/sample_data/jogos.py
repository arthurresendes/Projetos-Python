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
        contagem_palavras.update(text.split()) # Conta por cada palavra
        
    vocabulario = {'<PAD>': 0, '<UNK>': 1} # Pad -> texto é curto e o UNK é quando a palavra é desconhecida
    for palavra , _ in contagem_palavras.most_common(max_size - 2):
        vocabulario[palavra] = len(vocabulario) # Adiciona com índice crescente dos mais frequentes
    return vocabulario

print("Construindo vocabulário.")
vocabulario = construir_vocabulario(df['content_clean'], TAMANHO_MAX_VOCAB)
print(f"Vocabulário criado com {len(vocabulario)} palavras")

def sequencia_texto(text,vocabulario,max_len):
    palavras = text.split()[:max_len] # Pega no máximo MAX_LEN palavras
    sequencia = [vocabulario.get(palavra, vocabulario['<UNK>']) for palavra in palavras]
    
    sequencia += [vocabulario['<PAD>']] * (max_len - len(sequencia))
    return sequencia

'''
# Supondo vocabulário:

Ou seja PAD preenche até o tamanho com 0 e UNK quando uma palavra não é encontrada no vocabulario

vocabulario = {
    '<PAD>':0, '<UNK>':1, 'jogo':2, 'bom':3, 'ruim':4, 'esse':5
}

# "esse jogo é demais" → [5, 2, 1, 1] + padding
# Sequência final: [5, 2, 1, 1, 0, 0, 0, ...] (até 50 números)
'''

# 8 DataSet PYTHOCH

class RevisarDataset(Dataset):
    def __init__(self,textos,sentimentos,vocabulario,max_len):
        self.textos = textos
        self.sentimentos = sentimentos
        self.vocabulario = vocabulario
        self.max_len = max_len
    
    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, index):
        text = self.textos.iloc[index]
        sentiment = self.sentimentos.iloc[index]
        
        sequencia = sequencia_texto(text, self.vocabulario, self.max_len) # Converte para Sequência
        
        # Retorna um dicionario da sequencia e um do sentimentos
        return {
            'sequence': torch.tensor(sequencia, dtype=torch.long),
            'sentiment': torch.tensor(sentiment, dtype=torch.long)
        }


# 9 Divisão dos dados 

"""
Divide dados e prepara para treinamento em lotes

Divide dados em treino (80%), validação (10%) e teste (10%)
Usa stratify para manter proporção de classes

df (100%) 
├── df_train (80%) - Treino
└── df_temp (20%)
    ├── df_val (10%) - Validação  
    └── df_test (10%) - Teste

DATALOADER - O QUE FAZ?

Divide datasets em batches (lotes)
shuffle=True no treino: Embaralha dados a cada época
Batch = grupo de exemplos processados juntos


"""

df_treino, df_temp = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['sentiment'])

df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=df_temp['sentiment'])

print(f"\nDivisão dos dados:")
print(f"  Treino: {len(df_treino)} ({len(df_treino)/len(df)*100:.1f}%)")
print(f"  Validação: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)")
print(f"  Teste: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")

BATCH_SIZE = 64

treino_dataset = RevisarDataset(df_treino['content_clean'], df_treino['sentiment'], vocabulario, MAX_LEN)
valid_dataset = RevisarDataset(df_valid['content_clean'], df_valid['sentiment'], vocabulario, MAX_LEN)
test_dataset = RevisarDataset(df_test['content_clean'], df_test['sentiment'], vocabulario, MAX_LEN)

treino_loader = DataLoader(treino_dataset, batch_size=BATCH_SIZE,shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
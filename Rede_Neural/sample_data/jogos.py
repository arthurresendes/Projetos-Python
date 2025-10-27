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



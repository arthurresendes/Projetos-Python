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

# 1 -- Conifgs Iniciais

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
sns.set(style='whitegrid', palette='muted', font_scale=1.2) 


# 2 - Coletando dados

aplicativos_play = [
    "br.com.drogaraia",
    "br.com.rdsaude.healthPlatform.android",
    "br.com.raiadrogasil.apps.painelrd",
    "br.com.drogasil"
]

print("\nColetando informações dos aplicativos")

informacoes_apps = []
for aplic in tqdm(aplicativos_play, desc="Apps"):
    try:
        info = app(aplic, lang="pt", country="br")
        info.pop('comments', None)
        informacoes_apps.append(info)
    except Exception as e:
        print(f"Erro ao coletar {aplic}: {e}")
        continue

dataframe_app = pd.DataFrame(informacoes_apps)
print(f"{len(dataframe_app)} aplicativos coletados")
print(dataframe_app[['title', 'score', 'installs']].head(4))

# 3 - Reviews

reviews_app = []

for app_id in tqdm(aplicativos_play, desc="Reviews"):
    for pontuacao in range(1,6):
        for ordenacao in [Sort.MOST_RELEVANT, Sort.NEWEST]:
            try:
                import time
                time.sleep(1)
                count = 200 if pontuacao == 3 else 100
                
                rvs,_ = reviews(
                    app_id,
                    lang='pt',
                    country='br',
                    sort=ordenacao,
                    count=count,
                    filter_score_with=pontuacao
                )
                
                for r in rvs:
                    r['sortOrder'] = 'mais_relevante' if ordenacao == Sort.MOST_RELEVANT else 'mais_recente'
                    r['appId'] = app_id
                reviews_app.extend(rvs)
            except Exception as e:
                print(f"Erro {app_id} com avaliação {pontuacao}: {e}")
                continue

df = pd.DataFrame(reviews_app)

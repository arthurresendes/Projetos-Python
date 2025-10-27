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


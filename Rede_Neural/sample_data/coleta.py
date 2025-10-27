import json # Trabalha com {} , permitindo manipulações
import pandas as pd # Utilizado para tabelas, manipulação de dados , filtragem , etc
from tqdm import tqdm # Barra de progresso para loop longos mostrando pro user quanto falta para devida execução
import seaborn as sns # Biblioteca de visualização baseado no matlotlib com estilo mais bonito
import matplotlib.pyplot as plt # Biblioteca padrão de graficos
from pygments import highlight # Biblioteca para formatar json
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter # Formata json de forma legivel no terminal
from google_play_scraper import Sort, reviews, app # Coleta dados da PlayStore sem precisar de api oficial

sns.set(style='whitegrid', palette='muted', font_scale=1.2) # Estilo do grafico com fundo branco com grade discreta , cores suaves e font 

apps_ids = ['br.com.brainweb.ifood', 'com.cerveceriamodelo.modelonow',
'com.mcdo.mcdonalds', 'habibs.alphacode.com.br',
'com.ubercab.eats', 'com.grability.rappi','burgerking.com.br.appandroid','com.vanuatu.aiqfome'] # apps vindo da playstore

app_infos = [] # Armazena as informações dos apps no append
for ap in tqdm(apps_ids):
    try:
        info = app(ap,lang='pt', country='br')
        del info['comments'] # Deleta os comentarios do app
        app_infos.append(info)
    except:
        pass

app_infos_df = pd.DataFrame(app_infos) # Dataframe com as informações do app
print(app_infos_df.head(7)) # Mostra as primeiras 7 linhas dos app


app_reviews = [] # Vai coletar as revisões
for ap in tqdm(apps_ids):
    for score in list(range(1,6)): # Itera por notas possiveis de 1 ate 5
        for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]: # sort.Most_relevant = mais relevante , sort.newst = mais recente
            
            # Coleta reviews para o app atual e passa os parametros como linguagem , contagem  para balancear o dataset, etc
            rvs,_ = reviews(
                ap,
                lang='pt',
                country='br',
                sort=sort_order,
                count=400 if score == 3 else 200,
                filter_score_with=score
                )
            for r in rvs:
                r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest' # Marca se a review é mais relevante ou recente
                r['appId'] = ap # qual app a review pertence
            app_reviews.extend(rvs) # Adiciona todas reviews a lista principal

df = pd.DataFrame(app_reviews) # cria um dataframe das reviews
sns.countplot(df.score) # grafico das notas
plt.xlabel('review score') # label do grafico

# Avaliação dos sentimentos
def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

# Aplica a dunção a cada linha/coluna para transformar dados sem loops explicitos
df['sentiment'] = df.score.apply(to_sentiment)

class_names = ['negative', 'neutral' ,'positive']

ax = sns.countplot(df.sentiment)
plt.xlabel("review sentiment")
ax.set_xticklabels(class_names)
df.to_csv('reviews.csv', index=False, header=True) # Salva informações em umcsv
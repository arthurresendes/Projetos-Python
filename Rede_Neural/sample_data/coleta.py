import json # Trabalha com {} , permitindo manipulações
import pandas as pd # Utilizado para tabelas, manipulação de dados , filtragem , etc
from tqdm import tqdm # Barra de progresso para loop longos mostrando pro user quanto falta para devida execução
import seaborn as sns # Biblioteca de visualização baseado no matlotlib com estilo mais bonito
import matplotlib.pyplot as plt # Biblioteca padrão de graficos
from pygments import highlight # Biblioteca para formatar json
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter # Formata json de forma legivel no terminal
from google_play_scraper import Sort, reviews, app # Coleta dados da PlayStore sem precisar de api oficial

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

apps_ids = ['br.com.brainweb.ifood', 'com.cerveceriamodelo.modelonow',
'com.mcdo.mcdonalds', 'habibs.alphacode.com.br',
'com.ubercab.eats', 'com.grability.rappi','burgerking.com.br.appandroid','com.vanuatu.aiqfome']

app_infos = []
app_erros = []
for ap in tqdm(apps_ids):
    try:
        info = app(ap,lang='pt', country='br')
        del info['comments']
        app_infos.append(info)
    except:
        app_erros.append(ap)
        pass

app_infos_df = pd.DataFrame(app_infos)
print(app_infos_df.head(7))


app_reviews = []
for ap in tqdm(apps_ids):
    for score in list(range(1,6)):
        for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
            rvs,_ = reviews(
                ap,
                lang='pt',
                country='br',
                sort=sort_order,
                count=400 if score == 3 else 200,
                filter_score_with=score
                )
            for r in rvs:
                r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
                r['appId'] = ap
            app_reviews.extend(rvs)

df = pd.DataFrame(app_reviews)
sns.countplot(df.score)
plt.xlabel('review score')

def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

df['sentiment'] = df.score.apply(to_sentiment)

class_names = ['negative', 'neutral' ,'positive']

ax = sns.countplot(df.sentiment)
plt.xlabel("review sentiment")
ax.set_xticklabels(class_names)
df.to_csv('reviews.csv', index=False, header=True)

from google_play_scraper import app
# https://play.google.com/store/apps/details?id=com.supercell.clashroyale
# https://play.google.com/store/apps/details?id=com.dts.freefireth
# https://play.google.com/store/apps/details?id=com.supercell.brawlstars
# https://play.google.com/store/apps/details?id=com.tencent.ig

# Pegar oq o id recebe e para ver dados em JSON dos app da PlayStore

app_id_clash = 'com.supercell.clashroyale'
app_id_free_fire = 'com.dts.freefireth'
app_id_brawl_stars = 'com.supercell.brawlstars'
app_pubg = 'com.tencent.ig'

info_clash = app(app_id_clash, lang='pt', country='br')
info_free_fire = app(app_id_free_fire, lang='pt', country='br')
info_brawl_stars = app(app_id_brawl_stars, lang='pt', country='br')
info_pubg = app(app_pubg , lang='pt', country='br')

print(f'Jogo: {info_free_fire['title']} com a avaliação em {info_free_fire['score']}')
print(f'Jogo: {info_clash['title']} com a avaliação em {info_clash['score']}')
print(f'Jogo: {info_brawl_stars['title']} com a avaliação em {info_brawl_stars['score']}')
print(f'Jogo: {info_pubg['title']} com a avaliação em {info_pubg['score']}')


import requests

def api():
    nome = input("Digite o nome do filme: ")
    # Url da onde tem os filmes 
    url = "http://www.omdbapi.com/"
    # Parametros , chaveapi e t que pega o nome do filme
    params = {
    "apikey": "53ddc769",
    "t": nome
    }

    requisicao = requests.get(url, params=params)

    if requisicao.status_code == 200:
        try:
            dic_requisi = requisicao.json()
            print(f"Nome do filme: {dic_requisi['Title']}")
            print(f"Ano: {dic_requisi['Year']}")
            print(f"Gênero: {dic_requisi['Genre']}")
            print(f"Atores: {dic_requisi['Actors']}")
            print(f"Diretor: {dic_requisi['Director']}")
        except KeyError:
            print("Filme não encontrado!!")
    else:
        print("Erro na requisição:", requisicao.status_code)

if __name__ == "__main__":
    api()
import requests
from dotenv import load_dotenv
import os
import sys

load_dotenv()

API_KEY = os.getenv('AUTH_API_FOOTBAL')
headers = {
    "X-Auth-Token": API_KEY
}

def escolhe_liga() -> int:
    while True:
        try:
            print("="*50)
            print("Escolha uma das opções abaixo para ver a classificação no ano de 2025")
            print("1- Brasileirão")
            print("2- Premier League")
            print("3- La Liga")
            print("4- Bundesliga")
            print("5- Seria A italiana")
            print("6- Sair")
            opcao = int(input(": "))
            if opcao > 0 and opcao < 7:
                if opcao == 6:
                    sys.exit("Saindo.....")
                else:
                    return opcao
            else:
                print("Opção invalida!")
        except:
            print("Escolha uma opção valida")

escolha = escolhe_liga()
def filtro_liga() -> str:
    if escolha == 1:
        return 'BSA'
    elif escolha == 2:
        return 'PL'
    elif escolha == 3:
        return 'PD'
    elif escolha == 4:
        return 'BL1'
    elif escolha == 5:
        return 'SA'

res = filtro_liga()
url = f"https://api.football-data.org/v4/competitions/{res}/standings"
params = {"season": 2025}
requisicao = requests.get(url, headers=headers, params=params)
data = requisicao.json()

for team in data["standings"][0]["table"]:
    print(team["position"], team["team"]["name"], team["points"])
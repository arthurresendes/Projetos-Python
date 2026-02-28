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
            print("Escolha uma das opções abaixo para ver a classificação no ano de 2026")
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

def filtro_liga(escolha: int) -> str:
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

def get_leagues(): 
    escolha = escolhe_liga()
    res = filtro_liga(escolha)
    url = f"https://api.football-data.org/v4/competitions/{res}/standings"
    if res == 'BSA':
        params = {"season": 2026}
    else:
        params = {"season": 2025}
    requisicao = requests.get(url, headers=headers, params=params)
    data = requisicao.json()

    print("Posição | Time | Pontos")
    for team in data["standings"][0]["table"]:
        print(team["position"], team["team"]["name"], team["points"])

def get_specific_team():
    nome = input("Qual nome do time: ").lower()

    ligas = ['BSA', 'PL', 'PD', 'BL1', 'SA']

    for liga_code in ligas:

        url_table = f"https://api.football-data.org/v4/competitions/{liga_code}/standings"

        if liga_code == 'BSA':
            params_table = {"season": 2026}
        else:
            params_table = {"season": 2025}

        r = requests.get(url_table, headers=headers, params=params_table)
        data_table = r.json()

        for team in data_table["standings"][0]["table"]:
            if nome in team["team"]["name"].lower():

                print("="*40)
                print("Time:", team["team"]["name"])
                print("Posição:", team["position"])
                print("Pontos:", team["points"])
                print("Vitórias:", team["won"])
                print("Empates:", team["draw"])
                print("Derrotas:", team["lost"])

                aproveitamento = (team["points"] / (team["playedGames"] * 3)) * 100
                print(f"Aproveitamento: {aproveitamento:.2f}%")
                print("="*40)
                return

    print("Time não encontrado em nenhuma liga suportada.")


def menu():
    while True:
        try:
            print("="*30)
            print("1 - Ver ligas")
            print("2 - Ver time em especifico")
            print("3 - Sair")
            print("="*30)
            op = int(input(":"))
            if op == 1:
                get_leagues()
            elif op == 2:
                get_specific_team()
            elif op == 3:
                print("Saindo...")
                sys.exit(1)
            else:
                print("Digite um numero valido")
        except ValueError:
            print("Digite um numero inteiro de 1 até 3")

if __name__ == "__main__":
    menu()
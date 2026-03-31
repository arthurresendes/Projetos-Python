from dotenv import load_dotenv
import requests
import os

load_dotenv()
token = os.getenv("TOKEN_ACTIONS")

def menu():
    print("="*50)
    print("Aqui você podera ver valores de determinada ação!!")
    print("="*50)

def nome_acao() -> str:
    nome = input("Digite o codigo de identificação da ação: ").upper()
    return nome

def validacao():
    ticker = nome_acao()
    url = f"https://brapi.dev/api/quote/{ticker}?token={token}"
    response  = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        preco = data['results'][0]['regularMarketPrice']
        nome_completo = data['results'][0]['longName']
        ticked = data['results'][0]['symbol']
        print("Nome | Preço | Ticket")
        print(f"{nome_completo}: R$ {preco}. Identificação: {ticked}")
    else:
        print("Erro ao achar a ação")

if __name__ == "__main__":
    menu()
    validacao()
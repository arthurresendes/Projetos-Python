import requests
import json

requisicao = requests.get("https://economia.awesomeapi.com.br/json/last/USD-BRL")
dic_requisi = requisicao.json()
taxa_importacao = 1.15

def produto():
    nome = ""
    while not nome.strip():
        nome = input("Digite o nome do produto: ")

    while True:
        try:
            preco = float(input("Digite o preço do produto: "))
            break
        except ValueError:
            print("Preço inválido! Digite um número válido.")

    return nome, preco
nome_prod , preco_prod = produto()


def converte():
    dados = dic_requisi['USDBRL']
    number = float(dados['bid'])
    preco_total_dolar = preco_prod / number * taxa_importacao
    
    return preco_total_dolar
preco_pagar = converte()

dados_json = {
    "nome": nome_prod,
    "preco produto": preco_prod,
    "valor total dolar": preco_pagar,
    "valor total real": preco_prod * taxa_importacao
}

with open("pagamento.json", "w") as file:
    json.dump(dados_json,file,indent=4)

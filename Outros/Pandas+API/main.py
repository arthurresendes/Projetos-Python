import requests
import pandas as pd


def requisicao():
    requisicao = requests.get("https://economia.awesomeapi.com.br/json/last/USD-BRL,EUR-BRL,BTC-BRL")
    dic_requisicao = requisicao.json()
    cUSA = dic_requisicao['USDBRL']
    cEUR = dic_requisicao['EURBRL']
    cBTC = dic_requisicao['BTCBRL']
    
    valorUSA = float(cUSA['bid'])
    valorEUR = float(cEUR['bid'])
    valorBTC = float(cBTC['bid'])
    
    lista = [valorUSA,valorEUR,valorBTC]
    return lista

def ler_df():
    df = pd.read_excel("Custos.xlsx")
    return df

def conversao():
    dataframe = ler_df()
    valores = requisicao()
    
    print(dataframe)
    print(valores)


conversao()
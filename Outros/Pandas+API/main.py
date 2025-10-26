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

def dataframes(nome,totalBrl,totalMoeda):
    df = pd.DataFrame({
        'Nome': nome,
        'Total Brasil': totalBrl,
        'Total Conversão': totalMoeda
    })
    
    return df

def conversao():
    dataframe_xlsx = ler_df()
    valores = requisicao()
    
    
    total_df = dataframe_xlsx ['Total']
    nome_prods = dataframe_xlsx ['Nome']
    conversaoUSA = valores[0] * total_df
    conversaoEUR = valores[1] * total_df
    conversaoBTC = valores[2] * total_df
    
    dfUSA = dataframes(nome_prods,total_df,conversaoUSA)
    dfEUR = dataframes(nome_prods,total_df,conversaoEUR)
    dfBTC = dataframes(nome_prods,total_df,conversaoBTC)
    print(f"Conversão DOLAR:\n{dfUSA}")
    print(f"Conversão EURO:\n{dfEUR}")
    print(f"Conversão BITCOIN:\n{dfBTC}")
    dfUSA.to_excel("Custos_dolar.xlsx", index=False)
    dfEUR.to_excel("Custos_euro.xlsx",index=False)
    dfBTC.to_excel("Custos_bitcoin.xlsx",index=False)


conversao()




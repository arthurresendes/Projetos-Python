import pandas as pd

dados_grandes = [
    {"nome": "Alice", "idade": 30, "cidade": "Nova Iorque"},
    {"nome": "Bob", "idade": 25, "cidade": "Los Angeles"},
    {"nome": "Charlie", "idade": 35, "cidade": "Chicago"}
]

df = pd.DataFrame(dados_grandes)

df.to_json("dados.json")
df.to_excel("dados.xlsx")

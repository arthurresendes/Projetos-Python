import pandas as pd
import matplotlib.pyplot as plt

def dataFrame():
    global df 
    df = pd.DataFrame({
    'Data': pd.date_range(start='2025-09-01', periods=12, freq='D'),
    'Produto': ['Notebook', 'Celular', 'Mouse', 'Teclado', 'Monitor', 'Fone', 'Cadeira', 'Mesa', 'Geladeira', 'TV', 'Impressora', 'Tablet'],
    'Categoria': ['Eletrônicos','Eletrônicos','Eletrônicos','Eletrônicos','Eletrônicos','Eletrônicos',
                  'Móveis','Móveis','Eletrodoméstico','Eletrodoméstico','Eletrônicos','Eletrônicos'],
    'Preço': [3500, 2500, 150, 200, 1200, 300, 400, 800, 2800, 3200, 900, 1500],
    'Quantidade': [10, 25, 50, 30, 20, 60, 15, 10, 8, 12, 5, 7],
    'Vendedor': ['Arthur','Bruno','Carlos','Daniel','Eduardo','Fernanda','Gabriel','Helena','Isabela','João','Arthur','Lucas']})

def relatorio():
    global relatorio_vendas
    relatorio_vendas  = df.copy()
    relatorio_vendas['Total'] = relatorio_vendas['Preço'] * relatorio_vendas['Quantidade']
    qtd_vendedor = relatorio_vendas.value_counts('Vendedor')
    qtd_produto = relatorio_vendas.value_counts('Produto')
    pivot_table = relatorio_vendas.pivot_table(values="Total", index='Data', columns="Categoria")
    pivot_table.fillna(value=0, inplace=True)
    
    print(f"Relatorio:\n{relatorio_vendas}\n")
    print(f"Total de vendas por mês e categoria:\n{pivot_table}\n")
    print(f"Quantidade de vendas:\n{qtd_vendedor}\n")
    print(f"Quantidade de produtos:\n{qtd_produto}\n")

def dash():
    total_categoria = relatorio_vendas.groupby('Categoria')['Total'].sum()
    plt.figure(figsize=(6,6))
    plt.pie(total_categoria, labels=total_categoria.index, autopct="%1.1f%%", startangle=90, shadow=True)
    plt.title("Total faturado por categoria")
    plt.show()
    

def main():
    dataFrame()
    relatorio()
    dash()


if __name__ == "__main__":
    main()
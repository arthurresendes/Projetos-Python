import sqlite3

def insercao_pessoa(nome , cpf):
    conexao = sqlite3.connect("compra.db")
    cursor = conexao.cursor()
    
    pessoa = (nome,cpf)
    cursor.execute("INSERT INTO pessoa(nome_pessoa,CPF) VALUES(?,?)", pessoa)
    
    conexao.commit()
    conexao.close()

def selecao_produtos():
    conexao = sqlite3.connect("compra.db")
    cursor = conexao.cursor()
    
    cursor.execute("SELECT nome_prod, quantidade, preco_unitario FROM produto")
    result = cursor.fetchall()
    for indice,valor in enumerate(result):
        print(f"Indice: {indice+1} Informações: {valor}")
    
    conexao.commit()
    conexao.close()

def atualizar_por_indice(qtd,indice):
    conexao = sqlite3.connect("compra.db")
    cursor = conexao.cursor()
    
    valores = (qtd,indice)
    cursor.execute("UPDATE produto SET quantidade = quantidade - ? where id_produto = ?", valores)
    
    conexao.commit()
    conexao.close()

def selecionar_por_indice(indice):
    conexao = sqlite3.connect("compra.db")
    cursor = conexao.cursor()
    
    valor = (indice,) 
            
    sql_query = """
            SELECT 
                nome_prod, 
                quantidade, 
                preco_unitario 
            FROM 
                produto 
            WHERE 
                id_produto = ?
            """
            
    cursor.execute(sql_query, valor)
    result = cursor.fetchall()
    for i,valor in enumerate(result):
        print("Valor atualizado em estoque: ")
        print(f"Indice: {i+1} Informações: {valor}")
    
    conexao.commit()
    conexao.close()

def atualizar_pedido(indice):
    conexao = sqlite3.connect("compra.db")
    cursor = conexao.cursor()
    cursor.execute("SELECT max(id_pessoa) FROM pessoa") # Pega o id maximo , como é auto increment o ultimo a ser adicionado será o maximo
    res = cursor.fetchall()
    id_pessoa = res[0][0] if res else None
    valores = (id_pessoa,indice)
    cursor.execute("INSERT INTO pedido(id_pessoa,id_prod) VALUES(?,?)", valores)
    
    conexao.commit()
    conexao.close()


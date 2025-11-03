import sqlite3

def insercao_pessoa(nome , cpf):
    conexao = sqlite3.connect("compra.db")
    cursor = conexao.cursor()
    
    pessoa = (nome,cpf)
    cursor.execute("INSERT INTO pessoa(nome_pessoa,CPF) VALUES(?,?)", pessoa)
    
    conexao.commit()
    conexao.close()

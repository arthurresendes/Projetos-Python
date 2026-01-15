import sqlite3
import os

diretorio_atual = os.getcwd()
caminho_completo = os.path.join(diretorio_atual, "ChatBot_Banco", 'banco.db')

def criacao():
    conexao = sqlite3.connect(caminho_completo)
    cursor = conexao.cursor()

    cursor.execute('''
               CREATE TABLE Funcionario(
                   ID INTEGER PRIMARY KEY AUTOINCREMENT,
                   nome VARCHAR(100),
                   idade INTEGER,
                   SALARIO FLOAT
               )
               ''')

    conexao.commit()
    conexao.close()

def insercao(nome:str,idade:int,salario:float):
    conexao = sqlite3.connect(caminho_completo)
    cursor = conexao.cursor()

    cursor.execute('''
               INSERT INTO Funcionario(nome,idade,SALARIO) VALUES(?,?,?)
               ''',(nome,idade,salario))

    conexao.commit()
    conexao.close()
    return f"Funcionário {nome} inserido com sucesso na base"


def selecao():
    conexao = sqlite3.connect(caminho_completo)
    cursor = conexao.cursor()

    cursor.execute("SELECT id, nome, idade, salario FROM Funcionario")
    dados = cursor.fetchall()

    conexao.close()

    if not dados:
        return "Nenhum funcionário encontrado."

    resposta = "Funcionários cadastrados:\n\n"
    for id, nome, idade, salario in dados:
        resposta += f"""
        ID: {id}
        Nome: {nome}
        Idade: {idade}
        Salário: R$ {salario}
        -------------------------
        """

    return resposta

def atualizar(nome:str,idade:int,salario:float,id:int):
    conexao = sqlite3.connect(caminho_completo)
    cursor = conexao.cursor()

    cursor.execute("UPDATE Funcionario SET nome = ?, idade = ?, salario = ? where id = ?",(nome,idade,salario,id))

    conexao.commit()
    conexao.close()
    return f"Funcionário {nome} atualizado com sucesso"

def delete(id:int):
    conexao = sqlite3.connect(caminho_completo)
    cursor = conexao.cursor()

    cursor.execute("DELETE FROM Funcionario WHERE ID = ?", (id))
    conexao.commit()
    return f"Funcinário deletado com sucesso"
    
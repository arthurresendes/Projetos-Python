import sqlite3


def criando():
    conexao = sqlite3.connect("compra.db")
    cursor = conexao.cursor()
    
    
    cursor.executescript('''
                CREATE TABLE pessoa(
                    id_pessoa INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome_pessoa VARCHAR(50),
                    CPF VARCHAR(20)
                );
                
                CREATE TABLE produto(
                    id_produto INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome_prod VARCHAR(50),
                    quantidade INTEGER,
                    preco_unitario FLOAT
                );
                
                CREATE TABLE pedido(
                    id_pedido INTEGER PRIMARY KEY AUTOINCREMENT,
                    id_pessoa INTEGER,
                    id_prod INTEGER,
                    FOREIGN KEY (id_pessoa) REFERENCES pessoa(id_pessoa),
                    FOREIGN KEY (id_prod) REFERENCES produto(id_produto)
                );
    ''')
    
    conexao.commit()
    conexao.close()

if __name__ == "__main__":
    criando()
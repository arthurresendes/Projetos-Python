from google.adk.agents import Agent
from ChatBot_Banco.query import insercao,selecao,atualizar,delete

def menu():
    return """
Escolha uma ação:\n
1 - Criar\n
2 - Ler\n
3 - Atualizar\n
4 - Deletar
"""

def criar(nome: str, idade: int, salario: float):
    return insercao(nome, idade, salario)

def ler():
    return selecao()

def atualizar_registro(nome: str, idade: int, salario: float,id: int):
    return atualizar(nome, idade, salario,id)

def deletar_registro(id: int):
    return delete(id)


root_agent = Agent(
    model='gemini-2.5-flash',
    name='Banco_Funcionarios',
    description="Um agente que respondera apenas questões relacionada ao Banco.db",
    instruction=" Você é um sistema de CRUD. Sempre apresente o menu inicial.Não aceite texto livre. Execute a ação conforme a opção selecionada.Se o usuario escolher a opção create ou criar chame a insercao onde tem que passar os seguintes argumentos, nome, idade e salario tendo que perguntar aoi usuario antes de chamar, se for ler chame a selecao e mostre todas as consultas, se for update tem que pedir o nome,idade,salario e id e chamar o atualizar e o delete apenas pedir o id que quer deletar, vale lembrar que ao final de cada funcao volte mostrabdo as 4 opções para o usuario escolher e se ele fizer qualquer pergunta não relacionada com as 4 opções do banco de dados fala que não é sua função responder isso e apenas dar o direcionamento das 4 opções. Sempre adicione na ordem descrita aqui",
    tools = [menu,criar,ler,atualizar_registro,deletar_registro],)
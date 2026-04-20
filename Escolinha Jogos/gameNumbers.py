import random

def menu():
    print("--------------------------------")
    print("Bem-vindo ao jogo do adivinha\nEscolha uma das opções")
    print("1- Adição")
    print("2- Subtração")
    print("3- Multiplicação")
    print("4- Divisão")
    print("--------------------------------")
    while True:
        try:
            opcao = int(input(": "))
            if 1 <= opcao <= 4:
                break
            else:
                print("Digite um número entre 1 e 4!")
        except ValueError:
            print("Digite um número inteiro válido!")
    if opcao == 1:
        adicao()
    elif opcao == 2:
        sub()
    elif opcao == 3:
        multi()
    elif opcao == 4:
        divisao()
    else:
        print("Opção invalida!!")

def nivel():
    print("--------------------------")
    print("Escolha o nível do jogo: ")
    print("1- Fácil")
    print("2- Médio")
    print("3- Dificil")
    print("4- Extremo")
    while True:
        try:
            opcaoNivel = int(input(": "))
            if 1 <= opcaoNivel <= 4:
                return opcaoNivel
            else:
                print("Digite um número entre 1 e 4!")
        except ValueError:
            print("Digite um número inteiro válido!")


def nivelamento(nivel: int):
    if nivel == 1:
        return random.randint(1,10), random.randint(1,10)
    elif nivel == 2:
        return random.randint(1,100), random.randint(1,100)
    elif nivel == 3:
        return random.randint(1,1000), random.randint(1,1000)
    elif nivel == 4:
        return random.randint(1,10000), random.randint(1,10000)
    else:
        return "Erro ao encontrar nivel"

def jogo(nivelEscolhido:int,operacao: str):
    contadorAcertos = 0
    contadorWhile = 0
    while contadorWhile < 3:
            n1, n2 = nivelamento(nivelEscolhido)
            res = operacoes(operacao,n1,n2)
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} {operacao} ? = {res}: ")
            resposta = float(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
    print("----Fim de jogo----")
    print(f"Você acertou {contadorAcertos} exercicios!!")
    print(f"As respostas certas eram {lista_resposta}")

def operacoes(operador: str,n1,n2):
    if operador == '+':
        return n1+n2
    elif operador == '-':
        return n1-n2
    elif operador == '*':
        return n1*n2
    elif operador == '/':
        return n1/n2
    else:
        return "Erro"

lista_resposta = []
nivelEscolhido = nivel()

#--------------------------------------
# Adição
def adicao():
    jogo(nivelEscolhido,'+')

# -------------------------------------
# Subtração
def sub():
    jogo(nivelEscolhido,'-')

# -------------------------------------
# Multiplicação
def multi():
    jogo(nivelEscolhido,'*')

# -------------------------------------
# Divisão
def divisao():
    jogo(nivelEscolhido,'/')

if __name__ == "__main__":
    menu()
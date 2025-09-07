import random

'''
Na função menu temos uma leve de opções dos 4 metodos de operação matematico de teste : +,-,*,/

Abrimos um while para caso o user digitar uma opção que não esta entre 1 e 4 e tambem para caso não digite um numero inteiro

Por fim chamamos as funções de cada modo de jogo
'''
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

'''
Aqui onde é decidido o nível de jogo entre fácil ate o extreme e novamente temos um while para caso o user digite o numero errado

'''
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
                return opcaoNivel # Aqui retorna a opção nivel para podermos utilizar como base nas funções
            else:
                print("Digite um número entre 1 e 4!")
        except ValueError:
            print("Digite um número inteiro válido!")

'''
Aqui o jogo funciona como , o user tem que adivinhar o numero faltante entre os numeros que da o devido resultado 
'''
lista_resposta = []
def adicao():
    nivelEscolhido = nivel() # Aqui onde puxamos o return opcao nivel para saber qual nivel foi escolhido
    contadorAcertos = 0
    contadorWhile = 0
    if nivelEscolhido == 1:
        while contadorWhile < 3:
            n1 = random.randint(1,10)
            n2 = random.randint(1,10)
            res = n1 + n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} + ? = {res}: ")
            resposta = int(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 2:
        while contadorWhile < 3:
            n1 = random.randint(1,100)
            n2 = random.randint(1,100)
            res = n1 + n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} + ? = {res}: ")
            resposta = int(input(": "))
            contadorWhile +=1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 3:
        while contadorWhile < 3:
            n1 = random.randint(1,1000)
            n2 = random.randint(1,1000)
            res = n1 + n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} + ? = {res}: ")
            resposta = int(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 4:
        while contadorWhile < 3:
            n1 = random.randint(1,10000)
            n2 = random.randint(1,10000)
            n3 = random.randint(1,10000)
            res = n1 + n2 + n3
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} + ? + {n3} = {res}: ")
            resposta = int(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")
    else:
        print("Opção invalida!!")

# -------------------------------------
# Subtração
def sub():
    nivelEscolhido = nivel()
    contadorAcertos = 0
    contadorWhile = 0
    if nivelEscolhido == 1:
        while contadorWhile < 3:
            n1 = random.randint(1,10)
            n2 = random.randint(1,10)
            res = n1 - n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} - ? = {res}: ")
            resposta = int(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 2:
        while contadorWhile < 3:
            n1 = random.randint(1,100)
            n2 = random.randint(1,100)
            res = n1 - n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} - ? = {res}: ")
            resposta = int(input(": "))
            contadorWhile +=1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 3:
        while contadorWhile < 3:
            n1 = random.randint(1,1000)
            n2 = random.randint(1,1000)
            res = n1 - n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} - ? = {res}: ")
            resposta = int(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 4:
        while contadorWhile < 3:
            n1 = random.randint(1,10000)
            n2 = random.randint(1,10000)
            n3 = random.randint(1,10000)
            res = n1 - n2 - n3
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} - ? - {n3} = {res}: ")
            resposta = int(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")
    else:
        print("Opção invalida!!")

# -------------------------------------
# Multiplicação
def multi():
    nivelEscolhido = nivel()
    contadorAcertos = 0
    contadorWhile = 0
    if nivelEscolhido == 1:
        while contadorWhile < 3:
            n1 = random.randint(1,10)
            n2 = random.randint(1,10)
            res = n1 * n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} x ? = {res}: ")
            resposta = int(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 2:
        while contadorWhile < 3:
            n1 = random.randint(1,100)
            n2 = random.randint(1,100)
            res = n1 * n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} x ? = {res}: ")
            resposta = int(input(": "))
            contadorWhile +=1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 3:
        while contadorWhile < 3:
            n1 = random.randint(1,1000)
            n2 = random.randint(1,1000)
            res = n1 * n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} x ? = {res}: ")
            resposta = int(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 4:
        while contadorWhile < 3:
            n1 = random.randint(1,1000)
            n2 = random.randint(1,1000)
            n3 = random.randint(1,1000)
            res = n1 * n2 * n3
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} x ? x {n3} = {res}: ")
            resposta = int(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")
    else:
        print("Opção invalida!!")

# -------------------------------------
# Divisão
def divisao():
    nivelEscolhido = nivel()
    contadorAcertos = 0
    contadorWhile = 0
    if nivelEscolhido == 1:
        while contadorWhile < 3:
            n1 = random.randint(1,10)
            n2 = random.randint(1,10)
            res = n1 / n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} / ? = {res}: ")
            resposta = float(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 2:
        while contadorWhile < 3:
            n1 = random.randint(1,100)
            n2 = random.randint(1,100)
            res = n1 / n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} / ? = {res}: ")
            resposta = float(input(": "))
            contadorWhile +=1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 3:
        while contadorWhile < 3:
            n1 = random.randint(1,1000)
            n2 = random.randint(1,1000)
            res = n1 / n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} / ? = {res}: ")
            resposta = float(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")

    elif nivelEscolhido == 4:
        while contadorWhile < 3:
            n1 = random.randint(1,10000)
            n2 = random.randint(1,10000)
            res = n1 / n2
            lista_resposta.append(n2)
            print(f"Qual o número faltante de {n1} / ? = {res}: ")
            resposta = float(input(": "))
            contadorWhile += 1
            if resposta == n2:
                contadorAcertos += 1
        print("----Fim de jogo----")
        print(f"Você acertou {contadorAcertos} exercicios!!")
        print(f"As respostas certas eram {lista_resposta}")
    else:
        print("Opção invalida!!")

# O menu onde eh tudo começado e puxa as funções roda no comando principal main
if __name__ == "__main__":
    menu()
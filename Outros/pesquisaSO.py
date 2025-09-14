from collections import Counter as cd

print("Pesquisa sobre SO!")
lista_resposta = []

opcoes = {
    1: "Windows",
    2: "Unix",
    3: "Linux",
    4: "Netware",
    5: "Mac OS",
    6: "Outros"
}

while True:
    print("\nEscolha seu sistema operacional preferido:")
    print("1-Windows")
    print("2-Unix")
    print("3-Linux")
    print("4-Netware")
    print("5-Mac OS")
    print("6-Outros")
    print("0-Sair")

    try:
        resposta = int(input())
        if resposta == 0:
            contagem = cd(lista_resposta)
            total = sum(contagem.values())

            print("\nSistema Operacional    Votos  Porcentagem")
            print("-------------------    -----  -----------")
            for i in range(1, 7):
                nome = opcoes[i]
                votos = contagem.get(nome, 0)
                porcentagem = (votos / total) * 100 if total > 0 else 0
                print(f"{nome:<22}{votos:^7}{porcentagem:>10.1f}%")
            break
        elif resposta in opcoes:
            lista_resposta.append(opcoes[resposta])
        else:
            print("Opção inválida.")
    except ValueError:
        print("Digite apenas números válidos.")
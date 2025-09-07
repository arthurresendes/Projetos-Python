def produtos():
    total = 0
    qtd = int(input("Quantos produtos você comprou: "))
    for prod in range(1, qtd + 1):
        valor = float(input(f"Digite o valor do Produto {prod} R$ "))
        total += valor
    return total


while True:
    total = produtos()
    dinheiro = 0
    while dinheiro < total:
        dinheiro = int(input("Quanto de dinheiro você irá entregar: "))
        if dinheiro < total:
            print("Dinheiro faltando!!")
    print("O seu troco é: R$ " , dinheiro - total)
    sair = input("Quer compra de novo (s/n)?").lower()
    if sair == 's':
        break
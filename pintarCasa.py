PRECO_LITRO = 80
# 3m² = 1 Litro , 1-Balde = 18L

metros = int(input("Quantos mestros quadrados será pintado ? "))

litros , resto = divmod(metros, 3)

if resto > 0 :
    litros += 1

qtd_baldes, sobra = divmod(litros, 18)

if sobra > 0:
    qtd_baldes += 1

total = PRECO_LITRO * qtd_baldes

print(f"Você precisa de {litros}L de tinta.")
print(f"Quantidade de baldes de 18L: {qtd_baldes}")
print(f"Custo total: R$ {qtd_baldes * PRECO_LITRO:.2f}")

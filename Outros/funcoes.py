lista = [1,2,3,4,1,6,1]
novaLitsa = []

def removendo_repetidos() -> None:
    for i in lista:
        if i not in novaLitsa:
            novaLitsa.append(i)
    print(novaLitsa)

lista2 = [0,1,2,3,0,0,1]
novaLista2 = []

def removendo_zero() -> None:
    contador = 0
    for i in lista2:
        if i == 0 and contador == 0 or i != 0:
            novaLista2.append(i)
            if i == 0:
                contador += 1
    novaLista2.sort()
    print(novaLista2)

def palindromo(palavra: str) -> str:
    return palavra[::-1]

def main():
    removendo_repetidos()
    removendo_zero()
    palavra = input("Digite uma palavra: ")
    print(palindromo(palavra))

if __name__ == "__main__":
    main()
from validate_docbr import CPF
from comando import insercao_pessoa,selecao_produtos,selecionar_por_indice,atualizar_por_indice,atualizar_pedido
import time

def decoaracao():
    print(f"=" *50)


def main():
    decoaracao()
    print("Bem-vindo a sua loja de compra favorita !!!")
    decoaracao()
    
    cpf = CPF()
    nome = input("Digite seu nome: ")
    cpf_user = input("Digite seu CPF: ")

    '''
    while not cpf.validate(cpf_user):
        print("---")
        print("Digite um CPF válido!")
        cpf_user = input("Digite seu CPF: ")
    '''
    
    insercao_pessoa(nome,cpf_user)

    decoaracao()
    print("Nome e CPF cadastrado")
    decoaracao()

    selecao_produtos()
    linha_produto = int(input("Qual produto irá querer de acordo com o indice(1-20): "))
    
    while linha_produto > 21 or linha_produto < 0:
        print("---")
        print("Digite um indice valido !!")
        linha_produto = input("Qual produto irá querer de acordo com o indice(1-20): ")
        
    quantidade = int(input("Digite quantos desse produto você quer: "))
    
    atualizar_por_indice(quantidade,linha_produto)
    time.sleep(2)
    decoaracao()
    selecionar_por_indice(linha_produto)
    atualizar_pedido(linha_produto)
    decoaracao()


if __name__ == "__main__":
    main()

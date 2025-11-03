from validate_docbr import CPF
from insercoes import insercao_pessoa

def decoaracao():
    print(f"=" *50)


def main():
    decoaracao()
    print("Bem-vindo a sua loja de compra favorita !!!")
    decoaracao()
    
    cpf = CPF()
    
    nome = input("Digite seu nome: ")
    cpf_user = input("Digite seu CPF: ")
    while not cpf.validate(cpf_user):
        print("---")
        print("Digite um CPF válido!")
        cpf_user = input("Digite seu CPF: ")
    insercao_pessoa(nome,cpf_user)

    decoaracao()
    print("Nome e CPF cadastrado")
    decoaracao()


if __name__ == "__main__":
    main()

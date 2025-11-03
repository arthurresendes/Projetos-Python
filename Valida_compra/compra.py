from validate_docbr import CPF
from insercoes import insercao_pessoa

def main():
    print(f"=" *50)
    print("Bem-vindo a sua loja de compra favorita !!!")
    print(f"=" *50)
    
    cpf = CPF()
    
    nome = input("Digite seu nome: ")
    cpf_user = input("Digite seu CPF: ")
    while not cpf.validate(cpf_user):
        print("---")
        print("Digite um CPF válido!")
        cpf_user = input("Digite seu CPF: ")
    insercao_pessoa(nome,cpf_user)
    print("Nome e CPF cadastrado")


if __name__ == "__main__":
    main()

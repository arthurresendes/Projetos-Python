'''
Classes: Imovel, Casa, Apartamento, Corretor, Cliente.

Mostre como usar:
Herança (Casa e Apartamento herdam de Imovel);
Encapsulamento (preço privado);
Polimorfismo (método descricao() diferente);
Composição (Corretor vende Imovel para Cliente).

'''

class Imovel():
    def __init__(self,endereco:str,area:float,preco:float,disponivel:bool):
        self.endereco = endereco
        self.area = area
        self.__preco = preco
        self.disponivel = disponivel
    
    def ver_preco(self):
        return self.__preco
    
    def atualizacao_preco(self,novo_valor:float):
        if novo_valor <= 0:
            return "Valo invalido"
        else:
            self.__preco = novo_valor

class Casa(Imovel):
    def __init__(self,endereco:str,area:float,preco:float,disponivel:True,num_quartos:int , num_vagas:int , quintal:bool):
        super().__init__(endereco,area,preco,disponivel)
        self.numQuarto = num_quartos
        self.numVaga = num_vagas
        self.quintal = quintal

class Apartamento(Imovel):
    def __init__(self,endereco:str,area:float,preco:float,disponivel:True,andar:int , valor_condominio:int ,  varanda:bool):
        super().__init__(endereco,area,preco,disponivel)
        self.andar = andar
        self.__condominio = valor_condominio
        self.varanda = varanda

class Cliente():
    def __init__(self, nome:str,cpf:str,orcamento:float):
        self.nome = nome
        self.cpf = cpf
        self.__orcamento = orcamento

class Corretor():
    id_contador = 0
    def __init__(self,nome:str,creci:str):
        self.id = Corretor.id_contador+1
        self.nome = nome
        self.creci = creci
        self.vendas = []
        Corretor.id_contador = self.id
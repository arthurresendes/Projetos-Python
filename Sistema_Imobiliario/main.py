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
    
    def descricao(self):
        if self.disponivel:
            return f"Casa no endereço {self.endereco} no valor de {self.__preco} com a área de {self.area} m² e disponivel"
        else:
            return f"Casa no endereço {self.endereco} no valor de {self.__preco} com a área de {self.area} m² não disponivel"

class Casa(Imovel):
    def __init__(self,endereco:str,area:float,preco:float,disponivel:True,num_quartos:int , num_vagas:int , quintal:bool):
        super().__init__(endereco,area,preco,disponivel)
        self.numQuarto = num_quartos
        self.numVaga = num_vagas
        self.quintal = quintal
    
    def descricao(self):
        return (f"Casa com {self.numQuarto} quartos, {self.numVaga} vagas, "
                f"{'com quintal' if self.quintal else 'sem quintal'}, "
                f"localizada em {self.endereco}. Preço: R${self.ver_preco():,.2f}")

class Apartamento(Imovel):
    def __init__(self,endereco:str,area:float,preco:float,disponivel:True,andar:int , valor_condominio:int ,  varanda:bool):
        super().__init__(endereco,area,preco,disponivel)
        self.andar = andar
        self.__valorCondominio = valor_condominio
        self.varanda = varanda
    
    def ver_valor_condominio(self):
        return self.__valorCondominio
    
    def descricao(self):
        return (f"Apartamento no {self.andar}º andar, "
                f"{'com varanda' if self.varanda else 'sem varanda'}, "
                f"condomínio R${self.__valorCondominio:,.2f}, "
                f"localizado em {self.endereco}. Preço: R${self.ver_preco():,.2f}")

class Cliente():
    def __init__(self, nome:str,cpf:str,orcamento:float):
        self.nome = nome
        self.cpf = cpf
        self.__orcamento = orcamento
        
    def ver_orcamento(self):
        return self.__orcamento
    
    def atualizar_orcamento(self,novo_orcamento:float):
        if novo_orcamento <=0 :
            return "Erro"
        else:
            self.__orcamento = novo_orcamento
            return "Orçamento atualizado"
class Corretor():
    id_contador = 0
    def __init__(self,nome:str,creci:str):
        self.id = Corretor.id_contador+1
        self.nome = nome
        self.creci = creci
        self.vendas = []
        Corretor.id_contador = self.id
    
    def vender_imovel(self,cliente:Cliente, imovel:Imovel):
        preco = imovel.ver_preco()
        if cliente.ver_orcamento() >= preco and imovel.disponivel:
            cliente.atualizar_orcamento(cliente.ver_orcamento() - preco)
            imovel.disponivel = False
            self.vendas.append({"cliente": cliente.nome, "imovel": imovel.endereco, "valor": preco})
            print(f"Venda realizada por {self.nome} para {cliente.nome} no valor de R${preco:,.2f}")
        else:
            return "Erro"

    def listar_vendas(self):
        print(f"\nVendas realizadas por {self.nome}:")
        for venda in self.vendas:
            print(f" - Cliente: {venda['cliente']}, Imóvel: {venda['imovel']}, Valor: R${venda['valor']:,.2f}")


casa1 = Casa("Rua das Flores, 123", 120, 350000, True, 3, 2, True)
apto1 = Apartamento("Av. Central, 45", 80, 280000, True, 5, 500, True)
cliente1 = Cliente("Arthur", "123.456.789-00", 400000)
corretor1 = Corretor("João Corretor", "CRECI-9999")

print(casa1.descricao())
print(apto1.descricao())

corretor1.vender_imovel(cliente1, casa1)

corretor1.listar_vendas()

class Pessoa():
    def __init__(self,nome,idade,CPF):
        self.nome = nome
        self.idade = idade
        self.cpf = CPF
        
    def descricao(self):
        return f"Pessoa {self.nome}, com, {self.idade} anos e seu cpf é {self.cpf}!!"

class Paciente(Pessoa):
    def __init__(self, nome, idade, CPF,sintomas,saldo_mensal):
        self.sintomas = sintomas
        self.saldo_mensal = saldo_mensal
        super().__init__(nome, idade, CPF)
    
    def descricao(self):
        return f"Pessoa {self.nome}, com, {self.idade} anos e seu cpf é {self.cpf}. O(a) {self.nome} tem {self.sintomas}!!"

class Psicologo(Pessoa):
    def __init__(self, nome, idade, CPF,formacao, valor_consulta_mensal):
        self.formacao = formacao
        self.valor_mensal = valor_consulta_mensal
        super().__init__(nome, idade, CPF)
    
    def descricao(self):
        return f"Pessoa {self.nome}, com, {self.idade} anos e seu cpf é {self.cpf}. O(a) {self.nome} tem formação em {self.formacao}!!"

class Consulta():
    def __init__(self,psicologa: Psicologo, paciente: Paciente):
        self.paciente = paciente
        self.psicologo = psicologa
    
    def marcar_consulta(self):
        if self.paciente.saldo_mensal >= self.psicologo.valor_mensal:
            print(f"Consulta marcada do paciente {self.paciente.nome} com o psicologo(a) {self.psicologo.nome}")
            self.paciente.saldo_mensal = self.paciente.saldo_mensal - self.psicologo.valor_mensal
            print(f"Saldo do paciente {self.paciente.saldo_mensal}")
        else:
            print("Consulta não disponivel")

if __name__ == "__main__":
    paciente1 = Paciente("Maria", 15,"111333222-11","Depressão" ,1000)
    psciologo1 = Psicologo("Maria Jascenilde", 50, "222444555-11","Psicoterapia", 900)
    consulta = Consulta(psciologo1,paciente1)
    print(paciente1.descricao())
    print(psciologo1.descricao())
    consulta.marcar_consulta()
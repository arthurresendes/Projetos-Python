import uuid

class Carta:
    def __init__(self, nome:str , custo:int, descricao: str, id_unico: uuid.UUID):
        self._nome = nome
        self._custo = custo
        self._descricao = descricao
        self._id_unico = id_unico
    

class CartaMagica(Carta):
    def __init__(self,tipo_magia:str ,nome, custo, descricao, id_unico):
        self._tipo_magia = tipo_magia
        super().__init__(nome, custo, descricao, id_unico)

class CartaCriatura(Carta):
    def __init__(self,ataque:int , defesa_base: int , defesa_ataque: int, esta_no_campo: bool ,nome, custo, descricao, id_unico):
        self._ataque = ataque
        self._defesa_base = defesa_base
        self._defesa_ataque = defesa_ataque
        self._esta_no_campo = esta_no_campo
        super().__init__(nome, custo, descricao, id_unico)

class Baralho:
    def __init__(self, cartas: list[Carta], nome_baralho: str):
        self._cartas = cartas
        self._nome_baralho = nome_baralho

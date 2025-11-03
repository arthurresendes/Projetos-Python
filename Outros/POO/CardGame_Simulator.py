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

class Mao:
    def __init__(self, cartas_na_mao : list[Carta], limite_cartas : int):
        self._cartas_na_mao = cartas_na_mao
        self._limite_cartas = limite_cartas

class Jogador:
    def __init__(self,nome:str,vida_atual:int,mana_max:int , mana_atual:int, baralho_principal: Baralho, mao_jogador: Mao):
        self._nome = nome
        self._vida_atual = vida_atual
        self._mana_max = mana_max
        self._mana_atual = mana_atual
        self._baralho = baralho_principal
        self._mao_jogador = mao_jogador

class MesaJogo:
    def __init__(self, jogador: list[Jogador] ,turno_atual:int, jogador_ativo:Jogador,campo_batalha: dict ):
        self._jogador = jogador
        self._turno_atual = turno_atual
        self._jogador_ativo = jogador_ativo
        self._campo_batalha = campo_batalha



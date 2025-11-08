import uuid
import random
from typing import List, Dict, Optional

class JogoException(Exception):
    pass

class ManaInsuficienteException(JogoException):
    pass

class MaoCheiaException(JogoException):
    pass

class Carta:
    def __init__(self, nome: str, custo: int, descricao: str, id_unico: uuid.UUID = None):
        self._nome = nome
        self._custo = custo
        self._descricao = descricao
        self._id_unico = id_unico if id_unico else uuid.uuid4()
    
    @property
    def nome(self):
        return self._nome
    
    @property
    def custo(self):
        return self._custo
    
    @property
    def descricao(self):
        return self._descricao
    
    @property
    def id_unico(self):
        return self._id_unico
    
    def __str__(self):
        return f"{self._nome} (Custo: {self._custo}) - {self._descricao}"
    
    def __repr__(self):
        return f"Carta({self._nome}, {self._custo})"

class CartaMagica(Carta):
    def __init__(self, tipo_magia: str, nome: str, custo: int, descricao: str, id_unico: uuid.UUID = None):
        self._tipo_magia = tipo_magia
        super().__init__(nome, custo, descricao, id_unico)
    
    @property
    def tipo_magia(self):
        return self._tipo_magia
    
    def ativar_efeito(self, alvo=None):
        print(f"Ativando magia: {self._nome} - {self._descricao}")
        # Implementar efeitos específicos baseados no tipo_magia
        return True
    
    def __str__(self):
        return f"[MAGIA] {super().__str__()} - Tipo: {self._tipo_magia}"

class CartaCriatura(Carta):
    def __init__(self, ataque: int, defesa: int, nome: str, custo: int, descricao: str, 
                 id_unico: uuid.UUID = None, pode_atacar: bool = False):
        self._ataque = ataque
        self._defesa = defesa
        self._defesa_atual = defesa
        self._pode_atacar = pode_atacar
        self._esta_no_campo = False
        super().__init__(nome, custo, descricao, id_unico)
    
    @property
    def ataque(self):
        return self._ataque
    
    @property
    def defesa(self):
        return self._defesa
    
    @property
    def defesa_atual(self):
        return self._defesa_atual
    
    @property
    def pode_atacar(self):
        return self._pode_atacar
    
    @property
    def esta_no_campo(self):
        return self._esta_no_campo
    
    def colocar_no_campo(self):
        self._esta_no_campo = True
        self._pode_atacar = False  # Não pode atacar no turno que entra
    
    def preparar_para_ataque(self):
        self._pode_atacar = True
    
    def atacar(self, criatura_alvo: Optional['CartaCriatura'] = None):
        if not self._pode_atacar:
            raise JogoException("Esta criatura não pode atacar neste turno")
        
        if criatura_alvo:
            # Ataque contra outra criatura
            dano = self._ataque
            criatura_alvo.receber_dano(dano)
            print(f"{self._nome} ataca {criatura_alvo.nome} causando {dano} de dano")
        else:
            # Ataque direto ao jogador
            print(f"{self._nome} ataca o jogador inimigo causando {self._ataque} de dano")
            return self._ataque
        
        self._pode_atacar = False
        return 0
    
    def receber_dano(self, dano: int):
        self._defesa_atual -= dano
        if self._defesa_atual <= 0:
            self._esta_no_campo = False
            print(f"{self._nome} foi derrotada!")
    
    def curar(self, quantidade: int):
        self._defesa_atual = min(self._defesa, self._defesa_atual + quantidade)
    
    def __str__(self):
        status = "Pode atacar" if self._pode_atacar else "Não pode atacar"
        return f"[CRIATURA] {self._nome} (Ataque: {self._ataque}, Defesa: {self._defesa_atual}/{self._defesa}) - {status}"

class Baralho:
    def __init__(self, cartas: List[Carta], nome_baralho: str):
        self._cartas = cartas
        self._nome_baralho = nome_baralho
    
    @property
    def cartas(self):
        return self._cartas.copy()
    
    @property
    def nome_baralho(self):
        return self._nome_baralho
    
    @property
    def quantidade_cartas(self):
        return len(self._cartas)
    
    def embaralhar(self):
        random.shuffle(self._cartas)
        print(f"Baralho '{self._nome_baralho}' embaralhado")
    
    def comprar_carta(self) -> Optional[Carta]:
        if self._cartas:
            return self._cartas.pop()
        return None
    
    def adicionar_carta(self, carta: Carta):
        self._cartas.append(carta)
    
    def remover_carta(self, carta: Carta):
        if carta in self._cartas:
            self._cartas.remove(carta)
            return True
        return False
    
    def __str__(self):
        return f"Baralho '{self._nome_baralho}' - {len(self._cartas)} cartas"

class Mao:
    def __init__(self, cartas_na_mao: List[Carta] = None, limite_cartas: int = 7):
        self._cartas_na_mao = cartas_na_mao if cartas_na_mao else []
        self._limite_cartas = limite_cartas
    
    @property
    def cartas_na_mao(self):
        return self._cartas_na_mao.copy()
    
    @property
    def limite_cartas(self):
        return self._limite_cartas
    
    @property
    def quantidade_cartas(self):
        return len(self._cartas_na_mao)
    
    def pode_adicionar_carta(self) -> bool:
        return len(self._cartas_na_mao) < self._limite_cartas
    
    def adicionar_carta(self, carta: Carta) -> bool:
        if self.pode_adicionar_carta():
            self._cartas_na_mao.append(carta)
            return True
        raise MaoCheiaException("Mão cheia! Não é possível adicionar mais cartas")
    
    def remover_carta(self, carta: Carta) -> bool:
        if carta in self._cartas_na_mao:
            self._cartas_na_mao.remove(carta)
            return True
        return False
    
    def procurar_carta_por_nome(self, nome: str) -> Optional[Carta]:
        for carta in self._cartas_na_mao:
            if carta.nome.lower() == nome.lower():
                return carta
        return None
    
    def __str__(self):
        cartas_str = "\n".join([f"{i+1}. {carta}" for i, carta in enumerate(self._cartas_na_mao)])
        return f"Mão ({len(self._cartas_na_mao)}/{self._limite_cartas}):\n{cartas_str}"

class Jogador:
    def __init__(self, nome: str, vida_atual: int = 20, mana_max: int = 10, 
                 baralho_principal: Baralho = None, mao_jogador: Mao = None):
        self._nome = nome
        self._vida_atual = vida_atual
        self._vida_max = vida_atual
        self._mana_max = mana_max
        self._mana_atual = 0
        self._baralho = baralho_principal if baralho_principal else Baralho([], f"Baralho de {nome}")
        self._mao_jogador = mao_jogador if mao_jogador else Mao()
        self._campo_criaturas: List[CartaCriatura] = []
        self._cemiterio: List[Carta] = []
    
    @property
    def nome(self):
        return self._nome
    
    @property
    def vida_atual(self):
        return self._vida_atual
    
    @property
    def mana_atual(self):
        return self._mana_atual
    
    @property
    def mana_max(self):
        return self._mana_max
    
    @property
    def campo_criaturas(self):
        return self._campo_criaturas.copy()
    
    @property
    def esta_vivo(self):
        return self._vida_atual > 0
    
    def comprar_carta(self) -> bool:
        carta = self._baralho.comprar_carta()
        if carta:
            try:
                self._mao_jogador.adicionar_carta(carta)
                print(f"{self._nome} comprou: {carta.nome}")
                return True
            except MaoCheiaException:
                print(f"{self._nome} tentou comprar carta mas a mão está cheia!")
                self._cemiterio.append(carta)  # Descarta a carta
                return False
        return False
    
    def comprar_cartas_iniciais(self, quantidade: int = 3):
        for _ in range(quantidade):
            self.comprar_carta()
    
    def pode_jogar_carta(self, carta: Carta) -> bool:
        return self._mana_atual >= carta.custo and carta in self._mao_jogador.cartas_na_mao
    
    def jogar_carta(self, carta: Carta) -> bool:
        if not self.pode_jogar_carta(carta):
            raise ManaInsuficienteException(f"Mana insuficiente para jogar {carta.nome}")
        
        self._mana_atual -= carta.custo
        self._mao_jogador.remover_carta(carta)
        
        if isinstance(carta, CartaCriatura):
            carta.colocar_no_campo()
            self._campo_criaturas.append(carta)
            print(f"{self._nome} colocou {carta.nome} no campo")
        elif isinstance(carta, CartaMagica):
            carta.ativar_efeito()
            self._cemiterio.append(carta)  # Magias vão para o cemitério após uso
            print(f"{self._nome} usou a magia {carta.nome}")
        
        return True
    
    def receber_dano(self, dano: int):
        self._vida_atual = max(0, self._vida_atual - dano)
        print(f"{self._nome} recebeu {dano} de dano! Vida: {self._vida_atual}")
    
    def curar(self, quantidade: int):
        self._vida_atual = min(self._vida_max, self._vida_atual + quantidade)
        print(f"{self._nome} curou {quantidade} de vida! Vida: {self._vida_atual}")
    
    def preparar_criaturas(self):
        for criatura in self._campo_criaturas:
            criatura.preparar_para_ataque()
    
    def resetar_mana(self):
        self._mana_atual = min(self._mana_max, self._mana_atual + 1)
        if self._mana_atual < self._mana_max:
            self._mana_atual = self._mana_max
    
    def __str__(self):
        criaturas_str = "\n".join([f"  - {criatura}" for criatura in self._campo_criaturas]) or "  Nenhuma"
        return f"""Jogador: {self._nome}
Vida: {self._vida_atual}/{self._vida_max}
Mana: {self._mana_atual}/{self._mana_max}
Criaturas no campo:
{criaturas_str}
"""

class MesaJogo:
    def __init__(self, jogadores: List[Jogador] = None):
        self._jogadores = jogadores if jogadores else []
        self._turno_atual = 0
        self._jogador_ativo = self._jogadores[0] if self._jogadores else None
        self._campo_batalha: Dict[Jogador, List[CartaCriatura]] = {}
        self._jogo_ativo = False
    
    @property
    def jogadores(self):
        return self._jogadores.copy()
    
    @property
    def turno_atual(self):
        return self._turno_atual
    
    @property
    def jogador_ativo(self):
        return self._jogador_ativo
    
    @property
    def jogo_ativo(self):
        return self._jogo_ativo
    
    def adicionar_jogador(self, jogador: Jogador):
        self._jogadores.append(jogador)
        self._campo_batalha[jogador] = []
    
    def iniciar_jogo(self):
        if len(self._jogadores) < 2:
            raise JogoException("São necessários pelo menos 2 jogadores para iniciar o jogo")
        
        self._jogo_ativo = True
        self._turno_atual = 1
        self._jogador_ativo = self._jogadores[0]
        
        # Embaralhar baralhos e comprar cartas iniciais
        for jogador in self._jogadores:
            jogador._baralho.embaralhar()
            jogador.comprar_cartas_iniciais(3)
            jogador._mana_atual = 1  # Mana inicial
        
        print("=== JOGO INICIADO ===")
        print(f"Turno 1 - Jogador ativo: {self._jogador_ativo.nome}")
    
    def proximo_turno(self):
        if not self._jogo_ativo:
            raise JogoException("O jogo não está ativo")
        
        self._turno_atual += 1
        
        # Alternar jogador ativo
        index_atual = self._jogadores.index(self._jogador_ativo)
        proximo_index = (index_atual + 1) % len(self._jogadores)
        self._jogador_ativo = self._jogadores[proximo_index]
        
        # Resetar estado do jogador ativo
        self._jogador_ativo.resetar_mana()
        self._jogador_ativo.preparar_criaturas()
        self._jogador_ativo.comprar_carta()
        
        print(f"\n=== Turno {self._turno_atual} - {self._jogador_ativo.nome} ===")
    
    def verificar_vitoria(self) -> Optional[Jogador]:
        jogadores_vivos = [j for j in self._jogadores if j.esta_vivo]
        
        if len(jogadores_vivos) == 1:
            vencedor = jogadores_vivos[0]
            self._jogo_ativo = False
            return vencedor
        elif len(jogadores_vivos) == 0:
            self._jogo_ativo = False
            return None  # Empate
        
        return None
    
    def atacar_jogador(self, atacante: Jogador, defensor: Jogador, index_criatura: int):
        if atacante != self._jogador_ativo:
            raise JogoException("Não é seu turno!")
        
        if index_criatura < 0 or index_criatura >= len(atacante.campo_criaturas):
            raise JogoException("Criatura inválida!")
        
        criatura = atacante.campo_criaturas[index_criatura]
        dano = criatura.atacar()  # Ataque direto ao jogador
        defensor.receber_dano(dano)
    
    def __str__(self):
        status = "Ativo" if self._jogo_ativo else "Inativo"
        return f"Mesa de Jogo - Turno {self._turno_atual} - Status: {status}"

class CartaFactory:
    @staticmethod
    def criar_criatura(ataque: int, defesa: int, nome: str, custo: int, descricao: str = "") -> CartaCriatura:
        return CartaCriatura(ataque, defesa, nome, custo, descricao)
    
    @staticmethod
    def criar_magia(tipo_magia: str, nome: str, custo: int, descricao: str = "") -> CartaMagica:
        return CartaMagica(tipo_magia, nome, custo, descricao)

# Exemplo de uso
def criar_baralho_exemplo() -> Baralho:
    cartas = [
        CartaFactory.criar_criatura(2, 2, "Goblin", 1, "Uma criatura pequena e rápida"),
        CartaFactory.criar_criatura(3, 3, "Cavaleiro", 3, "Um guerreiro nobre"),
        CartaFactory.criar_criatura(5, 4, "Dragão", 6, "Uma criatura poderosa"),
        CartaFactory.criar_magia("Dano", "Bola de Fogo", 3, "Causa 3 de dano a uma criatura"),
        CartaFactory.criar_magia("Cura", "Poção de Cura", 2, "Cura 4 de vida"),
    ]
    return Baralho(cartas, "Baralho Inicial")

def main():
    # Criar jogadores e baralhos
    baralho1 = criar_baralho_exemplo()
    baralho2 = criar_baralho_exemplo()
    
    jogador1 = Jogador("Alice", baralho_principal=baralho1)
    jogador2 = Jogador("Bob", baralho_principal=baralho2)
    
    # Criar mesa e iniciar jogo
    mesa = MesaJogo()
    mesa.adicionar_jogador(jogador1)
    mesa.adicionar_jogador(jogador2)
    
    mesa.iniciar_jogo()
    
    # Simular alguns turnos
    try:
        for _ in range(3):
            mesa.proximo_turno()
            
            # Jogador atual compra e joga uma carta se possível
            jogador_atual = mesa.jogador_ativo
            if jogador_atual.mao_jogador.quantidade_cartas > 0:
                carta = jogador_atual.mao_jogador.cartas_na_mao[0]
                if jogador_atual.pode_jogar_carta(carta):
                    jogador_atual.jogar_carta(carta)
            
            # Verificar se há vencedor
            vencedor = mesa.verificar_vitoria()
            if vencedor:
                print(f"\n{vencedor.nome} venceu o jogo! 🎉")
                break
    
    except Exception as e:
        print(f"Erro durante o jogo: {e}")

if __name__ == "__main__":
    main()
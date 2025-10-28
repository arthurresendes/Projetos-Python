'''

Documentação por secção:


1 -- Confis Iniciais:
    Utilização de um numero magico que garante a reprodutilidade (RANDOM_SEED)

    Com np.random.seed conseguimos controlar aleatoriamente os numeros gerados

    torch.manual_seed(RANDOM_SEED) Controla a aleatoriedade , fazendo os modelos começar com o mesmo peso

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") Para saber se usa CPU ou GPU

    sns.set(...) -> Configurações de estilizações nos dashboards

2 -- Coletando dados

    Aqui passamos uma lista de aplicativos onde sera utilizado de base para pegar os scores os content, tudo isso vindo da lib da play store. Logo em seguida percorremos cada aplicativo armazenando as devidas informações em outra lista excluindo os comments que não serão necessarios e só daram atraso por conta de ser pesado. Usamos try execpt para tratar erro caso algum app de erro na hora de leitura. Por fim nessa secção criamos um dataframe onde ira pegar as informações que foi adicionada a nova lista que percorreu o loop for e mostramos com head do pandas os 3 primeiros apps e informações como o title(nome do app) , score(avaliação) e installs

    Aqui é o passo inicial 

3 -- Coletando reviews

    Já nessa parte é aonde começa a ficar um pouco mais complexo , criamos uma lista de aplicativos_reviews onde sera adicionado algumas informações , temos que criar um loop que passamo como parametro a lista de url dos jogos da play store , dentro dele outro loop de pontuação de 1 a 5 e dentro desse loop de pontuação criamos outro loop de ordenação que pode ser os Mais relevante ou mais recente. Dentro do ultimo loop encadeado fazemos um try execpt onde dentro do try damos um time.sleep(1) para dar uma carregada entre os dados e fazemos uma contagem de 200 para caso a pontuação seja == 3 e senao a pontuação e 100 , isso pq como uma avaliação 3 é mais rar a pontuação foi favorecida nesse quesito para o modelo aprender. Logo mais abaixo criamos o rvs, _ = reviews(), o rvs significa uma lista de reviews e o _ e para ignorar o token de paginação e dentro da reviews que vem da lib do play store passamos alguns parametros como o aplicativo , linguagem , pais, sort , contagem e filtragem de score. Logo em seguida usamos um loop para percorrer esse rvs onde marcamos uma nova coluna sortOrder passando se foi mais relevante ou recente e o id do app , fora do loop aplicamos esse rvs para a lista que foi criada fora de todos esses loops. Fazemos a tratativa de erro e para acabar fazemos um df principal onde recebe a lista criada no começo da secção
    

4 -- Analise de sentimento 

    Criação de dash que pega de base o df principal e faz um dashboard de acordo com os score

5 -- Mapeando sentimentos

    Aqui criamos uma função onde caso a avalição for menor ou igual a 2 retornamos 0 , == 3 retornamos 1 e se nao se retornamos 2. Isso serve para fazer a marcação da rede neural de acordo com negativo , neutro e positivo.Criamos uma coluna no df chamada sentiments onde vai pegar o score e ira aplicar essa função. Logo depois criamos uma lista com os 3 tipos de avaliação e por fim fazemos um grafico de forma estilizada que aparecam os números exato de cada cois

6 -- Pré processamento de dados

    Criamos uma função onde ira limpar os dados , evitar dados NaN e retorna o texto sem nenhuma url , menção , pontuação e espaços. Criamos uma nova coluna no df chamada de content_clean que vai pehgar de base o df['content'] e ira aplicar a função limpagem , logo em seguida atualizamos o df para tirar as reviews que tem menos que 10 de tamanho e mostramos quantas reviews limpas temos no nosso dataframe


7 -- BERT

    Importamos as libs que iremos utilizar para testar o modelo, criamos um model_name passando parametros da lingua portugues e logo em seguida criamos uma variavel tokenizer que vai autotokenizer com o modelo em portugues , passamos o ber_model onde vai receber o model_name em portugues e num_labels de 3 que no caso é negativo , neutro e positivo


8 -- DataSet BERT

    Criamos uma classe de dataset review passando como herança um dataset , no metodo construtor usamos os textos , sentimentos , tokenizer e o comprimento maximo, na função __len__ retornamos o tamanho do textos e no __getitem__ usamos para localizar via inde um texto e sentimento especifico. No enconding sera passado as configurações do tokenizer que sera o text expecifico vindo do iloc do indice , padding com tamanho fixo, ou seja todas palavras com mesmo comprimento ,trucation true mantem so os primeiros tokens, max_length vai receber o comprimento maximo do metodo construtor e o tensores em vez de lista no caso tensor([[101, 1234, 5678, ..., 102]]), como se comunicam via numero tem que converter


9 -- Divisão dos dados bert

    Criamos um função de dataset balanceado passando como parametro um df e max_por_classe com valor fixo de 400. Criamos uma lista de datasets e percorremos uma lista de [0,1,2] -- Negativo |neutro |positivo. Criamos uma variavel que armazena cada sentiment da classe especifica e se o tamanho for maior fazemos um alteração com ransom_seed,depois disso fazemos um append da variavel na lista datasets. Retornamos uma concatenação entre os datasets em um uncico so , ou seja ante era separados o negativo em um local , neutro em outro e positivo em um e com a concatenação foi possivel trazer tudo junto.
    
    Instanciamos essa função passando o dataframe e o max_por_classe padrão de 400 , mostramos quantos negativos , neutros e positivos temos balanceado e ai começamos as divisão dos dados. Separamos o treino em 70% e o temp em 30% e do 30% 50 serao de validação e 50 de teste.
    
    BATCH_SIZE -> Arupa as reviews , ali esta em 32 se tivermos 800 reviews fazemos 800/32
    
    Criamos dataset pegando da classe criado na secção 8 e passamos os content limp , sentimento, tokenizer e o comprimento maximo
    
    Criamos dataloader de cada passando os datset para ...
    

10 - Treinar modelo BERT

    Criamos uma função treinar que vamos receber de parametro model(modelo BERT que vamos treinar), tran_loader dados de treinto , dados de validação , numero de vezes que vera todos os dados (3) elearning_rate=2e-5: Taxa de aprendizado (0.00002)
    Configuramos para ele detectar se tem gpu se tiver utiliza esse para o testo senao usa a cpu mesmo , o model.to(device) Move o modelo para gpu ou pra cpu dependendo do caso
    
    Configurações de otimizador
    optimizer = AdamW(
    model.parameters(), 
    lr=learning_rate,
    weight_decay=0.01,  # Regularização
    eps=1e-8  # Estabilidade numérica
)

    total_setps seria o total de batches * epochs que seria as epocas
    
    best_accuracy guarda a melhor aucuracia, best_model_state guarda pesos do melhor modelos e history a historia do treino
    
    Loop de epoca mostrando qual epoca esta sendo executada
    
    model.train() -> Ativa o modo de treino
    total_train_loss -> Acumula loss total
    trains_steps -> Conta os batches processados
    
    train_progress = tqdm(train_loader...)
    for batch in train_progress Para cada batch em treino progressos
    
    Pega os dados do batch
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

'''
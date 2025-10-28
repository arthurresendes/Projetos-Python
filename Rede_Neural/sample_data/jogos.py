import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from google_play_scraper import Sort, reviews, app
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict, Counter
import re


# 1 --- Configs Iniciais

RANDOM_SEED = 42 # Numero magico na CD, garante reprodutilidade
np.random.seed(RANDOM_SEED) # Controla aleatoriamente os números gerados
torch.manual_seed(RANDOM_SEED) # Controla a aleatoriedade , modelo começa com os mesmo peso


sns.set(style='whitegrid', palette='muted', font_scale=1.2) # Estilo de graficos padrão


# 2 --- Coletando dados

jogos_ids = [
    "com.supercell.clashroyale",
    "com.dts.freefireth",
    "com.supercell.brawlstars",
    "com.tencent.ig",
    "br.pipacombate.maiworm",
    "com.duolingo",
    "com.kiloo.subwaysurf",
    "air.com.hypah.io.slither"
]

print("\nColetando informações dos aplicativos")


aplic_infos = []
for aplic in tqdm(jogos_ids, desc="Apps"):
    try:
        info = app(aplic, lang="pt", country="br")
        info.pop('comments', None)
        aplic_infos.append(info)
    except Exception as e:
        print(f"Erro ao coletar {aplic}: {e}")
        continue

dataframe_jogos = pd.DataFrame(aplic_infos)
print(f"{len(dataframe_jogos)} aplicativos coletados")
print(dataframe_jogos[['title', 'score', 'installs']].head(3))


# 3 - Coletando reviews
'''
Dados brutos: Textos que usuários escreveram
Metadados: Notas (1-5 estrelas), data, relevância
Diversidade: Reviews recentes + relevantes + todas as nota
'''

aplic_reviews = []
print("\nColetando reviews dos usuários")

for aplic_id in tqdm(jogos_ids, desc="Reviews"):
    # Passar por notas de 1 a 5
    for pontuacao in range(1,6):
        # Mais relevantes e mais recentes para gerar mais diversidade
        for ordem_ordenada in [Sort.MOST_RELEVANT, Sort.NEWEST]:
            try:
                import time
                time.sleep(1)
                count = 200 if pontuacao == 3 else 100 # Reviews neutros são mais raros, mas importantes para o modelo aprender, ou seja sempre 3 será mais valioso na hora de treinar
                
                # rvs -> Lista dos reviews e _ ignora o token de continuação (paginação) e reviews vem da playstore passando os devidos parametros (apli-aplicativo, linguagem , pais , sort(ordenação), contagem de pontuação , e a filtragem)
                rvs, _ = reviews(
                    aplic_id,
                    lang='pt',
                    country='br',
                    sort=ordem_ordenada,
                    count=count,
                    filter_score_with=pontuacao
                )
                
                for r in rvs:
                    r['sortOrder'] = 'mais_relevante' if ordem_ordenada == Sort.MOST_RELEVANT else 'mais_recente'
                    r['appId'] = aplic_id
                aplic_reviews.extend(rvs)
            except Exception as e:
                print(f"Erro {aplic_id} com avaliação {pontuacao}: {e}")
                continue

df = pd.DataFrame(aplic_reviews)
print(f"Total de reviews coletados: {len(df)}")


# 4 - Analise de sentimento (Dash)

plt.figure(figsize=(10,5)) # # Cria figura de 10x5 polegadas
sns.countplot(x='score', data=df, palette='viridis') #Pega a coluna score do df viridis esquema verde e azulado
plt.xlabel('Notas do Reviews') # Label x
plt.ylabel('Quantidade') # Label y
plt.title('Distribuição das Notas dos Usuários')
plt.tight_layout() # Ajusta espaçamento automático
plt.savefig('distribuicao_notas.png', dpi=300, bbox_inches='tight')
plt.show()


# 5 - Mapear os sentimentos

def mapeando_sentimentos(avaliacao):
    avaliacao = int(avaliacao)
    if avaliacao <=2:
        return 0
    elif avaliacao == 3:
        return 1
    else:
        return 2

df['sentiment'] = df['score'].apply(mapeando_sentimentos)
nomes_classes = ['Negativo', 'Neutro', 'Positivo']

plt.figure(figsize=(8,5))
ax = sns.countplot(x='sentiment', data=df, palette='Set2')
ax.set_xticks(range(len(nomes_classes)))
ax.set_xticklabels(nomes_classes)
plt.xlabel("Sentimento")
plt.ylabel("Quantidade")
plt.title("Distribuição dos Sentimentos")

# ax.patches são as 3 barras do graficos enumeradas, faz os calculos para aparecer o numero na barra
for i , p in enumerate(ax.patches):
    altura = p.get_height()
    ax.text(p.get_x() +  p.get_width()/2., altura + 50,f'{int(altura)}', ha="center", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('distribuicao_sentimentos.png', dpi=300, bbox_inches='tight')
plt.show()

df.to_csv('reviews.csv', index=False)
print("Dados salvos em 'reviews.csv'")

# 6 - PRÉ processamento de texto

'''
Limpagem de texto preparando para o modelo
Adicionando nova coluna no df onde tem essa limpeza que é aplicado no content(comentarios) e reviews com menos de 10 caracteres são removidas por serem curtas
'''

def limpagem(texto):
    # Protege contra valores nulos/NaN
    if pd.isna(texto):
        return ""
    texto = str(texto).lower() # Deixando minusculo
    texto = re.sub(r'http\S+|www\S+', '', texto)  # Remove URLs
    texto = re.sub(r'@\w+', '', texto)  # Remove menções
    texto = re.sub(r'[^\w\s]', ' ', texto)  # Remove pontuação
    texto = re.sub(r'\s+', ' ', texto).strip()  # Remove espaços extras
    return texto

df['content_clean'] = df['content'].apply(limpagem)
df = df[df['content_clean'].str.len() > 10] # Remove comentarios/reviews curtos
print(f"Reviews após limpeza: {len(df)}")


# 7 --- BERT


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

model_name = "neuralmind/bert-base-portuguese-cased" # Bert em portugues
tokenizer = AutoTokenizer.from_pretrained(model_name) # tokenização para portugues
bert_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3  # 3 classes: Negativo, Neutro, Positivo
)


# 8 --- DATASET BERT

class BERTReviewDataset(Dataset):
    def __init__(self, textos, sentimentos, tokenizer, max_length=128):
        self.textos = textos
        self.sentimentos = sentimentos
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, index):
        text = str(self.textos.iloc[index])
        sentiment = self.sentimentos.iloc[index]
        
        # Tokenização BERT 
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }

# 9 --- DIVISÃO DOS DADOS BERT 

print("\n" + "="*50)
print("PREPARANDO DADOS - VERSAO OTIMIZADA")
print("="*50)


def criar_dataset_balanceado(df, max_por_classe=400):
    datasets = []
    for classe in [0, 1, 2]:  # Negativo, Neutro, Positivo
        classe_data = df[df['sentiment'] == classe]
        if len(classe_data) > max_por_classe:
            classe_data = classe_data.sample(max_por_classe, random_state=RANDOM_SEED)
        datasets.append(classe_data)
    
    return pd.concat(datasets, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)

# Cria dataset balanceado (melhor para aprendizado)
df_balanceado = criar_dataset_balanceado(df, max_por_classe=400)
print(f"Dataset balanceado criado: {len(df_balanceado)} reviews")
print(f"Distribuição: Negativo={sum(df_balanceado['sentiment']==0)}, "
      f"Neutro={sum(df_balanceado['sentiment']==1)}, "
      f"Positivo={sum(df_balanceado['sentiment']==2)}")

# Divisão dos dados
df_treino, df_temp = train_test_split(
    df_balanceado, 
    test_size=0.3, 
    random_state=RANDOM_SEED, 
    stratify=df_balanceado['sentiment']
)
df_valid, df_test = train_test_split(
    df_temp, 
    test_size=0.5, 
    random_state=RANDOM_SEED, 
    stratify=df_temp['sentiment']
)

print(f"\nDivisão dos dados:")
print(f"  Treino: {len(df_treino)} ({len(df_treino)/len(df_balanceado)*100:.1f}%)")
print(f"  Validação: {len(df_valid)} ({len(df_valid)/len(df_balanceado)*100:.1f}%)")
print(f"  Teste: {len(df_test)} ({len(df_test)/len(df_balanceado)*100:.1f}%)")

# Configurações otimizadas
BATCH_SIZE = 8  # Balance entre velocidade e estabilidade
MAX_LENGTH = 128  # Textos mais curtos para mais velocidade

# Cria datasets BERT
treino_dataset = BERTReviewDataset(df_treino['content_clean'], df_treino['sentiment'], tokenizer, MAX_LENGTH)
valid_dataset = BERTReviewDataset(df_valid['content_clean'], df_valid['sentiment'], tokenizer, MAX_LENGTH)  
test_dataset = BERTReviewDataset(df_test['content_clean'], df_test['sentiment'], tokenizer, MAX_LENGTH)

# DataLoaders BERT
treino_loader = DataLoader(treino_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Batches de treino: {len(treino_loader)}")
print(f"Batches de validação: {len(valid_loader)}")
print(f"Batches de teste: {len(test_loader)}")

# 10 --- TREINAMENTO BERT

def treinar_bert_eficiente(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo de treino: {device}")
    model = model.to(device)
    
    # Otimizador com configurações robustas
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,  # Regularização
        eps=1e-8  # Estabilidade numérica
    )
    
    # Scheduler adaptativo
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% de warmup
        num_training_steps=total_steps
    )
    
    # Para acompanhamento
    best_accuracy = 0
    best_model_state = None
    history = {'train_loss': [], 'val_accuracy': []}
    
    print("\nINICIANDO TREINAMENTO...")
    
    for epoch in range(epochs):
        print(f"\n{'='*40}")
        print(f"ÉPOCA {epoch+1}/{epochs}")
        print(f"{'='*40}")
        
        # ========== FASE DE TREINO ==========
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        # Barra de progresso para treino
        train_progress = tqdm(train_loader, desc=f"Treino Época {epoch+1}", leave=False)
        
        for batch in train_progress:
            try:
                # Move dados para o dispositivo
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Zera gradientes
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass com tratamento seguro
                if loss.requires_grad:
                    loss.backward()
                    # Clip de gradientes para evitar explosão
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                
                train_steps += 1
                
                # Atualiza barra de progresso
                train_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Erro de memória, pulando batch...")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue
                else:
                    print(f"Erro no batch: {e}")
                    continue
            except Exception as e:
                print(f"Erro inesperado: {e}")
                continue
        
        # ========== FASE DE VALIDAÇÃO ==========
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Validação Época {epoch+1}", leave=False)
            
            for batch in val_progress:
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    preds = torch.argmax(outputs.logits, dim=1)
                    val_predictions.extend(preds.cpu().numpy())
                    val_true_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"Erro na validação: {e}")
                    continue
        
        # ========== CÁLCULO DE MÉTRICAS ==========
        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0
        
        if len(val_predictions) > 0:
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            
            # Relatório de classificação da validação
            if epoch == epochs - 1:  # Mostra relatório apenas na última época
                print(f"\nRELATÓRIO DE VALIDAÇÃO (Época {epoch+1}):")
                print(classification_report(val_true_labels, val_predictions, 
                                        target_names=nomes_classes, digits=4))
            
            print(f"\nRESUMO ÉPOCA {epoch+1}:")
            print(f"  Loss Treino: {avg_train_loss:.4f}")
            print(f"  Acurácia Validação: {val_accuracy:.4f}")
            
            # Salva histórico
            history['train_loss'].append(avg_train_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Salva melhor modelo
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                print(f" NOVO MELHOR MODELO! Acurácia: {val_accuracy:.4f}")
        else:
            print("Validação falhou - sem predições")
    
    # ========== FINALIZAÇÃO ==========
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n{'='*50}")
        print(f"TREINAMENTO CONCLUÍDO!")
        print(f"Melhor acurácia de validação: {best_accuracy:.4f}")
        print(f"{'='*50}")
    else:
        print(" Nenhum modelo válido foi salvo")
    
    return model, history

# Executa treinamento
print("\nINICIANDO TREINAMENTO BERT...")
modelo_treinado, historico = treinar_bert_eficiente(
    bert_model, 
    treino_loader, 
    valid_loader, 
    epochs=3,           # 3 épocas para melhor aprendizado
    learning_rate=2e-5  # Taxa de aprendizado padrão para BERT
)

# 11 --- AVALIAÇÃO COMPLETA NO TESTE

def avaliar_modelo_completo(model, test_loader, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    print(f"\n{'='*50}")
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print(f"{'='*50}")
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Processando teste")
        
        for batch in test_progress:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
            except Exception as e:
                print(f"Erro no batch de teste: {e}")
                continue
    
    if len(all_predictions) == 0:
        print(" Nenhuma predição foi feita no teste")
        return 0
    
    # Métricas detalhadas
    accuracy = accuracy_score(all_true_labels, all_predictions)
    
    print(f"\nRESULTADOS FINAIS:")
    print(f"Acurácia: {accuracy:.4f}")
    
    print(f"\nRELATÓRIO DE CLASSIFICAÇÃO DETALHADO:")
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=nomes_classes, digits=4))
    
    # Matriz de confusão
    try:
        cm = confusion_matrix(all_true_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=nomes_classes, yticklabels=nomes_classes,
                   cbar_kws={'label': 'Quantidade'})
        plt.xlabel('Predito')
        plt.ylabel('Real') 
        plt.title('Matriz de Confusão - BERT (Conjunto de Teste)')
        plt.tight_layout()
        plt.savefig('matriz_confusao_bert_final.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(" Matriz de confusão salva como 'matriz_confusao_bert_final.png'")
    except Exception as e:
        print(f" Erro ao gerar matriz de confusão: {e}")
    
    return accuracy

# Executa avaliação completa
acuracia_final = avaliar_modelo_completo(modelo_treinado, test_loader, tokenizer)

# 12 --- SISTEMA DE PREDIÇÃO ROBUSTO

def prever_sentimento_avancado(texto, model, tokenizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    try:
        # Tokenização com tratamento de erro
        encoding = tokenizer(
            texto,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities, dim=1).values.item()
        
        resultado = {
            'sentimento': nomes_classes[predicted_class],
            'confianca': confidence,
            'probabilidades': {
                'Negativo': probabilities[0][0].item(),
                'Neutro': probabilities[0][1].item(),
                'Positivo': probabilities[0][2].item()
            }
        }
        
        return resultado
        
    except Exception as e:
        print(f" Erro na predição: {e}")
        return None

# Teste com exemplos variados
print(f"\n{'='*50}")
print("TESTE DE PREDIÇÕES COM O MODELO TREINADO")
print(f"{'='*50}")

exemplos_teste = [
    "Péssimo jogo, não funciona nada! Travamentos constantes.",
    "É mais ou menos, poderia ser melhor mas até que dá pro gasto",
    "Jogo incrível! Gráficos lindos e jogabilidade viciante!",
    "Não gostei, muito repetitivo e cheio de anúncios",
    "Bom, mas tem alguns bugs que atrapalham a experiência",
    "Perfeito! Melhor jogo que já joguei no celular!",
    "Mais ou menos, nem bom nem ruim",
    "Horrível, não recomendo para ninguém",
    "Excelente! Muito divertido e bem otimizado"
]

print("\nRESULTADOS DAS PREDIÇÕES:")
print("-" * 80)

for i, texto in enumerate(exemplos_teste, 1):
    resultado = prever_sentimento_avancado(texto, modelo_treinado, tokenizer)
    
    if resultado:
        print(f"\nEXEMPLO {i}:")
        print(f"Texto: {texto}")
        print(f"Sentimento: {resultado['sentimento']} (Confiança: {resultado['confianca']:.1%})")
        print(f"Detalhes: Neg={resultado['probabilidades']['Negativo']:.1%} | "
              f"Neu={resultado['probabilidades']['Neutro']:.1%} | "
              f"Pos={resultado['probabilidades']['Positivo']:.1%}")
    else:
        print(f"\nEXEMPLO {i}: Falha na predição")

# 13 --- SALVAMENTO E RELATÓRIO FINAL

print(f"\n{'='*50}")
print("FINALIZANDO E SALVANDO RECURSOS")
print(f"{'='*50}")

# Salva o modelo treinado
try:
    modelo_treinado.save_pretrained('modelo_bert_final')
    tokenizer.save_pretrained('modelo_bert_final')
    print(" Modelo salvo em 'modelo_bert_final/'")
    
    # Salva métricas finais
    with open('metricas_treinamento.txt', 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DO TREINAMENTO BERT\n")
        f.write("=" * 40 + "\n")
        f.write(f"Acurácia final: {acuracia_final:.4f}\n")
        f.write(f"Total de reviews: {len(df_balanceado)}\n")
        f.write(f"Reviews de treino: {len(df_treino)}\n")
        f.write(f"Reviews de validação: {len(df_valid)}\n")
        f.write(f"Reviews de teste: {len(df_test)}\n")
        f.write(f"Data de treinamento: {pd.Timestamp.now()}\n")
    
    print("Métricas salvas em 'metricas_treinamento.txt'")
    
except Exception as e:
    print(f" Erro ao salvar recursos: {e}")

print(f"\n{'='*50}")
print("PROGRAMA CONCLUÍDO COM SUCESSO!")
print(f"{'='*50}")
print(f"Acurácia final alcançada: {acuracia_final:.1%}")
print(f"Modelo salvo para uso futuro")
print(f"Pronto para analisar sentimentos de novos reviews!")
print(f"{'='*50}")
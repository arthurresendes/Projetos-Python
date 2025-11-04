import json
import copy
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

# 1 -- Conifgs Iniciais

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
sns.set(style='whitegrid', palette='muted', font_scale=1.2) 


# 2 - Coletando dados

aplicativos_play = [
    "br.com.drogaraia",
    "br.com.rdsaude.healthPlatform.android",
    "br.com.raiadrogasil.apps.painelrd",
    "br.com.drogasil"
]

print("\nColetando informações dos aplicativos")

informacoes_apps = []
for aplic in tqdm(aplicativos_play, desc="Apps"):
    try:
        info = app(aplic, lang="pt", country="br")
        info.pop('comments', None)
        informacoes_apps.append(info)
    except Exception as e:
        print(f"Erro ao coletar {aplic}: {e}")
        continue

dataframe_app = pd.DataFrame(informacoes_apps)
print(f"{len(dataframe_app)} aplicativos coletados")
print(dataframe_app[['title', 'score', 'installs']].head(4))

# 3 - Reviews

reviews_app = []

for app_id in tqdm(aplicativos_play, desc="Reviews"):
    for pontuacao in range(1,6):
        for ordenacao in [Sort.MOST_RELEVANT, Sort.NEWEST]:
            try:
                import time
                time.sleep(1)
                count = 200 if pontuacao == 3 else 100
                
                rvs,_ = reviews(
                    app_id,
                    lang='pt',
                    country='br',
                    sort=ordenacao,
                    count=count,
                    filter_score_with=pontuacao
                )
                
                for r in rvs:
                    r['sortOrder'] = 'mais_relevante' if ordenacao == Sort.MOST_RELEVANT else 'mais_recente'
                    r['appId'] = app_id
                reviews_app.extend(rvs)
            except Exception as e:
                print(f"Erro {app_id} com avaliação {pontuacao}: {e}")
                continue

df = pd.DataFrame(reviews_app)
print(f"Total de reviews coletados: {len(df)}")

# 4 - Configurando grafico

plt.figure(figsize=(10,5))
sns.countplot(x='score', data=df, palette='viridis')
plt.xlabel('Notas do Reviews')
plt.ylabel('Quantidade')
plt.title('Distribuição das Notas dos Usuários')
plt.tight_layout()
plt.savefig('distribuicao_notas.png', dpi=300, bbox_inches='tight')
plt.show()

# 5- Mapeando os sentimentos

def mapeando_sentimento(avaliacao):
    avaliacao = int(avaliacao)
    if avaliacao <=2:
        return 0
    elif avaliacao == 3:
        return 1
    else:
        return 2

df['sentimento'] = df['score'].apply(mapeando_sentimento)
nomes_sentimentos = ['Negativo', 'Neutro', 'Positivo']


plt.figure(figsize=(8,5))
ax = sns.countplot(x='sentimento', data=df, palette='Set2')
ax.set_xticks(range(len(nomes_sentimentos)))
ax.set_xticklabels(nomes_sentimentos)
plt.xlabel("Sentimento")
plt.ylabel("Quantidade")
plt.title("Distribuição de sentimentos")

for i , p in enumerate(ax.patches):
    altura = p.get_height()
    ax.text(p.get_x() +  p.get_width()/2., altura + 50,f'{int(altura)}', ha="center", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('distribuicao_sentimentos.png', dpi=300, bbox_inches='tight')
plt.show()

df.to_csv('reviews.csv', index=False)
print("Dados salvos em 'reviews.csv'")

# 6 - Pré processamento dos dados

def limpando_dados(texto):
    if pd.isna(texto):
        return ""
    
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

df['content_limpo'] = df['content'].apply(limpando_dados)
df = df[df['content_limpo'].str.len() > 10]
print(f"Review apos a limpeza: {len(df)}")

# 7 -- BERT

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

nome_modelo = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
modelo_bert = AutoModelForSequenceClassification.from_pretrained(
    nome_modelo,
    num_labels=3
)


# 8 -- Dataset BERT
class ReviewDataset(Dataset):
    def __init__(self,textos,sentimentos,tokenizer, max_length=128):
        self.textos = textos
        self.sentimentos = sentimentos
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, index):
        texto = str(self.textos.iloc[index])
        sentimento = self.sentimentos.iloc[index]
        
        # Tokenização
        encoding = self.tokenizer(
            texto,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sentimento, dtype=torch.long)
        }

# 9 - Divisão dos dados BERT

print("\n" + "="*50)
print("Preparando dados")
print("="*50)

def criar_dataset_balanceado(df, max_por_classe=400):
    datasets = []
    for classe in [0,1,2]:
        classe_data = df[df['sentimento'] == classe]
        if len(classe_data) > max_por_classe:
            classe_data = classe_data.sample(max_por_classe,random_state = RANDOM_SEED)
        datasets.append(classe_data)
    
    return pd.concat(datasets , ignore_index=True).sample(frac=1 , random_state=RANDOM_SEED)

df_balanceado = criar_dataset_balanceado(df, max_por_classe=400)

print(f"Dataset balanceado criado: {len(df_balanceado)} reviews")
print(f"Distribuição: Negativo={sum(df_balanceado['sentimento']==0)}, "
      f"Neutro={sum(df_balanceado['sentimento']==1)}, "
      f"Positivo={sum(df_balanceado['sentimento']==2)}")


df_treino , df_temp = train_test_split(
    df_balanceado,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=df_balanceado['sentimento']
)

df_validacao, df_test = train_test_split(
    df_temp,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=df_temp['sentimento']
)

print(f"\nDivisão dos dados:")
print(f"  Treino: {len(df_treino)} ({len(df_treino)/len(df_balanceado)*100:.1f}%)")
print(f"  Validação: {len(df_validacao)} ({len(df_validacao)/len(df_balanceado)*100:.1f}%)")
print(f"  Teste: {len(df_test)} ({len(df_test)/len(df_balanceado)*100:.1f}%)")


BATCH_SIZE = 8
MAX_LENGTH = 192

treino_dataset = ReviewDataset(df_treino['content_limpo'], df_treino['sentimento'], tokenizer, MAX_LENGTH)
validacao_dataset = ReviewDataset(df_validacao['content_limpo'], df_validacao['sentimento'], tokenizer, MAX_LENGTH)
test_dataset = ReviewDataset(df_test['content_limpo'], df_test['sentimento'], tokenizer, MAX_LENGTH)

carregar_treino  = DataLoader(treino_dataset,batch_size=BATCH_SIZE, shuffle=True)
carregar_validacao  = DataLoader(validacao_dataset,batch_size=BATCH_SIZE, shuffle=True)
carregar_test  = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=True)

print(f"Batches de treino: {len(carregar_treino)}")
print(f"Batches de validação: {len(carregar_validacao)}")
print(f"Batches de teste: {len(carregar_test)}")

# 10 -- Treinamento BERT

def treinar_bert_eficiente(modelo, carregamento_treino, carregamento_validacao, epocas = 5, learning_rate=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo de treino: {device}")
    modelo = modelo.to(device)
    
    optimizador = AdamW(
        modelo.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )
    
    total_steps = len(carregamento_treino) * epocas
    scheduler = get_linear_schedule_with_warmup(
        optimizador,
        num_warmup_steps=int(0.1*total_steps),
        num_training_steps=total_steps
    )
    
    melhor_acuracia = 0
    status_melhor_modelo = None
    historico = {'train_loss': [], 'val_accuracy': []}
    
    print("\nIniciando treinamento")
    
    for epoca in range(epocas):
        print(f"\n{'='*40}")
        print(f"ÉPOCA  {epoca+1}/{epocas}")
        print(f"{'='*40}")
        
        modelo.train()
        perda_total_treino = 0
        passo_treinamento = 0
        
        progresso_treinamento = tqdm(carregamento_treino, desc=f"Treino Época {epoca + 1}",leave=False)
        
        for batch in progresso_treinamento:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizador.zero_grad()
                
                outputs = modelo(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                perda = outputs.loss
                perda_total_treino += perda.item()
                
                if perda.requires_grad:
                    perda.backward()
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                    optimizador.step()
                    scheduler.step()

                passo_treinamento += 1
                
                progresso_treinamento.set_postfix({
                    'Perda': f'{perda.item():.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Erro de memória, pulando batch")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue
                else:
                    print(f"Erro no batch: {e}")
                    continue
            except Exception as e:
                print(f"Erro inesperado: {e}")
                continue
        modelo.eval()
        
        predicoes_val = []
        true_labels_val = []
        
        with torch.no_grad():
            progresso_val = tqdm(carregamento_validacao,desc=f"Validação Época {epoca+1}", leave=False)
            
            for batch in progresso_val:
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = modelo(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    preds = torch.argmax(outputs.logits, dim=1)
                    predicoes_val.extend(preds.cpu().numpy())
                    true_labels_val.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"Erro na validação: {e}")
                    continue
        
        perda_media_treino = perda_total_treino / passo_treinamento if passo_treinamento > 0 else 0
        
        if len(predicoes_val) > 0:
            acuracia_val = accuracy_score(true_labels_val,predicoes_val)
            
            
            if epoca == epocas -1:
                 print(f"\nRelatorio de validação (Época {epoca+1}):")
                 print(classification_report(true_labels_val, predicoes_val,target_names=nomes_sentimentos, digits=4))
            
            print(f"\nResumo época {epoca+1}:")
            print(f"  Loss treino: {perda_media_treino:.4f}")
            print(f"  Acurácia validação: {acuracia_val:.4f}")
            historico['train_loss'].append(perda_media_treino)
            historico['val_accuracy'].append(acuracia_val)
            
            # Salva melhor modelo
            if acuracia_val > melhor_acuracia:
                melhor_acuracia = acuracia_val
                status_melhor_modelo = copy.deepcopy(modelo.state_dict())
                print(f" Melhor modelo! Acurácia: {acuracia_val:.4f}")
        else:
            print("Validação falhou")
        
    if status_melhor_modelo is not None:
        modelo.load_state_dict(status_melhor_modelo)
        print(f"\n{'='*50}")
        print(f"Treinamento concluido!")
        print(f"Melhor acurácia de validação: {melhor_acuracia:.4f}")
        print(f"{'='*50}")
    else:
        print(" Nenhum modelo válido foi salvo")
    
    return modelo, historico

print("\nIniciando treinamento bert")
modelo_treinado, historico = treinar_bert_eficiente(
    modelo_bert, 
    carregar_treino, 
    carregar_validacao, 
    epocas=5,           # 3 épocas para melhor aprendizado
    learning_rate=2e-5  # Taxa de aprendizado padrão para BERT
)

# 11 - Avaliação completa no teste

def avaliar_modelo_completo(modelo, carregamento_teste, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)
    modelo.eval()
    
    todas_predicoes  = []
    all_true_labels = []
    todas_probabilidades = []
    
    print(f"\n{'='*50}")
    print("Avaliação no conjunto de teste")
    print(f"{'='*50}")
    
    with torch.no_grad():
        progresso_teste = tqdm(carregamento_teste, desc="Processando teste")
        
        for batch in progresso_teste:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = modelo(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probabilities, dim=1)
                
                todas_predicoes.extend(preds.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                todas_probabilidades.extend(probabilities.cpu().numpy())
                
            except Exception as e:
                print(f"Erro no batch de teste: {e}")
                continue
            
    if len(todas_predicoes) == 0:
        print(" Nenhuma predição foi feita no teste")
        return 0
    
    # Métricas detalhadas
    acuracia = accuracy_score(all_true_labels, todas_predicoes)
    
    print(f"\nResultados finais:")
    print(f"Acurácia: {acuracia:.4f}")
    
    print(f"\nRelatorio Detalhado:")
    print(classification_report(all_true_labels, todas_predicoes, target_names=nomes_sentimentos, digits=4))
    

    try:
        cm = confusion_matrix(all_true_labels, todas_predicoes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=nomes_sentimentos, yticklabels=nomes_sentimentos,
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
    
    return acuracia


acuracia_final = avaliar_modelo_completo(modelo_treinado, carregar_test, tokenizer)


def prever_sentimento_avancado(texto, modelo, tokenizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelo = modelo.to(device)
    modelo.eval()
    
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
            outputs = modelo(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities, dim=1).values.item()
        
        resultado = {
            'sentimento': nomes_sentimentos[predicted_class],
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


print(f"\n{'='*50}")
print("Teste de predições com modelo treinado")
print(f"{'='*50}")

exemplos_teste = [
    "O aplicativo trava toda hora quando tento finalizar a compra. Péssimo!",
    "Entrega demorou demais, mas o atendimento foi bom.",
    "App muito bom! Consegui pedir meus remédios sem sair de casa.",
    "Interface confusa e difícil de achar os produtos, precisa melhorar.",
    "Excelente! Entrega super rápida e ainda ganhei desconto.",
    "Aplicativo razoável, poderia ter mais opções de pagamento.",
    "Não consegui usar o cupom de desconto, fiquei frustrado.",
    "Perfeito! Já comprei várias vezes e sempre chega antes do prazo.",
    "Muito bom, mas às vezes o app fecha sozinho durante o pedido.",
    "Horrível, não reconhece meu login e o suporte não responde."
]


print("\nResultado das predições:")
print("-" * 80)

for i, texto in enumerate(exemplos_teste, 1):
    resultado = prever_sentimento_avancado(texto, modelo_treinado, tokenizer)
    
    if resultado:
        print(f"\nExemplo {i}:")
        print(f"Texto: {texto}")
        print(f"Sentimento: {resultado['sentimento']} (Confiança: {resultado['confianca']:.1%})")
        print(f"Detalhes: Neg={resultado['probabilidades']['Negativo']:.1%} | "
              f"Neu={resultado['probabilidades']['Neutro']:.1%} | "
              f"Pos={resultado['probabilidades']['Positivo']:.1%}")
    else:
        print(f"\nExemplo {i}: Falha na predição")

# 13 --- SALVAMENTO E RELATÓRIO FINAL

print(f"\n{'='*50}")
print("Finalizando e salvando recursos")
print(f"{'='*50}")

# Salva o modelo treinado
try:
    modelo_treinado.save_pretrained('modelo_bert_final')
    tokenizer.save_pretrained('modelo_bert_final')
    print(" Modelo salvo em 'modelo_bert_final/'")
    
    # Salva métricas finais
    with open('metricas_treinamento.txt', 'w', encoding='utf-8') as f:
        f.write("Relatorio do treinamento BERT\n")
        f.write("=" * 40 + "\n")
        f.write(f"Acurácia final: {acuracia_final:.4f}\n")
        f.write(f"Total de reviews: {len(df_balanceado)}\n")
        f.write(f"Reviews de treino: {len(df_treino)}\n")
        f.write(f"Reviews de validação: {len(df_validacao)}\n")
        f.write(f"Reviews de teste: {len(df_test)}\n")
        f.write(f"Data de treinamento: {pd.Timestamp.now()}\n")
    
    print("Métricas salvas em 'metricas_treinamento.txt'")
    
except Exception as e:
    print(f" Erro ao salvar recursos: {e}")

print(f"\n{'='*50}")
print("Progama concluido!")
print(f"{'='*50}")
print(f"Acurácia final alcançada: {acuracia_final:.1%}")
print(f"Modelo salvo para uso futuro")
print(f"Pronto para analisar sentimentos de novos reviews!")
print(f"{'='*50}")
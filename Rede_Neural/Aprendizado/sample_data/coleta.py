"""
ANÁLISE DE SENTIMENTOS DE REVIEWS - GOOGLE PLAY STORE
Código otimizado com rede neural LSTM mais leve e rápida
"""

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Google Play Scraper
from google_play_scraper import Sort, reviews, app

# PyTorch
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict, Counter
import re


# ----- 1. CONFIGURAÇÃO INICIAL ---------

"""
Define seed para reprodutibilidade e configura o dispositivo (CPU/GPU)
"""
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo sendo usado: {device}")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# ----- 2. COLETA DE DADOS DOS APLICATIVOS ---------

"""
Coleta informações básicas sobre os apps de delivery de comida
"""
apps_ids = [
    'br.com.brainweb.ifood',           # iFood
    'com.cerveceriamodelo.modelonow',  # Modelo Now
    'com.mcdo.mcdonalds',              # McDonald's
    'habibs.alphacode.com.br',         # Habib's
    'com.ubercab.eats',                # Uber Eats
    'com.grability.rappi',             # Rappi
    'burgerking.com.br.appandroid',    # Burger King
    'com.vanuatu.aiqfome'              # Aiqfome
]

app_infos = []
print("\nColetando informações dos aplicativos")
for ap in tqdm(apps_ids, desc="Apps"):
    try:
        info = app(ap, lang='pt', country='br')
        info.pop('comments', None)  # Remove comentários desnecessários
        app_infos.append(info)
    except Exception as e:
        print(f"Erro ao coletar {ap}: {e}")
        continue

app_infos_df = pd.DataFrame(app_infos)
print(f"{len(app_infos_df)} aplicativos coletados")
print(app_infos_df[['title', 'score', 'installs']].head())


# ----- 3. COLETA DE REVIEWS ---------

"""
Coleta reviews dos usuários com diferentes notas e ordenações
- Mais reviews para nota 3 (neutro) pois é menos comum
- Sleep para evitar bloqueio da API
"""

app_reviews = []
print("\nColetando reviews dos usuários")
for ap in tqdm(apps_ids, desc="Reviews"):
    for score in range(1, 6):  # Notas de 1 a 5
        for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
            try:
                import time
                time.sleep(1)  # Previne bloqueio
                count = 200 if score == 3 else 100
                rvs, _ = reviews(
                    ap,
                    lang='pt',
                    country='br',
                    sort=sort_order,
                    count=count,
                    filter_score_with=score
                )
                for r in rvs:
                    r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
                    r['appId'] = ap
                app_reviews.extend(rvs)
            except Exception as e:
                print(f"Erro {ap} score {score}: {e}")
                continue

df = pd.DataFrame(app_reviews)
print(f"Total de reviews coletados: {len(df)}")


# ----- 4. ANÁLISE EXPLORATÓRIA ---

"""
Visualiza a distribuição das notas dadas pelos usuários
"""
plt.figure(figsize=(10,5))
sns.countplot(x='score', data=df, palette='viridis')
plt.xlabel('Nota do Review')
plt.ylabel('Quantidade')
plt.title('Distribuição das Notas dos Usuários')
plt.tight_layout()
plt.savefig('distribuicao_notas.png', dpi=300, bbox_inches='tight')
plt.show()


# ---- 5. MAPEAMENTO DE SENTIMENTOS ----

"""
Converte notas em 3 categorias de sentimento:
- Negativo (1-2 estrelas) = 0
- Neutro (3 estrelas) = 1  
- Positivo (4-5 estrelas) = 2
"""
def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0  # Negativo
    elif rating == 3:
        return 1  # Neutro
    else:
        return 2  # Positivo

df['sentiment'] = df['score'].apply(to_sentiment)
class_names = ['Negativo', 'Neutro', 'Positivo']

plt.figure(figsize=(8,5))
ax = sns.countplot(x='sentiment', data=df, palette='Set2')
ax.set_xticks(range(len(class_names)))
ax.set_xticklabels(class_names)
plt.xlabel("Sentimento")
plt.ylabel("Quantidade")
plt.title("Distribuição dos Sentimentos")
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 50,
            f'{int(height)}', ha="center", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('distribuicao_sentimentos.png', dpi=300, bbox_inches='tight')
plt.show()

# Salva os dados
df.to_csv('reviews.csv', index=False)
print("Dados salvos em 'reviews.csv'")


# ---- 6. PRÉ-PROCESSAMENTO DE TEXTO ----

"""
Limpa e prepara o texto para o modelo
- Remove URLs, menções, caracteres especiais
- Converte para minúsculas
"""
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove menções
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove pontuação
    text = re.sub(r'\s+', ' ', text).strip()  # Remove espaços extras
    return text

df['content_clean'] = df['content'].apply(clean_text)
df = df[df['content_clean'].str.len() > 10]  # Remove reviews muito curtos
print(f"Reviews após limpeza: {len(df)}")


# --- 7. CONSTRUÇÃO DO VOCABULÁRIO ---

"""
Cria vocabulário das palavras mais frequentes
- Limita a 5000 palavras para eficiência
- Adiciona tokens especiais: <PAD>, <UNK>
"""
MAX_VOCAB_SIZE = 5000
MAX_LEN = 50  # Comprimento máximo da sequência

def build_vocab(texts, max_size):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

print("Construindo vocabulário.")
vocab = build_vocab(df['content_clean'], MAX_VOCAB_SIZE)
print(f"Vocabulário criado com {len(vocab)} palavras")

def text_to_sequence(text, vocab, max_len):
    """Converte texto em sequência de índices"""
    words = text.split()[:max_len]
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    # Padding
    sequence += [vocab['<PAD>']] * (max_len - len(sequence))
    return sequence


# 8. DATASET PYTORCH

"""
Classe Dataset customizada para carregar dados em batches
"""
class ReviewDataset(Dataset):
    def __init__(self, texts, sentiments, vocab, max_len):
        self.texts = texts
        self.sentiments = sentiments
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        sentiment = self.sentiments.iloc[idx]
        
        sequence = text_to_sequence(text, self.vocab, self.max_len)
        
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'sentiment': torch.tensor(sentiment, dtype=torch.long)
        }


# 9. DIVISÃO DOS DADOS

"""
Divide dados em treino (80%), validação (10%) e teste (10%)
Usa stratify para manter proporção de classes
"""
df_train, df_temp = train_test_split(
    df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['sentiment']
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=df_temp['sentiment']
)

print(f"\nDivisão dos dados:")
print(f"  Treino: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
print(f"  Validação: {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
print(f"  Teste: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")

# Cria DataLoaders
BATCH_SIZE = 64

train_dataset = ReviewDataset(df_train['content_clean'], df_train['sentiment'], vocab, MAX_LEN)
val_dataset = ReviewDataset(df_val['content_clean'], df_val['sentiment'], vocab, MAX_LEN)
test_dataset = ReviewDataset(df_test['content_clean'], df_test['sentiment'], vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# 10. MODELO DE REDE NEURAL - LSTM

"""
Arquitetura da rede:
1. Embedding: Converte palavras em vetores densos (128 dimensões)
2. LSTM: Rede recorrente bidirecional que captura contexto (128 unidades)
3. Dropout: Previne overfitting (30%)
4. Fully Connected: Camada densa final para classificação (3 classes)

LSTM é mais leve que BERT e roda rápido em CPU
"""
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super().__init__()
        
        # Camada de Embedding: transforma índices em vetores
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM bidirecional: processa sequência nas duas direções
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)
        
        # Camada final: hidden_dim * 2 porque é bidirecional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, sequences):
        # sequences: [batch_size, seq_len]
        
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(sequences))
        
        # LSTM: output = [batch_size, seq_len, hidden_dim*2]
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatena último estado oculto das duas direções
        # hidden: [n_layers*2, batch_size, hidden_dim]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden: [batch_size, hidden_dim*2]
        
        hidden = self.dropout(hidden)
        
        # Classificação final
        return self.fc(hidden)

# Inicializa o modelo
model = SentimentLSTM(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    output_dim=len(class_names),
    n_layers=2,
    dropout=0.3
).to(device)

print(f"\nModelo criado:")
print(f"  Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# 11. CONFIGURAÇÃO DO TREINAMENTO

"""
- Otimizador Adam: adapta taxa de aprendizado
- CrossEntropyLoss: função de perda para classificação multiclasse
- 10 épocas de treinamento
"""
EPOCHS = 10
LEARNING_RATE = 0.001

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()


# 12. FUNÇÕES DE TREINO E AVALIAÇÃO

"""
Funções auxiliares para treinar e avaliar o modelo
"""
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Treinando", leave=False):
        sequences = batch['sequence'].to(device)
        sentiments = batch['sentiment'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(sequences)
        loss = criterion(outputs, sentiments)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += sentiments.size(0)
        correct += (predicted == sentiments).sum().item()
    
    return total_loss / len(dataloader), correct / total

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            sentiments = batch['sentiment'].to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, sentiments)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += sentiments.size(0)
            correct += (predicted == sentiments).sum().item()
    
    return total_loss / len(dataloader), correct / total


# 13. TREINAMENTO DO MODELO

"""
Loop de treinamento: treina e valida por múltiplas épocas
Salva o melhor modelo baseado na acurácia de validação
"""
print("\n🚀 Iniciando treinamento...\n")

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0

for epoch in range(EPOCHS):
    print(f"Época {epoch+1}/{EPOCHS}")
    print("-" * 50)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"  Treino    - Loss: {train_loss:.4f} | Acurácia: {train_acc:.4f}")
    print(f"  Validação - Loss: {val_loss:.4f} | Acurácia: {val_acc:.4f}")
    
    # Salva melhor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"Melhor modelo salvo! (Acurácia: {val_acc:.4f})")
    
    print()

print(f"Treinamento concluído!")
print(f"Melhor acurácia de validação: {best_val_acc:.4f}")


# 14. VISUALIZAÇÃO DO TREINAMENTO

"""
Plota curvas de aprendizado
"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss
ax1.plot(history['train_loss'], label='Treino', marker='o')
ax1.plot(history['val_loss'], label='Validação', marker='s')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.set_title('Evolução do Loss')
ax1.legend()
ax1.grid(True)

# Acurácia
ax2.plot(history['train_acc'], label='Treino', marker='o')
ax2.plot(history['val_acc'], label='Validação', marker='s')
ax2.set_xlabel('Época')
ax2.set_ylabel('Acurácia')
ax2.set_title('Evolução da Acurácia')
ax2.legend()
ax2.grid(True)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()


# 15. AVALIAÇÃO NO CONJUNTO DE TESTE

"""
Testa o modelo final em dados nunca vistos
"""
print("\nAvaliando no conjunto de teste...")

# Carrega melhor modelo
model.load_state_dict(torch.load('best_model.pt'))

test_loss, test_acc = eval_model(model, test_loader, criterion, device)
print(f"Acurácia no teste: {test_acc:.4f}")

# Predições detalhadas
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        sequences = batch['sequence'].to(device)
        sentiments = batch['sentiment'].to(device)
        
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(sentiments.cpu().numpy())

# Relatório de classificação
print("\n📋 Relatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Matriz de confusão
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


# 16. FUNÇÃO DE PREDIÇÃO PARA NOVOS TEXTOS

"""
Permite fazer predições em textos novos
"""
def predict_sentiment(text, model, vocab, max_len, device):
    model.eval()
    
    # Preprocessa o texto
    cleaned = clean_text(text)
    sequence = text_to_sequence(cleaned, vocab, max_len)
    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return class_names[predicted_class], probabilities[0].cpu().numpy()

# Testa com exemplos
print("\nTestando predições:")
exemplos = [
    "Péssimo app, não funciona e a entrega sempre atrasa!",
    "É ok, nada de especial mas cumpre o que promete",
    "Excelente aplicativo! Entrega rápida e comida sempre quentinha"
]

for texto in exemplos:
    sentimento, probs = predict_sentiment(texto, model, vocab, MAX_LEN, device)
    print(f"\nTexto: {texto}")
    print(f"Sentimento: {sentimento}")
    print(f"Probabilidades: Neg={probs[0]:.2%} | Neu={probs[1]:.2%} | Pos={probs[2]:.2%}")

print("\nPipeline completo executado com sucesso!")
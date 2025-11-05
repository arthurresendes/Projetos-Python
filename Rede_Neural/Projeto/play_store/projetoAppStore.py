import json
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict, Counter
import re
import os
import requests

# ===============================
# 1. CONFIGURAÇÕES INICIAIS
# ===============================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# ===============================
# 2. COLETA DE DADOS - APP STORE
# ===============================

aplicativos_ios = [
    # "id6446850878", -- Exemplo
    "id1045677511",
    "id1233364203",
    "id1093925421",
    "id1328994276"
]

def coletar_reviews_appstore(app_id, paginas=4):
    """Baixa avaliações do app usando a API pública da Apple"""
    comentarios = []
    for pagina in range(1, paginas + 1):
        try:
            url = f"https://itunes.apple.com/rss/customerreviews/page={pagina}/id={app_id}/sortby=mostrecent/json"
            resposta = requests.get(url)
            if resposta.status_code != 200:
                print(f"Erro {resposta.status_code} ao coletar app {app_id}, página {pagina}")
                break
            dados = resposta.json()
            if "feed" not in dados or "entry" not in dados["feed"]:
                break
            for entrada in dados["feed"]["entry"]:
                if "im:rating" in entrada:
                    comentarios.append({
                        "appId": app_id,
                        "autor": entrada.get("author", {}).get("name", {}).get("label", ""),
                        "titulo": entrada.get("title", {}).get("label", ""),
                        "content": entrada.get("content", {}).get("label", ""),
                        "score": int(entrada["im:rating"]["label"])
                    })
        except Exception as e:
            print(f"Erro ao coletar página {pagina} do app {app_id}: {e}")
            continue
    return comentarios

print("\nColetando avaliações da App Store...")

reviews_appstore = []
for app_id in tqdm(aplicativos_ios, desc="Apps iOS"):
    dados_app = coletar_reviews_appstore(app_id, paginas=10)
    reviews_appstore.extend(dados_app)

df = pd.DataFrame(reviews_appstore)
print(f"\nTotal de reviews coletados: {len(df)}")

# ===============================
# 3. ANÁLISE BÁSICA E GRÁFICOS
# ===============================

if len(df) > 0:
    plt.figure(figsize=(10, 5))
    sns.countplot(x='score', data=df, palette='viridis')
    plt.xlabel('Notas das Avaliações')
    plt.ylabel('Quantidade')
    plt.title('Distribuição das Notas dos Usuários (App Store)')
    plt.tight_layout()
    plt.savefig('distribuicao_notas_appstore.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===============================
# 4. MAPEAMENTO DE SENTIMENTOS
# ===============================

def mapear_sentimento(avaliacao):
    avaliacao = int(avaliacao)
    if avaliacao <= 2:
        return 0
    elif avaliacao == 3:
        return 1
    else:
        return 2

df['sentimento'] = df['score'].apply(mapear_sentimento)
nomes_sentimentos = ['Negativo', 'Neutro', 'Positivo']

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='sentimento', data=df, palette='Set2')
ax.set_xticks(range(len(nomes_sentimentos)))
ax.set_xticklabels(nomes_sentimentos)
plt.xlabel("Sentimento")
plt.ylabel("Quantidade")
plt.title("Distribuição de Sentimentos (App Store)")
plt.tight_layout()
plt.savefig('distribuicao_sentimentos_appstore.png', dpi=300, bbox_inches='tight')
plt.show()

# ===============================
# 5. PRÉ-PROCESSAMENTO
# ===============================

def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

df["content_limpo"] = df["content"].apply(limpar_texto)
df = df[df["content_limpo"].str.len() > 10]
print(f"Reviews após limpeza: {len(df)}")

# ===============================
# 6. BERT - TOKENIZAÇÃO E MODELO
# ===============================
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

nome_modelo = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
modelo_bert = AutoModelForSequenceClassification.from_pretrained(
    nome_modelo,
    num_labels=3
)

# ===============================
# 7. DATASET
# ===============================
class ReviewDataset(Dataset):
    def __init__(self, textos, sentimentos, tokenizer, max_length=128):
        self.textos = textos
        self.sentimentos = sentimentos
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, index):
        texto = str(self.textos.iloc[index])
        sentimento = self.sentimentos.iloc[index]
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

# ===============================
# 8. DIVISÃO DOS DADOS
# ===============================
def criar_dataset_balanceado(df, max_por_classe=400):
    conjuntos = []
    for classe in [0, 1, 2]:
        dados = df[df["sentimento"] == classe]
        if len(dados) > max_por_classe:
            dados = dados.sample(max_por_classe, random_state=RANDOM_SEED)
        conjuntos.append(dados)
    return pd.concat(conjuntos, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)

df_bal = criar_dataset_balanceado(df)

df_train, df_temp = train_test_split(df_bal, test_size=0.2, random_state=RANDOM_SEED, stratify=df_bal["sentimento"])
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=df_temp["sentimento"])

BATCH_SIZE = 8
MAX_LEN = 192

train_loader = DataLoader(ReviewDataset(df_train["content_limpo"], df_train["sentimento"], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ReviewDataset(df_val["content_limpo"], df_val["sentimento"], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ReviewDataset(df_test["content_limpo"], df_test["sentimento"], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# 9. TREINAMENTO
# ===============================
def treinar(modelo, train_loader, val_loader, epocas=5, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)
    optim = AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=len(train_loader) * epocas)
    melhor_acc = 0

    for epoca in range(epocas):
        modelo.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Treino {epoca+1}/{epocas}"):
            optim.zero_grad()
            outputs = modelo(input_ids=batch['input_ids'].to(device),
                             attention_mask=batch['attention_mask'].to(device),
                             labels=batch['labels'].to(device))
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optim.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Loss médio treino: {total_loss/len(train_loader):.4f}")

        # validação
        modelo.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = modelo(input_ids=batch['input_ids'].to(device),
                             attention_mask=batch['attention_mask'].to(device))
                pred = torch.argmax(out.logits, dim=1)
                preds.extend(pred.cpu().numpy())
                labels.extend(batch['labels'].numpy())
        acc = accuracy_score(labels, preds)
        print(f"Acurácia val: {acc:.4f}")
        if acc > melhor_acc:
            melhor_acc = acc
            melhor_modelo = copy.deepcopy(modelo.state_dict())
    modelo.load_state_dict(melhor_modelo)
    print(f"Melhor acurácia validação: {melhor_acc:.4f}")
    return modelo

modelo_treinado = treinar(modelo_bert, train_loader, val_loader)

# ===============================
# 10. FINAL
# ===============================
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


acuracia_final = avaliar_modelo_completo(modelo_treinado, test_loader, tokenizer)



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
    caminho_modelo = r"C:\Users\arregomes\OneDrive - rd.com.br\Modelo_RN\AppStore"
    os.makedirs(caminho_modelo, exist_ok=True)

    modelo_treinado.save_pretrained(caminho_modelo)
    tokenizer.save_pretrained(caminho_modelo)
    print(f"Modelo salvo com sucesso em: {caminho_modelo}")

    with open(os.path.join(caminho_modelo, 'metricas_treinamento.txt'), 'w', encoding='utf-8') as f:
        f.write("Relatório do treinamento BERT - App Store\n")
        f.write("=" * 50 + "\n")
        f.write(f"Acurácia final: {acuracia_final:.4f}\n")
        f.write(f"Total de reviews: {len(df_bal)}\n")
        f.write(f"Reviews de treino: {len(df_train)}\n")
        f.write(f"Reviews de validação: {len(df_val)}\n")
        f.write(f"Reviews de teste: {len(df_test)}\n")
        f.write(f"Data de treinamento: {pd.Timestamp.now()}\n")

    print("Métricas salvas com sucesso!")

except Exception as e:
    print(f"Erro ao salvar modelo ou métricas: {e}")

print(f"\n{'='*50}")
print("Progama concluido!")
print(f"{'='*50}")
print(f"Acurácia final alcançada: {acuracia_final:.1%}")
print(f"Modelo salvo para uso futuro")
print(f"Pronto para analisar sentimentos de novos reviews!")
print(f"{'='*50}")

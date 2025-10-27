from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup # BertTokenizer -> Converte texto em tokens, BertModel -> Modelo bert pré treinado usado para gerar representações vetoriais
PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased' # Modelo pré-treinado em português , cased -> diferencia maiusculas de minusuculas , preservando informações como nomes proprios
from torch.optim import AdamW
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME) # modelo BERT para português (BERTimbau cased), Carrega vocabulário e regras de tokenização do BERTimbau.


sample_txt = "Quem conta um conto aumenta um pouco"
tokens = tokenizer.tokenize(sample_txt) # divide as frases em tokes/subpalavras
token_ids = tokenizer.convert_tokens_to_ids(tokens) # transforma cada token em id numerico para alimentar o modelo
print(f' Sentence: {sample_txt}') # frase original
print(f'   Tokens: {tokens}') # tokens intermediarios
print(f'Token IDs: {token_ids}') #ids bert

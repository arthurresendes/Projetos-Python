import transformers
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
from torch.optim import AdamW
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


sample_txt = "Quem conta um conto aumenta um pouco"
tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')

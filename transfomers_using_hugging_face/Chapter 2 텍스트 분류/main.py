from transformers import DistilBertTokenizer
from transformers import AutoTokenizer


# ----------------------------------------------------------------
# 데이터 관련 - emotions 데이터셋 로드하기
from datasets import load_dataset

emotions = load_dataset('emotion')

# transfer to dataframe
emotions.set_format(type='pandas')
df = emotions['train'][:]




# ----------------------------------------------------------------
# 토큰화 관련
def tokenize(batch):
    model_ckpt = "distilbert-base-uncased"
    tokenizers = AutoTokenizer.from_pretrained(model_ckpt)
    return tokenizers(batch["text"], padding=True, truncation=True)


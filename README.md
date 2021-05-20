# MathBERT

```MathBERT``` is a BERT model trained on mathematics text.

<!-- MathBERT is trained on papers from the corpus of semanticscholar.org. Corpus size is 1.14M papers, 3.1B tokens. We use the full text of the papers in training, not just abstracts.

MathBERT has its own vocabulary (scivocab) that's built to best match the training corpus. We trained cased and uncased versions. We also include models trained on the original BERT vocabulary (basevocab) for comparison.

It results in state-of-the-art performance on a wide range of scientific domain nlp tasks. The details of the evaluation are in the paper. Evaluation code and data are included in this repo. -->

#### Downloading Trained Models
We release the tensorflow and the pytorch version of the trained models. The tensorflow version is compatible with code that works with the model from [Google Research](https://github.com/google-research/bert). The pytorch version is created using the [Hugging Face library](https://github.com/huggingface/transformers), and this repo shows how to use it. 
+ Tensorflow download
  + note to change the model name to ```mathbert-mathvocab-uncased``` for the mathvocab version
  ```
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/bert_config.json
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/vocab.txt
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/bert_model.ckpt.index
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/bert_model.ckpt.meta
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/bert_model.ckpt.data-00000-of-00001
+ Pytorch download
  + MathBERT models now can be installable directly within Huggingface's framework under the namespace tbs17:
```
from transformers import *

tokenizer = AutoTokenizer.from_pretrained('tbs17/MathBERT')
model = AutoModel.from_pretrained('tbs17/MathBERT')

tokenizer = AutoTokenizer.from_pretrained('tbs17/MathBERT-custom')
model = AutoModel.from_pretrained('tbs17/MathBERT-custom')
```




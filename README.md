# MathBERT

```MathBERT``` is a BERT model trained on the below mathematics text.

+ pre-k to high school math curriculum from engageny.org
+ G6-8 math curriculum from utahmiddleschoolmath.org
+ G6-high school math from illustrativemathematics.org
+ high school to college math text books from openculture.com
+ G6-8 math curriculum from ck12.org
+ College to graduate level MOOC math course syllabus from classcentral.com
+ math paper abstracts from arxiv.org

MathBERT has its own vocabulary (```mathVocab```) that's built via ```BertTokenizer``` to best match the training corpus. We also trained MathBERT with the original BERT vocabulary (```baseVocab```) for comparison. Both models are uncased versions.

<!-- It results in state-of-the-art performance on a wide range of scientific domain nlp tasks. The details of the evaluation are in the paper. Evaluation code and data are included in this repo. -->

#### Downloading Trained Models
We release the tensorflow and the pytorch version of the trained models. The tensorflow version is compatible with code that works with the model from [Google Research](https://github.com/google-research/bert). The pytorch version is created using the [Hugging Face library](https://github.com/huggingface/transformers).
+ Tensorflow download
  + note: to download mathbert-mathvocab version, change the model name to ```mathbert-mathvocab-uncased``` in the below code
  ```
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/bert_config.json
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/vocab.txt
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/bert_model.ckpt.index
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/bert_model.ckpt.meta
  wget http://tracy-nlp-models.s3.amazonaws.com/mathbert-basevocab-uncased/bert_model.ckpt.data-00000-of-00001
+ Pytorch download
  + MathBERT models now can be installable directly within Huggingface's framework under the name space tbs17 at https://huggingface.co/tbs17/MathBERT or https://huggingface.co/tbs17/MathBERT-custom.
```
from transformers import *

tokenizer = AutoTokenizer.from_pretrained('tbs17/MathBERT')
model = AutoModel.from_pretrained('tbs17/MathBERT')

tokenizer = AutoTokenizer.from_pretrained('tbs17/MathBERT-custom')
model = AutoModel.from_pretrained('tbs17/MathBERT-custom')
```

#### Pretraining and fine-tuning

The pretraining code is located at /mathbert/ and fine-tuning notebook is at /scripts/MathBERT_finetune.ipynb. Unfortunately, we can't release the fine-tuning data set per the data owner's request. All the packages we use is in the requirements.txt file. 




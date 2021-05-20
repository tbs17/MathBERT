# MathBERT

```MathBERT``` is a BERT model trained on mathematics text.

<!-- MathBERT is trained on papers from the corpus of semanticscholar.org. Corpus size is 1.14M papers, 3.1B tokens. We use the full text of the papers in training, not just abstracts.

MathBERT has its own vocabulary (scivocab) that's built to best match the training corpus. We trained cased and uncased versions. We also include models trained on the original BERT vocabulary (basevocab) for comparison.

It results in state-of-the-art performance on a wide range of scientific domain nlp tasks. The details of the evaluation are in the paper. Evaluation code and data are included in this repo. -->

#### Downloading Trained Models
Update! MathBERT models now installable directly within Huggingface's framework under the namespace tbs17:
```
from transformers import *

tokenizer = AutoTokenizer.from_pretrained('tbs17/MathBERT')
model = AutoModel.from_pretrained('tbs17/MathBERT')

tokenizer = AutoTokenizer.from_pretrained('tbs17/MathBERT-custom')
model = AutoModel.from_pretrained('tbs17/MathBERT-custom')
```
We release the tensorflow and the pytorch version of the trained models. The tensorflow version is compatible with code that works with the model from [Google Research](https://github.com/google-research/bert). The pytorch version is created using the [Hugging Face library](https://github.com/huggingface/transformers), and this repo shows how to use it. All combinations of mathvocab and basevocab uncased models are available below. Our evaluation shows that mathvocab-uncased usually gives the best results.

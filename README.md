# CS247_W22_Project
## Kaggle Competition: Feedback Prize - Evaluating Student Writing
The Data Folder are provided by Kaggle contest, as the link below.

https://www.kaggle.com/c/feedback-prize-2021

**Visualization.ipynb** presents some basic features of the dataset, done by Yuchen Liu.

**BERT_baseline.ipynb** is finished by Yuchen Liu, providing fully functional word level solver of this problem. Since the Bert model limits input size to 512 tokens, the longer part of some texts are truncated. 

**LSTM_baseline.ipynb** is finished by Yuchen Liu, providing fully functional sentence level solver of this problem. This model takes sentences as input, categorically classifying them without context. This notebook provides some simpy pre-processing and adjustable training parameters. **LSTM.py** is modified from it and run on his personal computer. Since the nature of RNN, PC is faster and than colab environment.

**BERT_experiment_XinyuZhao.ipynb** is a sentence level model trained by Xinyu Zhao.

**longformer_ner.ipynb** is a word level model trained by Ting-Po Huang.

**seq2seq.py**: implement LSTM with prior knowledge (not finished)

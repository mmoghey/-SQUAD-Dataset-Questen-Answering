# SQUAD-Dataset-Question-AnsweringQuestion Answering the SQUAD dataset Using Attention and HMN


Files :
data.py -- containes the loading and preprocessing of SQUAD Dataset
model.py -- contains the implementation of encoder and decoder model
train.py -- contains the implementation of training phase and test phase including F1 Score
setup.py -- contains steps to get the dataset from needed links and setup the system by installing needed softwares

Dataset:
Used squad dataset (training and test) which can be obtained from locations
https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
Alternatively, setup.py can be use to download the data

Glove Embeddings:
Glove embeddings can be downloaded fromwget http://nlp.stanford.edu/data/glove.840B.300d.zip.
Alternatively, setup.py can be used to download and prepare glove embeddings.

How To Run:
After downloading needed libraries and SQUAD dataset using setup.py run following command:
	py -3 train.py

Colab Version:
This this project was implemented on colab.google. Here is the copy of actual code on colab
https://colab.research.google.com/drive/1UZMXnGvbH7APZVej8cjjtsD1eVNZO7oE#scrollTo=eIzW9FmHZvVi


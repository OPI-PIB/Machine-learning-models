# Machine learning models offered by the National Information Processing Institute

## Neural language models 

### RoBERTa

A set of Polish neural language models that rely on the Transformer architecture and are trained using masked language modelling and the techniques described in [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). Two sizes of model are available: base and large. The base models contain approximately 100 million parameters; the large models contain 350 million parameters. The large models offer higher prediction quality in practical use, but require more computational resources. Large Polish text corpora (20–200 GB) were used to train the models. Each model comes in two variants, which makes it possible to read them in popular machine learning libraries: [Fairseq](https://github.com/pytorch/fairseq) and [Hugginface Transformers](https://github.com/huggingface/transformers).

Fairseq models: [base (version 1)](https://share.opi.org.pl/s/YammFDDFyymxHjA), [base (version 2)](https://share.opi.org.pl/s/X78QyWBXmbTmWTr), [large (version 1)](https://share.opi.org.pl/s/TBM8q5Bzrqaa5XF), [large (version 2)](https://share.opi.org.pl/s/zwK4mofafDtgBx2)

Huggingface Transformers models: [base (version 1)](https://share.opi.org.pl/s/j9A9Fmij6smDTe8), [base (version 2)](https://share.opi.org.pl/s/JonE4qDDjzsQAtT), [large (version 1)](https://share.opi.org.pl/s/RAmxCTKDNY4naWe), [large (version 2)](https://share.opi.org.pl/s/FTpq7ceAgdeyR5k)

### BART

A Transformer neural language model that uses the encoder–decoder architecture. The model was trained on a set of Polish documents of 200 GB using the method described in [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461). The model can be adapted to solving predictive tasks, but is primarily designed to be used in sequence-to-sequence tasks in which documents (for example, from machine translation or chatbots) serve both as the input and the output. The model comes in two variants, which makes it possible to read them in popular machine learning libraries: [Fairseq](https://github.com/pytorch/fairseq) and [Hugginface Transformers](https://github.com/huggingface/transformers).

Download: [Fairseq model](https://share.opi.org.pl/s/aw6o2g7joKS8m6D), [Huggingface Transformers model](https://share.opi.org.pl/s/nHPT3Ln7SBRyb5M)

### GPT-2

A neural language model that is based on the Transformer architecture and trained using the autoregressive language model method. The neural network architecture complies with the English-language GPT-2 models described in [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OPI PIB offers the models in two sizes: medium, which contains approximately 350 million parameters, and large, which contains approximately 700 million parameters. The files are compatible with the [Fairseq library](https://github.com/pytorch/fairseq).

Download: [medium model](https://share.opi.org.pl/s/9p32SjLsASgepqz), [large model](https://share.opi.org.pl/s/TGXs2CytKnTbjNx)

### ELMo

A language model that is based on the long short-term memory recurrent neural networks presented in [Deep contextualized word representations](https://arxiv.org/abs/1802.05365). The Polish-language model can be read by the [AllenNLP](https://github.com/allenai/allennlp).

Download: [model](https://share.opi.org.pl/s/KrKRTytyQp7yka9)


## Static representations of words

### Word2Vec

Classic vector representations of words for the Polish language trained using the method described in [Distributed Representations of Words and Phrases
and their Compositionality](https://arxiv.org/abs/1310.4546). A large corpus of Polish-language documents was used to train the vectors. The set contains approximately 2 million words—including the words that appear at least three times in the corpus—and other defined symbol categories, such as punctuation marks, numbers from 0 to 10,000, and Polish forenames and surnames. The vectors are compatible with the  [Gensim library](https://radimrehurek.com/gensim/). The vectors offered by OPI PIB range from 100-dimensional to 800-dimensional.

Download: [100d](https://share.opi.org.pl/s/w7eTXQWeAJXX8tP), [300d](https://share.opi.org.pl/s/PnZD2Yck3jQT4ye), [500d](https://share.opi.org.pl/s/NMQXAjbi3yx7gZL), [800d](https://share.opi.org.pl/s/QTz8Jt2gbMmtnkx)

### GloVe

Vector representations of words for the Polish language that have been trained using the [GloVe method](https://aclanthology.org/D14-1162/) developed by experts at Stanford University. A large corpus of documents in Polish was used to train the vectors. The set contains approximately 2 million words—including the words that appear at least three times in the corpus—and other defined symbol categories, such as punctuation marks, numbers from 0 to 10,000, and Polish forenames and surnames. The vectors are saved in a text format that is compatible with various libraries designed for this type of model. The vectors offered by OPI PIB range from 100-dimensional to 800-dimensional.

Download: [100d](https://share.opi.org.pl/s/qeWtsizPZxJZXCY), [300d](https://share.opi.org.pl/s/kzWtFTTWAnNnmS4), [500d](https://share.opi.org.pl/s/TEernXTfFco2EXt), [800d](https://share.opi.org.pl/s/MQ4LisDdagX5DWL)

### FastText

A model that contains vector representations of words and word parts in the Polish language. Unlike traditional, static representations of languages, the model is capable of generating new vectors for the words that are not included in dictionaries, based on the sum of the representations of their parts. The model was trained on a large corpus of Polish-language documents using the method described in [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606). The set contains approximately 2 million words—including the words that appear at least three times in the corpus—and other defined symbol categories, such as punctuation marks, numbers from 0 to 10,000, and Polish forenames and surnames. The vectors are compatible with the [Gensim library](https://radimrehurek.com/gensim/). The vectors offered by OPI PIB range from 100-dimensional to 800-dimensional.

Download: [100d](https://share.opi.org.pl/s/JGwNPApL4NH2Lza), [300d](https://share.opi.org.pl/s/5cGH7xMiJg3FzEW), [500d](https://share.opi.org.pl/s/kgMqjCL7WM3zQ62), [800d (part 1)](https://share.opi.org.pl/s/o2e37A6KsZ4odtd), [800d (part 2)](https://share.opi.org.pl/s/a6926zpKPLy9Bq7)

## Machine translation models

Polish–English and English–Polish models based on convolutional networks. The models are used in machine translation of documents included in the [Fairseq library](https://github.com/pytorch/fairseq). They are based on convolutional neural networks. OPI PIB offers two models: Polish–English and English–Polish. They were trained on the data available on the [OPUS website](http://opus.nlpl.eu/), which comprises a set of 40 million pairs of source and target language sentences.

Download: [Polish–English model](https://share.opi.org.pl/s/ztGPz7q7aHk4CfH), [English–Polish model](https://share.opi.org.pl/s/GTW5n4KdiyFcaAq)

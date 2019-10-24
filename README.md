# Learning to Generate Music with Sentiment

This repository contains the source code to reproduce the results of the [ISMIR'19](https://ismir2019.ewi.tudelft.nl/)
paper [Learning to Generate Music with Sentiment](http://www.lucasnferreira.com/papers/2019/ismir-learning.pdf).
This paper presented a generative LSTM that can be controlled to generate symbolic music with a given sentiment
(positive or negative). The LSTM is controlled by optimizing the weights of specific neurons that are responsible
for the sentiment signal. Such neurons are found plugging a Logistic Regression to the LSTM and training the
Logistic Regression to classify sentiment of symbolic music encoded with the mLSTM hidden states.

## Installing Dependencies

This project depends on tensorflow (2.0), numpy and music21, so you need to install them first:

```
$ pip3 install tensorflow tensorflow-gpu numpy music21
```

## Dowload Data

A new dataset called VGMIDI was created for this paper. It contains 95 labelled pieces according to sentiment as well as 728
non-labelled pieces. All pieces are piano arrangements of video game soundtracks in MIDI format. The VGMIDI dataset has its
own [GitHub repo](https://github.com/lucasnfe/vgmidi). You can download it as follws:

```
$ git clone https://github.com/lucasnfe/vgmidi.git
```

## Results

This paper has two main results: (a) sentiment analysis of symbolic music and (b) generation of symbolic
music with sentiment.

### (a) Sentiment Analysis

#### 1. Train a generative LSTM on unlabelled pieces:

```
python3.7 train_generative.py --train ../vgmidi/unlabelled/train/ --test ../vgmidi/unlabelled/test/ --embed 256 --units 512 --layers 4 --batch 64 --epochs 15 --lrate 0.00001 --seqlen 256 --drop 0.05
```

To sample from this trained generative model (it can't be controlled yet):

```
python3.7 midi_generator.py --model trained --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4
```

#### 2. Encode the labelled pieces with the final cell states of the generative LSTM and train a Logistic Regression to classify sentiment in symbolic music:

```
python3.7 train_classifier.py --model trained --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --train ../vgmidi/labelled/vgmidi_sent_train.csv --test ../vgmidi/labelled/vgmidi_sent_test.csv --cellix 4
```

### (b) Generative

TODO

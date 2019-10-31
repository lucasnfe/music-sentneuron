# Learning to Generate Music with Sentiment

This repository contains the source code to reproduce the results of the [ISMIR'19](https://ismir2019.ewi.tudelft.nl/)
paper [Learning to Generate Music with Sentiment](http://www.lucasnferreira.com/papers/2019/ismir-learning.pdf).
This paper presented a generative LSTM that can be controlled to generate symbolic music with a given sentiment
(positive or negative). The LSTM is controlled by optimizing the weights of specific neurons that are responsible
for the sentiment signal. Such neurons are found plugging a Logistic Regression to the LSTM and training the
Logistic Regression to classify sentiment of symbolic music encoded with the mLSTM hidden states.

## Examples of Generated Pieces

### Positive 
- [Piece 1](https://raw.githubusercontent.com/lucasnfe/music-sentneuron/master/generated/generated_pos1.wav)
- [Piece 2](https://raw.githubusercontent.com/lucasnfe/music-sentneuron/master/generated/generated_pos2.wav)
- [Piece 3](https://raw.githubusercontent.com/lucasnfe/music-sentneuron/master/generated/generated_pos3.wav)

### Negative
- [Piece 3](https://raw.githubusercontent.com/lucasnfe/music-sentneuron/master/generated/generated_neg1.wav)
- [Piece 4](https://raw.githubusercontent.com/lucasnfe/music-sentneuron/master/generated/generated_neg2.wav)
- [Piece 5](https://raw.githubusercontent.com/lucasnfe/music-sentneuron/master/generated/generated_neg3.wav)

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

## Reproducing Results

This paper has two main results: (a) sentiment analysis of symbolic music and (b) generation of symbolic
music with sentiment.

### (a) Sentiment Analysis

To train a sentiment classifier of symbolic music:

#### 1. Train a generative LSTM on unlabelled pieces:

```
python3.7 train_generative.py --train ../vgmidi/unlabelled/train/ --test ../vgmidi/unlabelled/test/ --embed 256 --units 512 --layers 4 --batch 64 --epochs 15 --lrate 0.00001 --seqlen 256 --drop 0.05
```
This script saves a checkpoint of the trained model after every epoch in the "trained/" folder.
To sample from this trained generative model (without sentiment control):

```
python3.7 midi_generator.py --model trained --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4
```

#### 2. Encode the labelled pieces with the final cell states of the generative LSTM and train a Logistic Regression to classify sentiment in symbolic music:

```
python3.7 train_classifier.py --model trained --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --train ../vgmidi/labelled/vgmidi_sent_train.csv --test ../vgmidi/labelled/vgmidi_sent_test.csv --cellix 4
```

After running this script, a binary file named "classifier_ckpt.p" containing the trained classifier logistic
regression is saved in the "trained/" folder.

### (b) Generative

To control the sentiment of the trained generative LSTM, we evolve the values of the neurons activated during
training the logistic regression classifier.

#### 1. Evolve neurons to generate positive pieces

```
evolve_generative.py --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --genmodel trained/ --clsmodel trained/classifier_ckpt.p --cellix 4 --elitism 0.1 --epochs 10 --sent 1
```

After running this script, a json file named "neurons_positive.json" containing the neuron values that control the generative model to be positive is saved in the "trained/" folder.

#### 2. Evolve neurons to generate negative pieces

```
python3.7 midi_generator.py --model trained --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --cellix
```

After running this script, a json file named "neurons_negative.json" containing the neuron values that control the generative model to be netagive is saved in the "trained/" folder.

#### 3. Generate positive pieces

```
python3 midi_generator.py --model trained/ --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --seqlen 512 --override trained/neurons_positive.json --cellix 4
```

#### 4. Generate negative pieces

```
python3 midi_generator.py --model trained/ --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --seqlen 512 --override trained/neurons_negative.json --cellix 4
```

## Citing this Work

If you use this method in your research, please cite:

```
@article{ferreira_ismir_2019,
  title={Learning to Generate Music with Sentiment},
  author={Ferreira, Lucas N. and Whitehead, Jim},
  booktitle = {Proceedings of the Conference of the International Society for Music Information Retrieval},
  series = {ISMIR'19},
  year={2019},
}
```

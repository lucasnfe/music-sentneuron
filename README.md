# Learning to Generate Music with Sentiment

This repository contains the source code to reproduce the results of the [ISMIR'19](https://ismir2019.ewi.tudelft.nl/)
paper [Learning to Generate Music with Sentiment](http://www.lucasnferreira.com/papers/2019/ismir-learning.pdf).
This paper presented a generative LSTM that can be controlled to generate symbolic music with a given sentiment
(positive or negative). The LSTM is controlled by optimizing with a Genetic Algorithm the values of specific neurons that are responsible
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
own [GitHub repo](https://github.com/lucasnfe/vgmidi) that is constantly updated. To reproduce the results of the ISMIR'19 paper, 
use the **VGMIDI 0.1** that can be downloaded as follows:

```
wget https://github.com/lucasnfe/vgmidi/archive/0.1.zip
```

To make sur the paths will match in the rest of this tutorial. Rename the dataset directory from `vgmidi-0.1` to `vgmidi` and make sure the `vgmidi` directory is inside the music-sentneuron directory.

## Encoding Data

To process the MIDI data with neural networks, one needs to encode the MIDI data as a sequence of vectors. To augment encode the In this work, we use a one-hot encoding strategy:

```
python3.7 midi_encoder.py --path vgmidi/unlabelled/train --transp 10 --strech 10
python3.7 midi_encoder.py --path vgmidi/unlabelled/test --transp 10 --strech 10
```

Note that this will create a .txt file for each .mid file in the `train` and `test` directories. Each line in these files represents an encoded augmented version of the original mid piece. For example, in the `train` directory there should be a `Dragon_Warrior_Battle_Theme.txt` file that starts with:

```
t_80 v_68 d_16th_0 n_38 n_41 w_1 v_76 n_41 n_44 n_47 w_1 ...
```

Make sure these txt files exist and are not empty before procedding. 

## Reproducing Results

This paper has two main results: (1) sentiment analysis of symbolic music and (2) generation of symbolic
music with sentiment.

### 1. Sentiment Analysis

To train a sentiment classifier of symbolic music:

#### 1.1 Train a generative LSTM on unlabelled pieces:

```
python3.7 train_generative.py --train ../vgmidi/unlabelled/train/ --test ../vgmidi/unlabelled/test/ --embed 256 --units 512 --layers 4 --batch 64 --epochs 15 --lrate 0.00001 --seqlen 256 --drop 0.05
```
This script saves a checkpoint of the trained model after every epoch in the "trained/" folder. Note that the model/training parameters are slightly different than the ones presented in the paper. These parameters should achieve similar results and are faster to train. Feel free to adjust the pameters to the original ones (`--embed 64, --units 4096 --batch 32 --epochs 4 --lrate 0.000001`). 

To sample from this trained generative model (without sentiment control):
```
python3.7 midi_generator.py --model trained --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4
```

#### 1.2 Train a Logistic Regression to classify sentiment in symbolic music:

The following script will encode the labelled pieces with the final cell states of the generative LSTM and train the logistic regression model:

```
python3.7 train_classifier.py --model trained --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --train ../vgmidi/labelled/vgmidi_sent_train.csv --test ../vgmidi/labelled/vgmidi_sent_test.csv --cellix 4
```

After running this script, a binary file named "classifier_ckpt.p" containing the trained classifier logistic regression is saved in the "trained/" folder.

### 2. Generative

To control the sentiment of the trained generative LSTM, we evolve the values of the neurons activated during
training the logistic regression classifier.

#### 2.1 Evolve neurons to generate positive pieces

```
evolve_generative.py --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --genmodel trained/ --clsmodel trained/classifier_ckpt.p --cellix 4 --elitism 0.1 --epochs 10 --sent 1
```

After running this script, a json file named "neurons_positive.json" containing the neuron values that control the generative model to be positive is saved in the "trained/" folder.

#### 2.2 Evolve neurons to generate negative pieces

```
python3.7 midi_generator.py --model trained --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --cellix
```

After running this script, a json file named "neurons_negative.json" containing the neuron values that control the generative model to be netagive is saved in the "trained/" folder.

#### 2.3 Generate positive pieces

```
python3 midi_generator.py --model trained/ --ch2ix trained/char2idx.json --embed 256 --units 512 --layers 4 --seqlen 512 --override trained/neurons_positive.json --cellix 4
```

#### 2.4 Generate negative pieces

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

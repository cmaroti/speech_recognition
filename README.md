# Speech Recognition and Pacman

## Kaggle Competition Overview
For my final project at Metis, I competed in the [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge) on Kaggle.

The data consisted of ~65,000 1-second recordings of single words, split into a training set and a validation set.

I trained a Convolutional Neural Network using Keras to recognize the 10 words "yes, no, up, down, left, right, stop, go, on, off." The model also classified other words as "unknown" and no word as "silence."

The Kaggle test set contained over 150,000 recordings. 

On the validation set I achieved 90% accuracy. On the Kaggle set I achieved 83% accuracy.

The files [speech-recognition-prepare-data.ipynb](/speech-recognition-prepare-data.ipynb) and [speech-recognition-best-model.ipynb](/speech-recognition-best-model.ipynb) contain my code for loading and preparing the wav files and training the neural network, respectively.

## Ms. Pacman Implementation
Using my model, I built a voice-controlled game of Ms. Pacman with Flask and JavaScript. I used OpenAI Gym's version of Ms. Pacman for the visuals.

You can download [this folder](/pacman_app_live) and follow the directions in the readme to play yourself!

# speech_recognition

For my final project at Metis, I competed in the TensorFlow Speech Recognition Challenge on Kaggle.

The data consisted of ~65,000 1-second recordings of single words, split into a training set and a validation set.

I trained a Convolutional Neural Network using Keras to recognize the 10 words "yes, no, up, down, left, right, stop, go, on, off." The model also labeled other words as "unknown" and no word as "silence."

The Kaggle test set contained over 150,000 recordings. 

On the validation set I achieved 90% accuracy. On the Kaggle set I achieved 83% accuracy.

Using my model, I built a voice-controlled game of Ms. Pacman with Flask and JavaScript. 

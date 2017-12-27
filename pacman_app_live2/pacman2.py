import flask
import numpy as np
import pandas as pd
from copy import deepcopy
import gym
import time
import threading
import keras
import glob
import scipy
from scipy.signal import butter, lfilter, freqz
import librosa
import json
import imageio

# import model
model = keras.models.load_model("model_dec7_3CNNs_melpower.h5")

ONE_SECOND = []
debounce_flag = False
debounce_counter = 0
prediction_window = []
timer = 0

def process_wav(X, max_len = 16000, n_mels = 128):
    """Turns audio into a spectrogram"""
    x_spec = np.zeros((len(X), 128, 32, 1))
    for i, fn in enumerate(X):
        wave = X[0]
        # order = 5
        # fs = 16.0       # sample rate, kHz
        # cutoff = 3.667
        # wave = butter_lowpass_filter(raw, cutoff, fs, order)
        wave = wave / np.max([np.max(abs(wave)), 0.00001])
        wave = wave[:max_len]
        wave = np.pad(wave, (0, max_len-wave.shape[0]), 'minimum')
        S = librosa.feature.melspectrogram(wave, sr=16000, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max)
        x_spec[i,:,:,0] = log_S
    return x_spec


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def save_spec(spec):
    flip = np.flip(spec[0,:,:,0], axis=0)
    rep = np.repeat(flip,10, axis=1)
    imageio.imwrite("static/spec.png", rep)


# Initialize the app
app = flask.Flask(__name__)

@app.before_first_request
def activate_job():
    def run_job():
        env = gym.make("MsPacman-v0")
        img_path_bytes = b"static/pacman.png"
        env.reset()
        env.env.ale.saveScreenPNG(img_path_bytes)
        time.sleep(5)

        done = False
        lives = 4
        new_lives = 3
        action = 0

        while not done:
            if new_lives != lives:
                print("restarting...")
                wait_range = 0
                if action == 0:
                    wait_range = 85
                else:
                    wait_range = 38
                for i in range(wait_range):
                    env.step(0)
                    env.env.ale.saveScreenPNG(img_path_bytes)
                    time.sleep(.1)

            lives = new_lives
            with open("move.txt", "r") as f:
                action = int(f.read())

            _, reward, done, inf = env.step(action)
            new_lives = inf["ale.lives"]
            env.env.ale.saveScreenPNG(img_path_bytes)
            time.sleep(.25)

    thread = threading.Thread(target=run_job)
    thread.start()


@app.route("/")
def viz_page():
    with open("index.html", 'r') as viz_file:
        return viz_file.read()


@app.route("/pacman", methods=["POST"])
def pacman():
    """
    When A POST request with json data is made to this url,
    Process the audio, update and send it back
    """
    global ONE_SECOND
    global debounce_flag
    global debounce_counter
    global prediction_window
    global timer

    data = flask.request.json
    data_array = np.array(list(data["move"].values()))
    resampled = scipy.signal.resample(data_array, 1486)
    if len(ONE_SECOND) < 16000:
        ONE_SECOND.extend(resampled)
        return flask.jsonify({'moved': "waiting for full second"})

    elif debounce_flag == True:
        debounce_counter += 1
        ONE_SECOND = ONE_SECOND[1486:]
        ONE_SECOND.extend(resampled)
        if len(prediction_window) < 8:
            prediction_window.append(11)
        else:
            prediction_window.pop(0)
            prediction_window.append(11)
        if debounce_counter == 7:
            debounce_flag = False
            debounce_counter = 0
        return flask.jsonify({'moved': "debouncing"})

    else:
        ONE_SECOND = ONE_SECOND[1486:]
        ONE_SECOND.extend(resampled)
        spec = process_wav([np.array(ONE_SECOND)])
        # print(timer)
        if timer % 2 == 0:
            save_spec(spec)
        timer += 1
        pred_array = model.predict(spec)
        pred = np.argmax(pred_array)
        # print(pred)
        # print(prediction_window)
        if len(prediction_window) < 8:
            prediction_window.append(pred)
        else:
            prediction_window.pop(0)
            prediction_window.append(pred)
        move = 0
        if prediction_window.count(2) > 6:
            debounce_flag = True
            move = 1
        elif prediction_window.count(3) > 4:
            debounce_flag = True
            move = 4
        elif prediction_window.count(4) > 4:
            debounce_flag = True
            move = 3
        elif prediction_window.count(5) > 4:
            debounce_flag = True
            move = 2
        if move != 0:
            with open("move.txt", "w") as f:
                f.write(str(move))
                return flask.jsonify({'moved': move})
        return flask.jsonify({'moved': "no"})

#--------- RUN WEB APP SERVER ------------#

app.run(port=5000)

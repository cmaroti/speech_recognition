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
import librosa

# import model
model = keras.models.load_model("model_dec2_3CNNs_melpower.h5")

def process_wav(X, max_len = 16000, n_mels = 128):
    """Turns audio into a spectrogram, used on validation data"""
    x_spec = np.zeros((len(X), 128, 32, 1))
    print(x_spec.shape)
    for i, fn in enumerate(X):
        print(i, fn)
        _, wave = scipy.io.wavfile.read(fn)
        wave = wave[:max_len]
        wave = np.pad(wave, (0, max_len-wave.shape[0]), 'minimum')
        S = librosa.feature.melspectrogram(wave, sr=16000, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max)
        x_spec[i,:,:,0] = log_S
    return x_spec

# Initialize the app
app = flask.Flask(__name__)

@app.before_first_request
def activate_job():
    def run_job():
        env = gym.make("MsPacman-v0")
        img_path_bytes = b"static/pacman.png"
        env.reset()
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
            time.sleep(.2)

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
    Read the grid from the json, update and send it back
    """
    data = flask.request.json
    print(data)
    spec = process_wav([data["move"]])
    pred_array = model.predict(spec)
    pred = np.argmax(pred_array)
    move = 0
    if pred == 2:
        move = 1
    elif pred == 3:
        move = 4
    elif pred == 4:
        move = 3
    elif pred == 5:
        move = 2
    if move != 0:
        with open("move.txt", "w") as f:
            f.write(str(move))

    return flask.jsonify({'moved': "yes"})

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port=5000)

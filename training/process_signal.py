import pickle
from scipy import signal
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

base_folder = "/"

signal_base_path = os.path.join(base_folder, "paramsearch")
save_base_path = os.path.join(base_folder, "ps_json")
data_path = os.path.join(base_folder, "json_done")
visualize = False

for exp_folder in tqdm(os.listdir(signal_base_path)):
    if "kmeans" not in exp_folder:
        continue
    signal_path = os.path.join(signal_base_path, exp_folder)
    for filename in os.listdir(signal_path):
        for smoothing_size in [5, 10, 20, 30, 40, 50]:
            for order in [0, 1, 10, 20, 30, 40, 50]:
                exp_folder_save = f"{exp_folder}_{smoothing_size}_{order}"

                def highscore(signal_data):
                    highscore = np.zeros_like(signal_data)
                    for i in range(1, signal_data.size-1):
                        highscore[i] = np.max([get_diff(signal_data, i, -1), get_diff(signal_data, i, 1)])
                    return highscore

                def get_diff(signal_data, starting_index, direction):
                    i = starting_index + direction
                    old_diff = 0
                    diff = signal_data[starting_index]-signal_data[i]
                    while diff > old_diff:
                        i += direction
                        if i < 0 or i == signal_data.size:
                            break
                        old_diff = diff
                        diff = signal_data[starting_index]-signal_data[i]
                    return old_diff

                def change_indices(score, t = None):
                    if t is None:
                        m = np.mean(score)
                        s = np.std(score)
                        t = m+(s/2)
                    return signal.find_peaks(score, height=t)[0]

                def change_indices_am(score, order):
                    if order == 0:
                        return change_indices(score)
                    return signal.argrelmax(score, order=order)[0]

                ## load file
                with open(os.path.join(signal_path, filename), "rb") as f:
                    sig = np.nan_to_num(pickle.load(f), nan=0)



                def split_to_sentences(full_data, verbose = False):
                    """ Splits data into sentences and scenes """

                    text = full_data["text"]
                    sentences = []
                    ends = {}
                    begins = {}
                    pairs = []
                    sentence_beginnings = []
                    for i, pair in enumerate(full_data["sentences"]):
                        _b = pair["begin"]
                        _e = pair["end"]
                        ends[_e] = i
                        begins[_b] = i
                        sentences.append(text[_b:_e])
                        pairs.append([_b, _e])
                        sentence_beginnings.append(pair["begin"])

                    scene_changes = []
                    for pair in full_data["scenes"]:
                        _b = pair["begin"]
                        try:
                            scene_changes.append(begins[_b])
                        except KeyError:
                            for i, sentpair in enumerate(pairs):
                                if sentpair[0] <= _b < sentpair[1]:
                                    scene_changes.append(i)
                                    if verbose:
                                        print("assign", _b, "to", i, "offset:", _b-sentpair[0])
                                    break

                    return sentences, np.asarray(scene_changes), np.asarray(sentence_beginnings)

                with open(os.path.join(data_path, filename[:-7]), "r") as f:
                    data = json.loads(f.read())
                sents, changes, sentence_beginnings = split_to_sentences(data, True)
                ##

                ## process signal
                sig_smooth = signal.convolve(sig, np.ones(smoothing_size)/smoothing_size, mode="same")
                peaks = change_indices_am(sig_smooth, order)
                ##

                ##
                if visualize:
                    plt.plot(sentence_beginnings, sig_smooth)
                    plt.plot(sentence_beginnings[peaks], sig_smooth[peaks], linestyle="None", marker="x")
                    plt.plot(sentence_beginnings[changes], sig_smooth[changes], linestyle="None", marker="o")
                    plt.show()
                ##

                scene_borders = sorted(list(set([0]+list(sentence_beginnings[peaks])+[len(data["text"])])))

                scenes = []
                for i in range(len(scene_borders)-1):
                    sbeg = int(scene_borders[i])
                    send = int(scene_borders[i+1])
                    scenes.append({"begin": sbeg, "end": send, "type": "Scene"})

                data["scenes"] = scenes

                save_dir = os.path.join(save_base_path, exp_folder_save)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_filename = os.path.join(save_dir, filename[:-7])
                with open(save_filename, "w") as f:
                    f.write(json.dumps(data, ensure_ascii=False))

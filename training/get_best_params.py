import os
import numpy as np

base_folder = "/"
paramfolder = os.path.join(base_folder, "ps_results")

def get_f1_from_file(fname):
    with open(fname, "r") as f:
        for l in f:
            if "f1 score over all files" in l:
                return float(l.split("f1 score over all files: ")[1].strip())

scores = []
names = []

for fname in os.listdir(paramfolder):
    names.append(fname)
    fname_full = os.path.join(paramfolder, fname)
    score = get_f1_from_file(fname_full)
    if score is None:
        score = 0
    scores.append(score)

score_indices = np.argsort(scores)

for i in range(1, 21):
    print(names[score_indices[-i]], "-", scores[score_indices[-i]])


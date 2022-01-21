import spacy
import pickle
from tqdm import tqdm
import json
import os
import numpy as np
import random
from scipy import signal
from scipy.spatial.distance import cosine
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

random_seed = 42

MAX_ITER = 500

base_folder = "/"

json_base = os.path.join(base_folder, "json_done")
temp_data_base = os.path.join(base_folder, "temp/")

best_param_candidates = [
    "15_1_500_kmeans_5_0",
    "35_1_500_kmeans_5_10",
    "50_0_500_kmeans_10_0",
    "25_0_1000_kmeans_5_0",
    "25_0_1000_kmeans_10_20",
    "15_1_500_kmeans_20_0",
    "25_1_500_kmeans_5_20",
    "15_0_1000_kmeans_10_0",
    "50_1_500_kmeans_5_20",
    "25_0_1000_kmeans_10_0",
    "25_1_500_kmeans_5_10",
    "50_1_500_kmeans_5_10",
    "50_1_500_kmeans_5_0",
    "25_0_1000_kmeans_10_10",
    "35_1_500_kmeans_5_0"
        ]


def get_params_from_string(params):
    # window_size, num_clusters, smoothing_size, order 
    params_list = [params.split("_")[0]]
    params_list += [params.split("_")[2]]
    params_list += params.split("_")[4:]
    return [int(p) for p in params_list]

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

    return sentences, scene_changes, np.asarray(sentence_beginnings)

def get_vectors(sents):
    print("get vectors")
    nlp = spacy.load("de_core_news_lg")
    nlp.max_length = 10000000
    sents_v =  []
    vectors = []
    for sent in tqdm(sents):
        s_v = []
        s_l = []
        s_p = []
        s_m = []
        processed = nlp(sent)
        for token in processed:
            s_v.append(token.vector)
            vectors.append(token.vector)
        sents_v.append( s_v )
    return sents_v, vectors

class normed_cosine:
    def __init__(self, standard_vector, codebook = None):
        self.codebook = codebook
        self.standard = standard_vector.astype(float)
        print(self.standard.shape)
        self.standard /= np.linalg.norm(self.standard)
        self.num_bins = self.standard.size


    def func(self, a, b):
        if self.codebook is not None:
            _a = tok_to_vec(a, self.codebook)
            _b = tok_to_vec(b, self.codebook)
        else:
            _a = np.bincount(a, minlength=self.num_bins).astype(float)
            _b = np.bincount(b, minlength=self.num_bins).astype(float)
        _a = (_a/np.linalg.norm(_a))-self.standard
        _b = (_b/np.linalg.norm(_b))-self.standard
        return cosine(_a, _b)

def get_vectors_from_sentences(sentences, start, end):
    vs = []
    for v in sentences[start:end]:
        if len(v) > 0:
            vs.append(v)
    if len(vs) > 0:
        return np.concatenate(vs, axis = 0)
    else:
        return []

def double_sliding_window(sentences, window_width, fobj):
    result_signal = []
    num_sentences = len(sentences) 
    assert(2 * window_width < num_sentences)
    for cursor in tqdm(range(window_width, num_sentences-window_width)):
        left = get_vectors_from_sentences(sentences, cursor-window_width, cursor)
        right = get_vectors_from_sentences(sentences, cursor, cursor+window_width)
        result_signal.append(fobj.func(left, right))
    return np.asarray(result_signal)

def double_sliding_window_mirror(sentences, window_width, fobj):
    result_signal = []
    num_sentences = len(sentences) 
    assert(2 * window_width < num_sentences)
    sentences_mod = []
    for i in range(window_width, 0, -1):
        sentences_mod.append(sentences[i])
    sentences_mod += sentences
    for i in range(len(sentences)-2, len(sentences)-(window_width+2), -1):
        sentences_mod.append(sentences[i])
    num_sentences = len(sentences_mod) 

    for cursor in tqdm(range(window_width, num_sentences-window_width)):
        left = get_vectors_from_sentences(sentences_mod, cursor-window_width, cursor)
        right = get_vectors_from_sentences(sentences_mod, cursor, cursor+window_width)
        result_signal.append(fobj.func(left, right))
    return np.asarray(result_signal)

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

def get_scene_dict(sentence_beginnings, peaks, data):
    scene_borders = sorted(list(set([0]+list(sentence_beginnings[peaks])+[len(data["text"])])))

    scenes = []
    for i in range(len(scene_borders)-1):
        sbeg = int(scene_borders[i])
        send = int(scene_borders[i+1])
        scenes.append({"begin": sbeg, "end": send, "type": "Scene"})
    return scenes


def sdict_to_scenes(sdict):
    print(sdict)
    scenes = []
    for s in sdict:
        scenes.append([int(s["begin"]), int(s["end"])])
    return scenes

def iou_index(gt, pred, gt_i):
    gbeg = gt[gt_i][0]
    gend = gt[gt_i][1]

    iou_best = 0
    pred_i = 0
    for i, p in enumerate(pred):
        pbeg = p[0]
        pend = p[1]
        iou = 0
        if pbeg >= gend or gbeg >= pend:
            continue
        if gbeg <= pbeg <= gend <= pend:
            iou = gend-pbeg
        if pbeg <= gbeg <= pend <= gend:
            iou = pend-gbeg
        if pbeg <= gbeg <= gend <= pend:
            iou = gend-gbeg
        if gbeg <= pbeg <= pend <= gbeg:
            iou = pend-pbeg

        if iou > iou_best:
            iou_best = iou
            pred_i = i
    return pred_i, iou_best



def eval_scenes(gt_dict, pred_dict):
    gt = sdict_to_scenes(gt_dict)
    pred = sdict_to_scenes(pred_dict)
    gt_chosen = [False]*len(gt)
    pred_chosen = [False]*len(pred)
    
    iou_full = 0
    for i, g in enumerate(gt):
        pred_i, iou = iou_index(gt, pred, i)
        if pred_chosen[pred_i] == False:
            pred_chosen[pred_i] = True
            iou_full += iou

    return iou_full

def start_log():
    with open("params.log", "w") as f:
        f.write("window_size\tnum_clusters\tsmoothing_size\torder\tiou\n")

def write_log(window_size, num_clusters, smoothing_size, order, iou):
    with open("params.log", "a") as f:
        f.write(f"{window_size}\t{num_clusters}\t{smoothing_size}\t{order}\t{iou}\n")


def get_scene_features(data, sents_v):

    scenes = [] 
    sents_per_scene = []
    scene_types = []
    labels = []
    for jscene in data["scenes"]:
        current_scene = []
        scbeg = int(jscene["begin"])
        scend = int(jscene["end"])
        sctype = 1 if jscene["type"] == "Scene" else 0
        labels.append(sctype)
        for jsent in data["sentences"]:
            sebeg = int(jsent["begin"])
            seend = int(jsent["end"])
            if sebeg >= scbeg and seend <= scend:
                current_scene.append([sebeg, seend])
        scenes.append(current_scene)
        sents_per_scene.append(len(current_scene))

    features = []

    for scene in scenes:
        s_features = []

        # number of sentences
        s_features.append(len(scene))
        lengths = []
        for sent in scene:
            lengths.append(sent[1]-sent[0])

        # average sentence length
        s_features.append(np.mean(lengths))

        # sentence stddev
        s_features.append(np.std(lengths))

        # scene length
        s_features.append(scene[-1][1]-scene[0][0])

        features.append(s_features)

    return np.asarray(features), labels




# get vectors for all files
fdata = []
fsents = []
fvectors = []
fchanges = []
fsents_v = []
fvectors = []
fsentence_beginnings = []
filenames = []


print("get all word vectors")
wvfile = os.path.join(temp_data_base, "f.pickle")
if os.path.exists(wvfile):
    print("file already exists, loading...")
    with open(wvfile, "rb") as f:
        fdata, fsents, fvectors, fchanges, fsents_v, fvectors, fsentence_beginnings, filenames = pickle.load(f)
    print("done!")
else:
    for json_file in tqdm(os.listdir(json_base)):
        if not ".json" in json_file:
            continue
        print(f"working with file {json_file}")
        filenames.append(json_file)
        filename = os.path.join(json_base, json_file)
        with open(filename, "r") as f:
            data = json.loads(f.read())
        fdata.append(data)
        sents, changes, sentence_beginnings = split_to_sentences(data, False)
        fsents.append(sents)
        fchanges.append(changes)
        fsentence_beginnings.append(sentence_beginnings)
        sents_v, vectors = get_vectors(sents)
        fsents_v.append(sents_v)
        fvectors.append(vectors)
    with open(wvfile, "wb") as f:
        pickle.dump([fdata, fsents, fvectors, fchanges, fsents_v, fvectors, fsentence_beginnings, filenames], f)

sents_v = fsents_v[0]
data = fdata[0]

np.random.seed(random_seed)
random.seed(random_seed)


features, labels = get_scene_features(data, sents_v)

for i in range(1, len(fdata)):
    _features, _labels = get_scene_features(fdata[i], fsents_v[i])
    features = np.concatenate([features, _features], axis = 0)
    labels += _labels
print(features.shape)
labels = np.asarray(labels)

lb_pos = labels[labels == 1]
ft_pos = features[labels == 1, :]
print(lb_pos.shape)
print(ft_pos.shape)

lb_neg = labels[labels == 0]
ft_neg = features[labels == 0, :]
print(lb_neg.shape)
print(ft_neg.shape)

for C in [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000]:
    
    clf = make_pipeline(StandardScaler(), SVC(C=C, gamma='auto'))
    scores = cross_val_score(clf, features, labels, cv=10, scoring='f1')
    #print(f"C: {C}, score: {clf.score(features, labels)}, score noscenes: {clf.score(ft_neg, lb_neg)}, score scenes: {clf.score(ft_pos, lb_pos)}")
    print(f"C: {C}, score: {np.mean(scores):.2f}+-{np.std(scores):.2f}")


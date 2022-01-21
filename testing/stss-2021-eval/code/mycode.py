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


## functions

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

## main function

def add_scenes_to_json(data, svc):

    ## values from experiments
    window_size = 25
    num_clusters = 1000
    smoothing_size = 10
    order = 20
    MAX_ITER = 500

    # preprocess json
    sents, changes, sentence_beginnings = split_to_sentences(data, False)
    sents_v, vectors = get_vectors(sents)
    km = MiniBatchKMeans(n_clusters=num_clusters, max_iter=MAX_ITER, verbose=False)
    km = km.fit(vectors)
    labels = km.predict(vectors)

    sents_labels = []
    for vs in tqdm(sents_v):
        sents_labels.append(km.predict(vs))

    fobj = normed_cosine(np.bincount(labels, minlength=num_clusters))

    sig = np.nan_to_num(double_sliding_window_mirror(sents_labels, window_size, fobj), nan=0)
    sig_smooth = signal.convolve(sig, np.ones(smoothing_size)/smoothing_size, mode="same")
    peaks = change_indices_am(sig_smooth, order)

    scenes_pred = get_scene_dict(sentence_beginnings, peaks, data)

    data["scenes"] = scenes_pred

    features, labels =  get_scene_features(data, sents_v)
    
    labels = svc.predict(features)
    for s in range(labels.size):
        data["scenes"][s]["type"] = "Scene" if labels[s] == 1 else "Nonscene"

    return data

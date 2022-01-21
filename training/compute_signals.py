import os
import json
import random
import pickle
import spacy
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.cluster import MiniBatchKMeans

base_folder = "/"

savebase = os.path.join(base_folder, "paramsearch")
database = os.path.join(base_folder, "json_done")

random.seed(42)

window_sizes = [15, 25, 35, 50]
min_ner_nums = [0, 1]
clusters = [500, 1000]
MAX_ITER = 500

configurations = []

for window_size in window_sizes:
    for min_number in min_ner_nums:
        for num_clusters in clusters:
            config = {"window_size": window_size,
                    "min_number": min_number,
                    "num_clusters": num_clusters,
                    }
            configurations.append(config)


def get_save_folder(folder_base, suffix):
    fldr = os.path.join(folder_base, f"{window_size}_{min_number}_{num_clusters}_{suffix}")
    if not os.path.exists(fldr):
        os.mkdir(fldr)
    return fldr

def get_save_filename(folder_base, suffix, filename):
    return os.path.join(get_save_folder(folder_base, suffix), filename)

def save_data(data, filename):
    with open(filename, "w") as f:
        f.write(json.dumps(data))

def save_pickle(data, filename):
    with open(filename+".pickle", "wb") as f:
        pickle.dump(data, f)


for jsonfile in tqdm(os.listdir(database)):
    print(f"working with file {jsonfile}")
    if not ".json" in jsonfile:
        continue

    FILENAME = os.path.join(database, jsonfile)

    with open(FILENAME, "r") as f:
        data = json.loads(f.read())



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

        return sentences, scene_changes, sentence_beginnings

    print("create vectors")
    sents, changes, sentence_beginnings = split_to_sentences(data, False)
    nlp = spacy.load("de_core_news_lg")
    #nlp = spacy.load("de_dep_news_trf")
    nlp.max_length = 10000000
    sents_v =  []
    sents_l = []
    sents_p = []
    sents_m = []
    vectors = []
    persons = {}
    locations = {}
    mixed = {}
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
        sents_l.append( s_l )
        sents_p.append( s_p )
        sents_m.append( s_m )


    # begin experiments on file

    for config in tqdm(configurations):
        print(f"working with configuration {config}")
        window_size = config["window_size"]
        min_number = config["min_number"]
        num_clusters = config["num_clusters"]

        random.seed(42)
        km = MiniBatchKMeans(n_clusters=num_clusters, max_iter=MAX_ITER)
        print("fit kmeans with", num_clusters, "clusters")
        print("train")
        km = km.fit(vectors)
        labels = km.predict(vectors)

        sents_labels = []
        for vs in tqdm(sents_v):
            sents_labels.append(km.predict(vs))


        person_indices = {}
        index = 0
        for pk in persons.keys():
            p = persons[pk]
            if p > min_number:
                person_indices[pk] = index
                index+=1
        num_persons = index
        print(num_persons)

        location_indices = {}
        index = 0
        for lk in locations.keys():
            l = locations[lk]
            if l > min_number:
                location_indices[lk] = index
                index+=1
        num_locations = index
        print(num_locations)

        mixed_indices = {}
        index = 0
        for lk in mixed.keys():
            l = mixed[lk]
            if l > min_number:
                mixed_indices[lk] = index
                index+=1
        num_mixed = index
        print(num_mixed)

        def sents_to_vec(sents, codebook):
            vec = np.zeros(len(codebook.keys()))
            for s in sents:
                for n in s:
                    if n in codebook:
                        vec[codebook[n]] += 1
            return vec

        def tok_to_vec(toks, codebook):
            vec = np.zeros(len(codebook.keys()))
            for t in toks:
                if t in codebook:
                    vec[codebook[t]] += 1.0
            return vec

        mixed_vec = sents_to_vec(sents_m, mixed_indices)
        locations_vec = sents_to_vec(sents_l, location_indices)
        persons_vec = sents_to_vec(sents_p, person_indices)

        class normed_cosine_simple:
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

        class normed_cosine_kmeans:
            def __init__(self, kmeans, num_bins, train_labels):
                self.km = kmeans
                self.standard = np.bincount(train_labels, minlength = num_bins).astype(float)
                self.standard /= np.linalg.norm(self.standard)
                self.num_bins = self.standard.size

            def func(self, a, b):
                _a = self.km.predict(a.astype(float))
                _b = self.km.predict(b.astype(float))
                _a = np.bincount(_a, minlength = self.num_bins).astype(float)
                _b = np.bincount(_b, minlength = self.num_bins).astype(float)
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





        fobj = normed_cosine_simple(np.bincount(labels, minlength=num_clusters))
        num_sentences = len(sents)

        #csignal[window_size:num_sentences-window_size] = double_sliding_window(sents_labels, window_size, fobj)
        csignal = np.nan_to_num(double_sliding_window_mirror(sents_labels, window_size, fobj), nan=0)
        csignal[csignal == None] = 0

        ksig = csignal
        fn = get_save_filename(savebase, "kmeans", jsonfile)
        save_pickle(ksig, fn)


        def combine_sents(sents_a, sents_b):
            sents = []
            for sa, sb in zip(sents_a, sents_b):
                sents.append(sa+sb)
            return sents

        fobj_l = normed_cosine_simple(locations_vec, location_indices)
        num_sentences = len(sents)
        csignal = np.nan_to_num(double_sliding_window_mirror(sents_l, window_size, fobj_l), nan=0)
        csignal[csignal == None] = 0
        lsig = csignal
        fn = get_save_filename(savebase, "locations", jsonfile )
        save_pickle(lsig, fn)

        fobj_p = normed_cosine_simple(persons_vec, person_indices)
        num_sentences = len(sents)
        csignal = np.nan_to_num(double_sliding_window_mirror(sents_p, window_size, fobj_p), nan=0)
        csignal[csignal == None] = 0
        psig = csignal
        fn = get_save_filename(savebase, "persons", jsonfile )
        save_pickle(psig, fn)

        fobj_m = normed_cosine_simple(mixed_vec, mixed_indices)
        num_sentences = len(sents)
        csignal = np.nan_to_num(double_sliding_window_mirror(sents_m, window_size, fobj_m), nan=0)
        csignal[csignal == None] = 0
        msig = csignal

        fn = get_save_filename(savebase, "mixed", jsonfile )
        save_pickle(msig, fn)

        # short pause for the cpu
        time.sleep(3)

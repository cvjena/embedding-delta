from code.mycode import add_scenes_to_json
import os
import json
import spacy
import pickle


data_folder = "data/test"
results_folder = "predictions"

# download the language model
# remove this line if you have no internet access or want to download it by hand
spacy.cli.download("de_core_news_lg")

with open("code/svm.pickle", "rb") as f:
    svc = pickle.load(f)


for fn in os.listdir(data_folder):
    fnf = os.path.join(data_folder, fn)
    print("opening", fnf)
    with open(fnf, "r") as f:
        content = json.load(f)

    content = add_scenes_to_json(content, svc)

    fnf = os.path.join(results_folder, fn)
    print("saving", fnf)
    with open(fnf, "w") as f:
        f.write(json.dumps(content))


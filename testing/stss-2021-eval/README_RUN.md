# Spacy Model German

Um das Programm durchlaufen zu lassen, muss entweder eine Internetverbindung vorhanden sein, oder das spaCy-Modell `de_core_news_lg` vorhanden sein. Zur Zeit wird es automatisch in `prediction.py` mit der Zeile `spacy.cli.download("de_core_news_lg")` heruntergeladen - sollte das nicht möglich sein, muss die Zeile auskommentiert und das Modell von Hand mit `python -m spacy download de_core_news_lg` installiert werden.

To run the programm, you need either an internet connection or the spaCy model `de_core_news_lg`. Currently it is downloaded in `prediction.py` in the line `spacy.cli.download("de_core_news_lg")`. If that is not possible, this line needs to be commented out. and the model needs to be downloaded with the command `python -m spacy download de_core_news_lg`.

# Requirements

* RAM: at worst about 3GB, maybe more for longer files
* CPU cores: The more the better, works with 8, but also works with 1
* GPU: none


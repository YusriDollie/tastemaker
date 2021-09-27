from os import environ, makedirs, listdir
from os.path import exists, join, isfile
from time import localtime, strftime

try:  # _pickle (cPickle) is faster but may(?) not always resolve
    import _pickle as pickle
except:
    import pickle

from .constants import CLF_DIR

if not environ.get("USER"):
    CLIENT_USERNAME = "username"
else:
    CLIENT_USERNAME = environ.get("USER")


def save_classifier(clf, clfName=None):
    """
    Writes a classifier to disk.
    Creates CLF_DIR if it doesn't exist. Generates filename if none is
    specified.
    """

    if not exists(CLF_DIR):
        makedirs(CLF_DIR)

    if not clfName:
        clfName = CLIENT_USERNAME + strftime("%Y-%m-%d@%H%M%S", localtime())

    if not clfName.endswith(".pkl"):
        clfName += ".pkl"

    filename = join(CLF_DIR, clfName)

    pickle.dump(clf, open(filename, "wb"))
    print("Successfully saved classifier: " + filename)


def load_classifier(filename):
    """
    Loads a classifier from disk with the specified filename, assumed to be
    located in the CLF_DIR directory.
    """
    filename = join(CLF_DIR, filename)
    return pickle.load(open(filename, "rb"))


def load_all_classifiers():
    classifiers = []
    for f in listdir(CLF_DIR):
        if isfile(join(CLF_DIR, f)):
            classifiers.append((load_classifier(f), f))
    return classifiers

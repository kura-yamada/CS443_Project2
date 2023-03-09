'''imdb.py
Loads and preprocesses the IMDb dataset
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 2: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import re
import numpy as np
import pandas as pd


def tokenize_words(text):
    '''Transforms a string sentence into words.

    Parameters:
    -----------
    text: string. Sentence of text.

    Returns:
    -----------
    list of strings. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def make_corpus(data, min_sent_size=5):
    '''Make the text corpus.
    Transforms text documents (list of strings) into a list of list of words (both Python lists).
    The format is [[<sentence>], [<sentence>], ...], where <sentence> = [<word>, <word>, ...].

    For the IMDb data, this transforms a list of reviews (each is a single string) into a list of
    sentences, where each sentence is represented as a list of string words. So the elements of the
    resulting list are the i-th sentence overall; we lose information about which review the
    sentence comes from.

    Parameters:
    -----------
    data: list of strings.
    min_sent_size: int. Don't add sentences LESS THAN this number of words to the corpus (skip over them).
        This is important because it removes empty sentences (bad parsing) and those with not enough
        word context.

    Returns:
    -----------
    corpus: list of lists (sentences) of strings (words in each sentence)

    TODO:
    - Split each review into sentences based on periods.
    - Tokenize the sentence into individual word strings (via tokenize_words())
    - Only add a list of words to the corpus if the length is at least `min_sent_size`.
    '''
    pass


def find_unique_words(corpus):
    '''Define the vocabulary in the corpus (unique words). Finds and returns a list of the unique words in the corpus.

    Parameters:
    -----------
    corpus: list of lists (sentences) of strings (words in each sentence).

    Returns:
    -----------
    unique_words: list of unique words in the corpus.
    '''
    pass


def make_word2ind_mapping(vocab):
    '''Create dictionary that looks up a word index (int) by its string.
    Indices for each word are in the range [0, vocab_sz-1].

    Parameters:
    -----------
    vocab: list of strings. Unique words in corpus.

    Returns:
    -----------
    Python dictionary with key,value pairs: string,int
    '''
    pass


def make_ind2word_mapping(vocab):
    '''Create dictionary that uses a word int code to look up its word string
    Indices for each word are in the range [0, vocab_sz-1].

    Parameters:
    -----------
    vocab: list of strings. Unique words in corpus.

    Returns:
    -----------
    Python dictionary with key,value pairs: int,string
    '''
    pass


def make_target_context_word_lists(corpus, word2ind, vocab_sz, context_win_sz=2):
    '''Make the target word array (training data) and context word array ("classes")

    Parameters:
    -----------
    corpus: list of lists (sentences) of strings (words in each sentence).
    word2ind: Dictionary mapping word string -> int code index. Range is [0, vocab_sz-1] inclusive.
    context_win_sz: int. How many words to include before/after the target word in sentences for context.

    Returns:
    -----------
    target_words_int: ndarray. shape=(N,) = (#target_words,)
        Each entry is the i-th int coded target word in corpus.
    context_words_int: ndarray of ndarrays. dtype=object.
        Each entry is an 1D ndarray containg the int codes for the context words associated with the i-th target word.
        outer shape: shape=(N,) = (#target_words,)
        shape(each inner ndarray) = (#context_words,).
        NOTE: #context_words is a variable value (NOT a constant!) in the range [context_win_sz, 2*context_win_sz].
        It is not always the same because of sentence boundary effects. This is why we're using a
        ndarray of ndarrays (not simply one multidimensional ndarray).
        NOTE: The context_words_int array needs to be created with dtype=object. This is because ndarrays need to have
        rectangular shapes — they can't be jagged (e.g. 1st row has length 10, 2nd row has length 11). Setting
        dtype=object allows the array to behave more like a Python list — it will assume you are storing a 1D array of
        objects of any kind (hence the dtype name) and not try to establish a consistent rectangular shape across the axes.
        In this case the "object" we are storing in the ndarray are other ndarray objects (each of which hold ints).

    HINT:
    - Search in a window `context_win_sz` words before after the current target in its sentence.
    Add int code indices of these context words to a ndarray and add this ndarray to the
    `context_words_int` list.
        - Only add context words if they are valid within the window. For example, only int codes of
        words on the right side of the first word of a sentence are added for that target word.


    Example:
    corpus = [['with', 'all', 'this', 'stuff', ...], ...]
    target_words_int  =   array([0, 1, 2, 3, ...])
    context_words_int =   array([array([1, 2]),
                                 array([0, 2, 3]),
                                 array([0, 1, 3, 4]),
                                 array([1, 2, 4, 5]),...])
    '''
    pass


def get_imdb(path2imdb, num_reviews):
    '''Preprocesses the raw IMDb dataset into sets of target and context words.

    This function ties together all the functions you wrote and tested in the notebook.
    For the most part, you should be able to copy-paste.

    Parameters:
    -----------
    path2imdb: str.
        Filename and relative path to the IMDb dataset .csv file.
        e.g. 'data/imdb_train.csv'
    num_reviews: int.
        Number of reviews to extract (starting from the beginning) from the IMDb dataset.

    Returns:
    -----------
    target_words_int: ndarray. shape=(N,) = (#target_words,)
        Each entry is the i-th int coded target word in corpus.
    context_words_int: ndarray of ndarrays. dtype=object.
        Each entry is an 1D ndarray containg the int codes for the context words associated with the i-th target word.
        outer shape: shape=(N,) = (#target_words,)
        shape(each inner ndarray) = (#context_words,).
    unique_words: List of unique words in the corpus.
    word2ind: Dictionary with key,value pairs: string,int mapping word to int-code.

    TODO:
    - Import the dataset from disk
    - Select a subset of the reviews that go into the corpus
    - Make the corpus
    - Identify all the unique words in the corpus
    - Make word <-> int-code lookup table(s)
    - Collect int coded target words, int-coded context words
    '''
    pass

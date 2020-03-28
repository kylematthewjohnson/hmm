import json
import math
import copy

TRAIN_FILE = open("data/twt.train.json", "r", encoding="utf8")
# TEST_FILE = open("data/twt.train.json", "r", encoding="utf8")
TEST_FILE = open("data/twt.dev.json", "r", encoding="utf8")
# TEST_FILE = open("data/twt.test.json", "r", encoding="utf8")
START_SYMBOL = '<s>'
STOP_SYMBOL = '</s>'
N = 0
K = 0.001
L1, L2, L3, L4, L5 = 0.8, 0.1, 0.1, 0.9, 0.1
CORPUS, TEST_DATA, TEST_DATA_COPY = [], [], []
WORD_COUNTS, WORDS, TAGS = {}, [], []
E, UNI, BI, TRI = {}, {}, {}, {}


def process_file(file):

    data = []

    for line in file:
        line_data = json.loads(line)
        data.append(line_data)

    return data


def get_word_counts():

    global WORD_COUNTS
    global WORDS
    global N

    WORD_COUNTS, WORDS = {}, []

    for line in CORPUS:
        for pair in line:
            pair[0] = pair[0].lower()
            if pair[0] in WORD_COUNTS:
                WORD_COUNTS[pair[0]] += 1
                N += 1
            else:
                WORD_COUNTS[pair[0]] = 1
                N += 1

    WORDS = list(WORD_COUNTS.keys())

    return


def unk_corpus():

    global CORPUS

    for line in CORPUS:
        for pair in line:
            if WORD_COUNTS[pair[0]] < 2:
                pair[0] = unk(pair[0])

    return


def unk(word):
    if word.startswith("#"):
        return "<unk_hashtag>"
    elif word.startswith("@"):
        return "<unk_username>"
    elif word.isnumeric():
        return "<unk_number>"
    elif word.startswith("http"):
        return "<unk_website>"
    else:
        return "<unk>"


def unk_test_data():

    global TEST_DATA, TEST_DATA_COPY

    for line in TEST_DATA:
        for pair in line:
            pair[0] = pair[0].lower()
            if pair[0] not in WORD_COUNTS.keys():
                pair[0] = unk(pair[0])

    TEST_DATA_COPY = copy.deepcopy(TEST_DATA)

    return


def get_tags():

    global TAGS

    for line in CORPUS:
        for pair in line:
            if pair[1] not in TAGS:
                TAGS.append(pair[1])

    return


def learn_e():

    global E

    for line in CORPUS:
        for pair in line:
            if (pair[0], pair[1]) in E:
                E[(pair[0], pair[1])] += 1
            else:
                E[(pair[0], pair[1])] = 1

    add_k()

    return


def add_k():

    global E

    for xi in WORDS:
        for yi in TAGS:
            if (xi, yi) in E:
                E[(xi, yi)] += K
            else:
                E[(xi, yi)] = K

    E = normalize(E)

    return


def unigram():

    global UNI

    n = 0

    for line in CORPUS:
        for pair in line:
            if pair[1] in UNI:
                UNI[pair[1]] += 1
                n += 1
            else:
                UNI[pair[1]] = 1
                n += 1

        if STOP_SYMBOL in UNI:
            UNI[STOP_SYMBOL] += 1
            n += 1
        else:
            UNI[STOP_SYMBOL] = 1
            n += 1

    for gram in UNI:
        UNI[gram] = UNI[gram] / n

    return


def bigram():

    global BI

    for line in CORPUS:
        yi_1 = START_SYMBOL
        for yi in line:
            if (yi[1], yi_1) in BI:
                BI[(yi[1], yi_1)] += 1
            else:
                BI[(yi[1], yi_1)] = 1
            yi_1 = yi[1]

        if (STOP_SYMBOL, yi_1) in BI:
            BI[(STOP_SYMBOL, yi_1)] += 1
        else:
            BI[(STOP_SYMBOL, yi_1)] = 1

    BI = normalize(BI)

    return


def trigram():

    global TRI

    yi_2, yi_1 = START_SYMBOL, START_SYMBOL

    for line in CORPUS:
        yi_2 = START_SYMBOL
        yi_1 = START_SYMBOL
        for yi in line:
            if (yi[1], yi_2, yi_1) in TRI:
                TRI[(yi[1], yi_2, yi_1)] += 1
            else:
                TRI[(yi[1], yi_2, yi_1)] = 1
            yi_2 = yi_1
            yi_1 = yi[1]

    if (STOP_SYMBOL, yi_2, yi_1) in TRI:
        TRI[(STOP_SYMBOL, yi_2, yi_1)] += 1
    else:
        TRI[(STOP_SYMBOL, yi_2, yi_1)] = 1

    TRI = normalize(TRI)

    return


def normalize(model):

    n = 0
    for token in model:
        n += model[token]

    for token in model:
        model[token] = model[token] / n

    return model


def linear_bi(yi, yi_1):

    if (yi, yi_1) in BI:
        p_bi = BI[(yi, yi_1)]
    else:
        p_bi = 0

    p_uni = UNI[yi]

    if p_bi == 0:
        return p_uni
    else:
        return L4 * p_bi + L5 * p_uni


def linear_tri(yi, yi_1, yi_2):

    if (yi, yi_2, yi_1) in TRI:
        p_tri = TRI[(yi, yi_2, yi_1)]
    else:
        p_tri = 0

    if (yi, yi_1) in BI:
        p_bi = BI[(yi, yi_1)]
    else:
        p_bi = 0

    p_uni = UNI[yi]

    if p_tri == 0:
        if p_bi == 0:
            return p_uni
        else:
            return L4 * p_bi + L5 * p_uni
    else:
        return L1 * p_tri + L2 * p_bi + L3 * p_uni


def tag_list(i):
    if i in (-1, 0):
        return [START_SYMBOL]
    else:
        return TAGS


def viterbi_bi():

    pi = {}
    bp = {}

    for line in TEST_DATA:
        n = len(line)
        for i in range(1, n + 1):
            for yi in tag_list(i):
                max_score = -float('Inf')
                max_tag = None
                for yi_1 in tag_list(i - 1):
                    e_score = E[(line[i - 1][0], yi)]
                    q_score = linear_bi(yi, yi_1)
                    pi_score = pi.get((i - 1, yi_1), 1)
                    score = math.log(e_score) + math.log(q_score) + pi_score
                    if score > max_score:
                        max_score = score
                        max_tag = yi_1
                pi[(i, yi)] = max_score
                bp[(i, yi)] = max_tag

        max_score = -float('Inf')
        max_tag = None
        for yi in tag_list(n):
            q_score = linear_bi(STOP_SYMBOL, yi)
            pi_score = pi.get((n, yi), 1)
            score = math.log(q_score) + pi_score
            if score > max_score:
                max_score = score
                max_tag = yi

        tags = [max_tag]
        for i, k in enumerate(range(n - 1, 0, -1)):
            tags.append(bp[(k + 1, tags[i])])
        tags.reverse()

        for j in range(n):
            line[j][1] = tags[j]

    return


def viterbi_tri():

    pi = {}
    bp = {}

    for line in TEST_DATA:
        n = len(line)
        for i in range(1, n + 1):
            for yi in tag_list(i):
                for yi_1 in tag_list(i - 1):
                    max_score = -float('Inf')
                    max_tag = None
                    for yi_2 in tag_list(i - 2):
                        e_score = E[(line[i - 1][0], yi)]
                        q_score = linear_tri(yi, yi_1, yi_2)
                        pi_score = pi.get((i - 1, yi_1, yi_2), 1)
                        score = math.log(e_score) + math.log(q_score) + pi_score
                        if score > max_score:
                            max_score = score
                            max_tag = yi_2
                    pi[(i, yi, yi_1)] = max_score
                    bp[(i, yi, yi_1)] = max_tag

        max_score = -float('Inf')
        max_yi_2, max_yi_1 = None, None
        for yi_1 in tag_list(n):
            for yi_2 in tag_list(n - 1):
                q_score = linear_tri(STOP_SYMBOL, yi_1, yi_2)
                pi_score = pi.get((n, yi_1, yi_2), 1)
                score = math.log(q_score) + pi_score
                if score > max_score:
                    max_score = score
                    max_yi_2 = yi_2
                    max_yi_1 = yi_1

        tags = [max_yi_1, max_yi_2]

        for i, k in enumerate(range(n - 2, 0, -1)):
            tags.append(bp[(k + 2, tags[i], tags[i + 1])])
        tags.reverse()

        for j in range(n):
            line[j][1] = tags[j]

    return


def accuracy():
    cnt = 0
    total = 0

    for i in range(len(TEST_DATA)):
        for j in range(len(TEST_DATA[i])):
            if TEST_DATA[i][j][1] == TEST_DATA_COPY[i][j][1]:
                cnt += 1
                total += 1
            else:
                total += 1

    print("Accuracy: " + "{:.2%}".format(cnt / total))

    return


def processing():

    global CORPUS, TEST_DATA

    CORPUS = process_file(TRAIN_FILE)
    TEST_DATA = process_file(TEST_FILE)

    get_word_counts()
    unk_corpus()
    get_word_counts()
    unk_test_data()
    get_tags()

    return


def learning():
    learn_e()
    unigram()
    bigram()
    trigram()
    return


def inference():
    viterbi_bi()
    # viterbi_tri()
    accuracy()
    return


processing()
learning()
inference()

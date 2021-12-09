import numpy as np
import tensorflow as tf

from keras.layers import TextVectorization

import json
from collections import defaultdict
import pickle
import string

numTopAnswers = 1000
maxLen = 25
numDims = 300
imDim = 2048

# read the data files
def readData():
    trainQs = json.load(
        open(
            "/Users/ayushalag/Documents/Documents - Ayush’s MacBook Pro/cos529/project4/questions/v2_OpenEnded_mscoco_train2014_questions.json"
        )
    )["questions"]
    valQs = json.load(
        open(
            "/Users/ayushalag/Documents/Documents - Ayush’s MacBook Pro/cos529/project4/questions/v2_OpenEnded_mscoco_val2014_questions.json"
        )
    )["questions"]
    trainAns = json.load(
        open(
            "/Users/ayushalag/Documents/Documents - Ayush’s MacBook Pro/cos529/project4/annotations/v2_mscoco_train2014_annotations.json"
        )
    )["annotations"]
    valAns = json.load(
        open(
            "/Users/ayushalag/Documents/Documents - Ayush’s MacBook Pro/cos529/project4/annotations/v2_mscoco_val2014_annotations.json"
        )
    )["annotations"]

    return trainQs, valQs, trainAns, valAns


# get the data fields
def loadData(qs, anns, numTopAnswers):
    # load the questions and count freq: map question id to answer
    qidAnn = {}
    topIdx = defaultdict(int)
    for ann in anns:
        qidAnn[ann["question_id"]] = ann
        topIdx[ann["multiple_choice_answer"]] += 1

    # take the top n answers
    sortDict = sorted(topIdx.items(), key=lambda v: v[1], reverse=True)[:numTopAnswers]
    bestAns = set()
    for ans, _ in sortDict:
        bestAns.add(ans)

    # map question id to question if the question's answer is in top n
    idxQs = {}
    for q in qs:
        if qidAnn[q["question_id"]]["multiple_choice_answer"] in bestAns:
            idxQs[q["question_id"]] = q

    # unroll into the question list and answer list (correspondings)
    questData = []
    annData = []
    for qid, q in idxQs.items():
        questData.append(q)
        annData.append(qidAnn[qid])

    return questData, annData


def vectorizeVocab(data):
    vectorizer = TextVectorization(output_sequence_length=maxLen)
    text_ds = tf.data.Dataset.from_tensor_slices(data).batch(32)
    vectorizer.adapt(text_ds)
    vocab = vectorizer.get_vocabulary()
    wordidx = dict(zip(vocab, range(len(vocab))))
    return vocab, wordidx


def getEmbeddings():
    # glovePath = "/Users/ayushalag/embeds/glove.6B.200d.txt"
    glovePath = "/Users/ayushalag/Documents/Documents - Ayush’s MacBook Pro/cos529/project4/embeddings/glove.6B.300d.txt"
    word2vec = {}

    with open(glovePath) as file:
        for embedding in file:
            embedArray = embedding.split()
            word2vec[embedArray[0]] = np.array(embedArray[1:]).astype(float)

    return word2vec


# word2vec is the glove embeddings, vectorizer
# maps each word to a index and has a vocabulary of every word in the training set
def embedMatrix(word2vec, vocab, wordidx):
    numToks = len(vocab) + 2
    embMat = np.zeros((numToks, numDims))
    for word, i in wordidx.items():
        if word in word2vec:
            vector = word2vec[word]
            embMat[i] = vector

    return embMat, numToks


def readPickle(pickleFile):
    return pickle.load(open(pickleFile, "rb"))


def readFeatures():
    trainFeatures = readPickle(
        "/Users/ayushalag/Documents/Documents - Ayush’s MacBook Pro/cos529/project4/features/train.pickle"
    )
    valFeatures = readPickle(
        "/Users/ayushalag/Documents/Documents - Ayush’s MacBook Pro/cos529/project4/features/val.pickle"
    )
    return trainFeatures, valFeatures


def matchImageQuestion(imfeat, quests, wordidx, maxLen):
    imFeatureList = np.zeros((len(quests), imDim))
    questList = np.zeros((len(quests), maxLen))
    questMap = {}
    for idx, question in enumerate(quests):
        quest = (
            question["question"]
            .lower()
            .translate(str.maketrans("", "", string.punctuation))
        )
        words = quest.split()
        for i, word in enumerate(words):
            if word in wordidx:
                questList[idx, i] = wordidx[word]

        imFeatureList[idx] = imfeat[question["image_id"]]
        questMap[questList[idx].tobytes()] = question
    return imFeatureList, questList, questMap


def settleValY(annData, annToIdx):
    vecData = np.empty((len(annData), 1))
    for i, ann in enumerate(annData):
        ans = ann["multiple_choice_answer"]
        if ans in annToIdx:
            vecData[i] = annToIdx[ans]
        else:
            vecData[i] = -1
    return vecData


def settleYData(annData):
    annToIdx = {}
    idxToAnn = {}
    count = 0
    for ann in annData:
        ans = ann["multiple_choice_answer"]
        if ans not in annToIdx:
            annToIdx[ans] = count
            idxToAnn[count] = ans
            count += 1

    vecData = [annToIdx[ann["multiple_choice_answer"]] for ann in annData]

    nb_classes = numTopAnswers
    targets = np.array(vecData).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets, idxToAnn, annToIdx


if __name__ == "__main__":
    print("starting up...")
    trainQs, valQs, trainAns, valAns = readData()
    trainQuest, trainAnnotations = loadData(trainQs, trainAns, numTopAnswers)
    valQuest, valAnnotations = loadData(valQs, valAns, numTopAnswers)
    print("loaded data...")
    questions = []
    for quest in trainQuest:
        questions.append(quest["question"])
    print("getting embeddings...")
    vocab, wordidx = vectorizeVocab(questions)
    word2vec = getEmbeddings()
    embMat, numToks = embedMatrix(word2vec, vocab, wordidx)
    trainF, valF = readFeatures()

    print("matching data and saving")
    # training data
    vecData, idxToAnn, annToIdx = settleYData(trainAnnotations)
    featList, questList, questMapTrain = matchImageQuestion(
        trainF, trainQuest, wordidx, maxLen
    )

    # validation data
    valVecData = settleValY(valAnnotations, annToIdx=annToIdx)
    valFeatList, valQuestList, questMap = matchImageQuestion(
        valF, valQuest, wordidx, maxLen
    )

    with open("questMap.pickle", "wb") as handle:
        pickle.dump(questMap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("idxMap.pickle", "wb") as handle:
        pickle.dump(idxToAnn, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("questMapTr.pickle", "wb") as handle:
        pickle.dump(questMapTrain, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save("emb.npy", embMat)
    np.save("trainIm.npy", featList)
    np.save("trainQs.npy", questList)
    np.save("trainAnns.npy", vecData)

    np.save("valIm.npy", valFeatList)
    np.save("valQuestList.npy", valQuestList)
    np.save("valAnns.npy", valVecData)

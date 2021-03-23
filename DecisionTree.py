import math
import operator


def createDataSet():
    # for outlook label, 0 is sunny, 1 is rainy, 2 is overcast.
    # for temperature label 1 is over 84, 0 is not
    # for humidity label 1 is over 82.5, 0 is not
    dataSet = [[0, 1, 1, 0, 'no'],
               [0, 0, 1, 1, 'no'],
               [2, 0, 1, 0, 'yes'],
               [1, 0, 1, 0, 'yes'],
               [1, 0, 0, 0, 'yes'],
               [1, 0, 0, 1, 'no'],
               [2, 0, 0, 1, 'yes'],
               [0, 0, 1, 0, 'no'],
               [0, 0, 0, 0, 'yes'],
               [1, 0, 0, 0, 'yes'],
               [0, 0, 0, 1, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 0, 0, 0, 'yes'],
               [1, 0, 1, 1, 'no']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels


def calculateEntropy(dataSet):
    num = len(dataSet)
    labelCounts = {}
    for item in dataSet:
        currentLabel = item[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/num
        Entropy -= prob * math.log(prob, 2)
    return Entropy


def splitDataSet(dataSet, i, value):
    transDateSet = []
    for item in dataSet:
        if item[i] == value:
            reducedItem = item[:i]
            reducedItem.extend(item[i+1:])
            transDateSet.append(reducedItem)
    return transDateSet


def GetBestFeature(dataSet):
    featuresNum = len(dataSet[0])-1
    Entropy = calculateEntropy(dataSet)
    bestGain = 0.0
    bestFeature = -1
    for i in range(featuresNum):
        featureList = [example[i] for example in dataSet]
        valueList = set(featureList)
        newEntropy = 0.0
        for value in valueList:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calculateEntropy(subDataSet)
        Gain = Entropy - newEntropy
        if Gain > bestGain:
            bestGain = Gain
            bestFeature = i
    return bestFeature


def majorityContent(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDecisionTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityContent(classList)
    bestFeature = GetBestFeature(dataSet)
    bestFeatureLabel = labels[bestFeature]
    decisionTree = {bestFeatureLabel: {}}
    del(labels[bestFeature])
    Values = [example[bestFeature] for example in dataSet]
    valueList = set(Values)
    for value in valueList:
        subLabels = labels[:]
        decisionTree[bestFeatureLabel][value] = createDecisionTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return decisionTree

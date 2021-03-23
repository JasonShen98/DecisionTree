import DecisionTree

if __name__ == '__main__':
    dataSet, labels = DecisionTree.createDataSet()
    decisionTree = DecisionTree.createDecisionTree(dataSet, labels)
    print('InformationGain\n')
    print(decisionTree)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

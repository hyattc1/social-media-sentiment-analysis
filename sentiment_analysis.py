"""
15-110 Hw6 - Social Media Analytics Project
Name: Connor Hyatt
AndrewID: cjhyatt
"""

import sentiment_tests as test

project = "Social" # don't edit this

### SECTION 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]


'''
parseLabel(label)
Parameters: str
Returns: dict mapping str to str
'''
def parseLabel(label):
    dictinfo = {}
    name = ""
    x = 0
    while "(" not in name:
        name += label[6+x]
        x += 1
    name = name[:len(name)-2]
    dictinfo["name"] = name
    position = ""
    y = 0
    while " " not in position:
        position += label[6 + len(name) + 2 + y]
        y += 1
    position = position[:len(position)-1]
    dictinfo["position"] = position
    state = ""
    z = 0
    while ")" not in state:
        state += label[6 + len(name) + 2 + 6+ len(position) + z]
        z += 1
    state = state[:len(state)-1]
    dictinfo["state"] = state
    return dictinfo


'''
getRegionFromState(stateDf, state)
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    region = ""
    for index, row in stateDf.iterrows():
        if state in row["state"]:
            region = row["region"]
    return region


'''
findHashtags(message)
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    i = 0
    hashtagLst = []
    while i < len(message):
        if message[i] == "#":
            word = ""
            i += 1
            while i < len(message) and message[i] not in endChars:
                word += message[i]
                i += 1
            word = "#" + word
            hashtagLst += [word]
        else:
            i += 1
    return hashtagLst

'''
findSentiment(classifier, message)
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    sen = classifier.polarity_scores(message)['compound']
    if sen > 0.1:
        return "positive"
    elif sen <= 0.1 and sen >= -0.1:
        return "neutral"
    elif sen < -0.1:
        return "negative"


'''
addColumns(data, stateDf)
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names = []
    positions = []
    states = []
    regions = []
    hashtags = []
    sentiments = []
    intensity = SentimentIntensityAnalyzer()
    for i in range(len(data)):
        rowData = parseLabel(data["label"][i])
        rowName = rowData["name"]
        rowPosition = rowData["position"]
        rowState = rowData["state"]
        rowRegion = getRegionFromState(stateDf, rowState)
        names.append(rowName)
        positions.append(rowPosition)
        states.append(rowState)
        regions.append(rowRegion)
    for i in range(len(data)):
        rowInfo = data["text"][i]
        rowHashtags = findHashtags(rowInfo)
        rowSen = findSentiment(intensity, rowInfo)
        hashtags.append(rowHashtags)
        sentiments.append(rowSen)
    data["name"] = names
    data["position"] = positions
    data["state"] = states
    data["region"] = regions
    data["hashtags"] = hashtags
    data["sentiment"] = sentiments
    return


### SECTION 2 ###


'''
getDataCountByState(data, colName, dataToCount)
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    #print(data)
    if colName == "" and dataToCount == "":
        stateCounts = {}
        for state in data["state"]:
            if state in stateCounts:
                stateCounts[state] += 1
            else:
                stateCounts[state] = 1
        return stateCounts
    
    filteredData = data[data[colName] == dataToCount]
    stateCounts = {}
    for state in filteredData["state"]:
        if state in stateCounts:
            stateCounts[state] += 1
        else:
            stateCounts[state] = 1
    return stateCounts


'''
getDataForRegion(data, colName)
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    d = {}
    
    for i in range(len(data)):
        region = data["region"][i]
        state = data[colName][i]
        if region not in d:
            d[region] = {}
        if region in d:
            if state not in d[region]:
                d[region][state] = 1
            else:
                d[region][state] += 1
    #print(d)
    return d


'''
getHashtagRates(data)
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    #print(data)
    d = {}
    for hashtag in data["hashtags"]:
        #print(hashtag)
        if len(hashtag) == 1:
            name = hashtag[0]
            #print(name)
            if name not in d:
                d[name] = 1
            else:
                d[name] += 1
        if len(hashtag) > 1:
            for i in range(len(hashtag)):
                name = hashtag[i]
                if name not in d:
                    d[name] = 1
                else:
                    d[name] += 1
    #print(d)
    return d


'''
mostCommonHashtags(hashtags, count)
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    #print(hashtags)
    biggestNum = -1
    d = {}
    if len(hashtags) == 1:
        return hashtags
    for tag in range(count):
        word = ""
        for tag in hashtags:
            if hashtags[tag] >= biggestNum:
                if tag not in d:
                    word = tag
                    biggestNum = hashtags[tag]
        if word not in d:
            d[word] = biggestNum
        biggestNum = -1
    #print(d)
    return d


'''
getHashtagSentiment(data, hashtag)
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    total = 0
    count = 0
    for i in range(len(data["hashtags"])):
        if hashtag in data["hashtags"][i]:
            if data["sentiment"][i] == "positive":
                total += 1.0
                count += 1
            if data["sentiment"][i] == "neutral":
                total += 0.0
                count += 1
            if data["sentiment"][i] == "negative":
                total += -1
                count += 1
    num = total / count
    #print(total, count, num)
    return num


### SECTION 3 ###

'''
graphStateCounts(stateCounts, title)
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    stateKey = []
    stateValue = []
    for state in stateCounts:
        stateKey.append(state)
        stateValue.append(stateCounts[state])
    plt.title(title)
    plt.bar(stateKey, stateValue)
    plt.xticks(ticks=list(range(len(stateKey))), labels=stateKey, rotation="vertical")
    #plt.xticks(rotation = 90)
    plt.show()
    
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    biggestNum = -1
    d = {}
    for state in range(n):
        word = ""
        for state in stateFeatureCounts:
            if (stateFeatureCounts[state] / stateCounts[state]) >= biggestNum:
                if state not in d:
                    word = state
                    biggestNum = stateFeatureCounts[state] / stateCounts[state]
                    biggestNum = round(biggestNum,3)
        if word not in d:
            d[word] = biggestNum
        biggestNum = -1
    plt.title(title)
    graphStateCounts(d, title)
    plt.show()
    return d


'''
graphRegionComparison(regionDicts, title)
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''

def graphRegionComparison(regionDicts, title):
    
    #print(regionDicts)
    featureNames = []
    regionNames = []
    regionFeatureNames = []
    for region in regionDicts:
        regionNames.append(region)
    for region in regionDicts:
        d = regionDicts[region]
        for featName in regionDicts[region]:
            if featName not in featureNames:
                featureNames.append(featName)
    
    for region in regionDicts:
        lst = []
        for i in range(len(featureNames)):
            if featureNames[i] in regionDicts[region]:
                feat = regionDicts[region][featureNames[i]]
                lst.append(feat)
            else:
                lst.append(0)
        regionFeatureNames.append(lst)
        
    titleOfFunction = title
    sideBySideBarPlots(featureNames, regionNames, regionFeatureNames, titleOfFunction)
    plt.show()
    
    return


'''
graphHashtagSentimentByFrequency(data)
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    hashdict = getHashtagRates(data)
    topDict = mostCommonHashtags(hashdict, 50)
    hashtagLst = []
    freqLst = []
    sentLst = []
    for key in topDict:
        hashtagLst.append(key)
        freqLst.append(topDict[key])
        sent = getHashtagSentiment(data, key)
        sentLst.append(sent)
    
    title = "Hashtag Sentiment By Frequency"
    scatterPlot(freqLst, sentLst, hashtagLst, title)
    plt.show()
    return


#### SECTION 3 CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    #this is the code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    '''
    print("\n" + "#"*15 + " SECTION 1 TESTS " +  "#" * 16 + "\n")
    test.section1Tests()
    print("\n" + "#"*15 + " SECTION 1 OUTPUT " + "#" * 15 + "\n")
    test.runSection1()
    '''

    ## Uncomment these for SECTION 2 ##
    '''print("\n" + "#"*15 + " SECTION 2 TESTS " +  "#" * 16 + "\n")
    test.section2Tests()
    print("\n" + "#"*15 + " SECTION 2 OUTPUT " + "#" * 15 + "\n")
    test.runSection2()'''

    ## Uncomment these for SECTION 3 ##
    print("\n" + "#"*15 + " SECTION 3 OUTPUT " + "#" * 15 + "\n")
    test.runSection3()
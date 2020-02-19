import numpy as np
import pandas as pd
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.data.data_stream import DataStream
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.evaluation import EvaluatePrequential

#holdFilesFolder = "results/Holdout/"
#preqFilesFolder = "results/Prequential/"
#saveFolder = "results/processed_results/"
#fileType = ".csv"
def process(types, streams,  algorithms):
    filesFolder = "results/"
    saveFolder = "results/processed_results/"
    fileType = ".csv"
    folderType = ["Holdout", "Prequential"]
    metrics = [
        "Stream",
        "Clf",
        "Accuracy",
        "Kappa",
        "Kappa_m",
        "Kappa_t",
        "Comp_time"
        ]

    dFramCol = [
        'RSLVQ_SGD',
        'RSLVQ_Adadelta',
        'RSLVQ_RMSprop', 
        'RSLVQ_Adam'
        ]
    resultcolumns=['Data Stream','RSLVQ_SGD','RSLVQ_Adadelta','RSLVQ_RMSprop', 'RSLVQ_Adam']

    rankFix = [
        "Agrawal",
        "Agrawal_drift",
        "Hyperplane",
        "Hyperplane_drift",
        "Sine",
        "Sine_drift",
        "Cess_data",
        "Move_squares",
        "Sea_data",  
        "Electric",
        "Poker",
        "Weather",
        "Rialto"
        ]

    frames = []

    tmpRawSynth = []
    tmpRawReal = []

    tmpSynthFrames = []
    tmpRealFrames = []

    #first loop for the eval types
    for i in range(len(types)):
        #second loop to get seperate synth. and real sets.
        for n in range(len(streams)):
            #third loop to iterate through the metrics
            for k in range(2,len(metrics)):
            #forth loop to iterate through the streams
                for j in range(len(streams[n])):
                    test0 = pd.read_csv(filesFolder+folderType[i]+'/'+types[i]+streams[n][j]+algorithms[0]+fileType)
                    test1 = pd.read_csv(filesFolder+folderType[i]+'/'+types[i]+streams[n][j]+algorithms[1]+fileType)
                    test2 = pd.read_csv(filesFolder+folderType[i]+'/'+types[i]+streams[n][j]+algorithms[2]+fileType)
                    test3 = pd.read_csv(filesFolder+folderType[i]+'/'+types[i]+streams[n][j]+algorithms[3]+fileType)

                    #convert it to an array (should somehow be possible to get single items with the dataframes, just can not figure it out)
                    data_sgd = test0.to_numpy()
                    data_adadelta = test1.to_numpy()
                    data_rms = test2.to_numpy()
                    data_adam = test3.to_numpy()

                    if k != 6:
                        dataframe =  [[data_sgd[0][0],round(float(data_sgd[0][k]),4),round(float(data_adadelta[0][k]),4),round(float(data_rms[0][k]),4),round(float(data_adam[0][k]),4)]]
                        rawFrame = pd.DataFrame(dataframe,columns=resultcolumns) 
                        frames.append(rawFrame)
                    else:
                        dataframe =  [[data_sgd[0][0],round(float(data_sgd[0][k]),2),round(float(data_adadelta[0][k]),2),round(float(data_rms[0][k]),2),round(float(data_adam[0][k]),2)]]
                        rawFrame = pd.DataFrame(dataframe,columns=resultcolumns) 
                        frames.append(rawFrame)

                #end fourth loop
                mainFrame = frames[0]
                for l in range(1,len(frames)):
                    #put all the frames into a single one
                    mainFrame = mainFrame.append(frames[l], ignore_index = True)

                #replace the 0.0 value from the Sine_sgd stream with NaN
                mainFrame = mainFrame.replace(0.0, np.nan)
                if len(streams[n]) == 9:
                    tmpRawSynth.append(mainFrame)
                else:
                    tmpRawReal.append(mainFrame)

                #next calculate for each table the columns averages
                meanValueRow = []
                if len(streams[n])==9:
                    meanValue = ['Synth. Mean']
                else:
                    meanValue = ['Real Mean']

                for m in range(len(dFramCol)):
                    meanValue.append(round(mainFrame[dFramCol[m]].mean(),4))
                meanValueRow.append(meanValue)
           
                #create a df so tha it can be addded to the mainframe
                meanframe = pd.DataFrame(meanValueRow,columns=resultcolumns)
                mainFrame = mainFrame.append(meanframe, ignore_index=True)

                #ranked the mean values from 1-4 and add it to the table
                #print(mainFrame)
                if len(streams[n]) == 9:
                    if k != 6:
                        rankedFrame = mainFrame.rank(1,na_option='bottom',ascending=False)
                        mainFrame = mainFrame.append(rankedFrame.iloc[9],ignore_index=True)
                    else:
                        rankedFrame = mainFrame.rank(1,na_option='bottom')
                        mainFrame = mainFrame.append(rankedFrame.iloc[9],ignore_index=True)

                    resultpath = saveFolder+types[i]+metrics[k]+"_ranked"+fileType
                    #rankedFrame.to_csv(resultpath, index = None, header=True)
                else:
                    if k != 6:
                        rankedFrame = mainFrame.rank(1,na_option='bottom',ascending=False)
                        mainFrame = mainFrame.append(rankedFrame.iloc[4],ignore_index=True)
                    else:
                        rankedFrame = mainFrame.rank(1,na_option='bottom')
                        mainFrame = mainFrame.append(rankedFrame.iloc[4],ignore_index=True)

                    resultpath = saveFolder+types[i]+metrics[k]+"_ranked"+fileType
                    #rankedFrame.to_csv(resultpath, index = None, header=True)

                #replace NaN with Synth. Rank
                if len(streams[n])==9:
                    mainFrame.loc[10, 'Data Stream'] = 'Synth. Rank'
                    tmpSynthFrames.append(mainFrame)   
                else:
                    mainFrame.loc[5, 'Data Stream'] = 'Real Rank'
                    tmpRealFrames.append(mainFrame)
                    
                #save this frame and the rank table
                
                #mainFrame.to_csv(resultpath, index = None, header=True)
                mainFrame = []
                frames = []

            print("")
            print(folderType[i] + "finished.")
            print("")
            #end third loop

    j = 0
    for i in range(len(types)):
        for k in range(2,len(metrics)):
            resultpath = saveFolder+types[i]+metrics[k]+fileType
            endResultFile = tmpSynthFrames[j]
            endResultFile = endResultFile.append(tmpRealFrames[j],ignore_index=True)

            overallStats = tmpRawSynth[j]
            overallStats = overallStats.append(tmpRawReal[j],ignore_index=True)

            meanValueRow = []
            meanValue = ['Overall Mean']
            for m in range(len(dFramCol)):
                meanValue.append(round(overallStats[dFramCol[m]].mean(),4))
            meanValueRow.append(meanValue)
            meanframe = pd.DataFrame(meanValueRow,columns=resultcolumns)
            endResultFile = endResultFile.append(meanframe, ignore_index=True)

            if k != 6:
                overallRank = meanframe.rank(1,na_option='bottom',ascending=False)
                endResultFile = endResultFile.append(overallRank.iloc[0],ignore_index=True)

                overallRank = overallStats.rank(1,na_option='bottom',ascending=False)
                rankPath = saveFolder+types[i]+metrics[k]+"_ranked"+fileType
                overallRank.insert(0,'Data Streams',rankFix)
                overallRank.to_csv(rankPath, index = None, header=True)
                print('')
                print(overallRank)
            else:
                overallRank = meanframe.rank(1,na_option='bottom')
                endResultFile = endResultFile.append(overallRank.iloc[0],ignore_index=True)

                overallRank = overallStats.rank(1,na_option='bottom')
                rankPath = saveFolder+types[i]+metrics[k]+"_ranked"+fileType
                overallRank.insert(0,'Data Streams',rankFix)
                overallRank.to_csv(rankPath, index = None, header=True)
                print('')
                print(overallRank)

            endResultFile.loc[18, 'Data Stream'] = 'Overall Rank'
            endResultFile.to_csv(resultpath, index = None, header=True)
    
            j += 1

evaltypes = ["Holdout_","Prequential_"]

streams = [
    [
        "Agrawal",
        "Agrawal_drift",
        "Hyperplane",
        "Hyperplane_drift",
        "Sine",
        "Sine_drift",
        "Cess_data",
        "Move_squares",
        "Sea_data",
    ],
    [
        "Electric",
        "Poker",
        "Weather",
        "Rialto"
    ]]

algorithms = [
    "_RSLVQ_SGD",
    "_RSLVQ_Adadelta",
    "_RSLVQ_RMSprop",
    "_RSLVQ_Adam"
    ]

process(evaltypes, streams, algorithms)
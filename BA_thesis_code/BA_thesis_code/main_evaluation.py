import numpy
import pandas as pd
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
from Algorithms.rslvq_sgd import RSLVQSgd
from Algorithms.rslvq_adadelta import RSLVQAdadelta
from Algorithms.rslvq_rmsprop import RSLVQRMSprop
from Algorithms.rslvq_adam import RSLVQAdam
from skmultiflow.data.agrawal_generator import AGRAWALGenerator 
from skmultiflow.data.concept_drift_stream import ConceptDriftStream 
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator 
from skmultiflow.data.sine_generator import SineGenerator
from skmultiflow.data.data_stream import DataStream
from skmultiflow.data.file_stream import FileStream
from custom_evaluation import custom_evaluation


def main():
    usedSynthData = [
        ["synthData/cess_data.csv","synthData/cess_targets.csv"],
        ["synthData/move_square_data.csv","synthData/move_square_targets.csv"],
        ["synthData/sea_data.csv", "synthData/sea_targets.csv"]
        ]

    #Name of the datastreams
    synthDataStreams_names = [
        "Cess_data",
        "Move_squares",
        "Sea_data",
        ]

    realDataFiles = [
        ["realData/electric_data.csv","realData/electric_targets.csv"],
        ["realData/poker_data.csv","realData/poker_targets.csv"],
        ["realData/weather_data.csv","realData/weather_targets.csv"],
        ["realData/rialto_data.csv","realData/rialto_targets.csv"]
        ]

    #Name of the datastreams
    realDataStreams_names = [
        "Electric",
        "Poker",
        "Weather",
        "Rialto"
        ]

    #fixe the poker dataset
    #dfX=pd.read_csv("realData/poker_data_broken.csv")
    #dfY=pd.read_csv(realTargetFiles[1])
    #print(dfX.dtypes)

    #remove the false columns
    #dfX = dfX.drop(columns = ['feat_11', 'feat_12'])
    #print(dfX.dtypes)

    #save fixed data as csv
    #dfX.to_csv(r'realData/poker_data.csv', index = None, header=True)

    #check if saved correctly
    #X=pd.read_csv(realDataFiles[1])
    #print(X.dtypes)

    #fix electirc dataset
    #dfX=pd.read_csv("realData/electric_data_broken.csv")
    #print(dfX.dtypes)

    #remove the false columns
    #dfX = dfX.drop(columns = ['feat_1', 'feat_2'])
    #print(dfX.dtypes)
    #dfX.to_csv(r'realData/electric_data.csv', index = None, header=True)

    #check if saved correctly
    #X=pd.read_csv(realDataFiles[0])
    #print(X.dtypes)

    #Stream with synth generated data from generators, synth data stream that were used in other works and real data streams
    synthDataStreams = [
        [AGRAWALGenerator(random_state=112, perturbation=0.1),"Agrawal"],
        [ConceptDriftStream(stream = AGRAWALGenerator(random_state=112),
                           drift_stream = AGRAWALGenerator(random_state=112,perturbation=0.1),
                           position = 40000,
                           width = 10000),"Agrawal_drift"],
        [HyperplaneGenerator(mag_change=0.001, noise_percentage=0.1),"Hyperplane"],
        [ConceptDriftStream(stream = HyperplaneGenerator(),
                           drift_stream = HyperplaneGenerator(),
                           position = 40000,
                           width = 10000),"Hyperplane_drift"],
        [SineGenerator(random_state=112),"Sine"],
        [ConceptDriftStream(stream = SineGenerator(random_state=112),
                           drift_stream = SineGenerator(random_state=112),
                           position = 40000,
                           width = 10000),"Sine_drift"]
        ]

    synthDataStreamsUsed = []
    for i in range(len(usedSynthData)):
        synthDataStreamsUsed.append([DataStream(pd.read_csv(usedSynthData[i][0]),pd.read_csv(usedSynthData[i][1])),synthDataStreams_names[i]])

    realDataStreams = []
    for i in range(len(realDataFiles)):
        realDataStreams.append([DataStream(pd.read_csv(realDataFiles[i][0]),pd.read_csv(realDataFiles[i][1])),realDataStreams_names[i]])

    clfs = [
        [RSLVQSgd(),'RSLVQ_SGD'],
        [RSLVQAdadelta(),'RSLVQ_Adadelta'],
        [RSLVQRMSprop(),'RSLVQ_RMSprop'],
        [RSLVQAdam(),'RSLVQ_Adam']
        ]
   
    max_items = 40000

    #insert the dataset array that should be evaluated, if the reform exception occurs, set the dataset 
    #that is effected by it as the first one in the array and run again
    for i in range(len(synthDataStreams)):
        for j in range(len(clfs)):
            #print('bla')
            custom_evaluation(synthDataStreams[i], clfs[j], max_items, False)
            custom_evaluation(synthDataStreams[i], clfs[j], max_items, True)


main()



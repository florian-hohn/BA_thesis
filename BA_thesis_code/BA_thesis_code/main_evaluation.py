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
        "synthData/cess_data.csv",
        "synthData/hyper_data.csv",
        "synthData/mixeddrift_data.csv",
        "synthData/move_square_data.csv",
        "synthData/sea_data.csv"
        ]

    usedSynthTargets = [
        "synthData/cess_targets.csv",
        "synthData/hyper_targets.csv",
        "synthData/mixeddrift_targets.csv",
        "synthData/move_square_targets.csv",
        "synthData/sea_targets.csv"
        ]


    realDataFiles = ["realData/electric_data.csv",
            #"realData/poker_data.csv",
            "realData/weather_data.csv",]
            #"realData/rialto_data.csv"]

    realTargetFiles = ["realData/electric_targets.csv",
            #"realData/poker_targets.csv",
            "realData/weather_targets.csv",]
            #"realData/rialto_targets.csv"]

    #Stream with synth generated data from generators, synth data stream that were used in other works and real data streams
    synthDataStreams = [
        AGRAWALGenerator(random_state=112),
        ConceptDriftStream(stream = AGRAWALGenerator(random_state=112),
                           drift_stream = AGRAWALGenerator(random_state=112),
                           position = 40000,
                           width = 10000),
        HyperplaneGenerator(mag_change=0.001, noise_percentage=0.1),
        ConceptDriftStream(stream = HyperplaneGenerator(),
                           drift_stream = HyperplaneGenerator(),
                          position = 40000,
                          width = 10000),
        ConceptDriftStream(stream = HyperplaneGenerator(),
                           drift_stream = HyperplaneGenerator(),
                          alpha = 90,
                          position = 40000,
                          width = 10000),
        SineGenerator(random_state=112),
        ConceptDriftStream(stream = SineGenerator(random_state=112),
                           drift_stream = SineGenerator(random_state=112),
                          position = 40000,
                          width = 10000)]

    synthDataStreamsNotGenerated = []
    for i in range(len(usedSynthData)):
        synthDataStreamsNotGenerated.append(DataStream(pd.read_csv(usedSynthData[i]),pd.read_csv(usedSynthTargets[i])))

    realDataStreams = []
    for i in range(len(realDataFiles)):
        realDataStreams.append(DataStream(pd.read_csv(realDataFiles[i]),pd.read_csv(realTargetFiles[i])))

    pokerStream = DataStream(pd.read_csv("realData/poker_data.csv"),pd.read_csv("realData/poker_targets.csv"))

    #X=pd.read_csv(realDataFiles[0])
    #Y=pd.read_csv(realTargetFiles[0])

    #print(X.dtypes)
    #print(Y.dtypes)

    #Name of the datastreams
    synthDataStreams_names = ["agrawal",
                        "agrawal_drift",
                        "hyperplane",
                        "hyperplane_drift",
                        "hyperplane_drift_90",
                        "sine",
                        "sine_drift"]
    synthDataStreamsNotGenerated_names = ["cess_data",
                        "hyper_data",
                        "mixeddrift_data",
                        "move_squares",
                        "sea_data"]
    realDataStreams_names= ["electric",
                        #"poker",
                        "weather",]
                        #"rialto"]
    realPoker_names = ["poker"]

    clfs = [#RSLVQSgd(),
            RSLVQAdadelta(),
            RSLVQRMSprop(),
            RSLVQAdam()]

    clfs_names = [#'RSLVQ_SGD', 
                      'RSLVQ_Adadelta', 
                      'RSLVQ_RMSprop',
                      'RSLVQ_Adam']

    max_items = 40000



    for i in range(len(realPoker_names)):
        for j in range(len(clfs)):
            custom_evaluation(pokerStream, realPoker_names[0], clfs[j], clfs_names[j], max_items)
            #custom_evaluation(realDataStreams[i], realDataStreams_names[i], clfs[j], clfs_names[j], max_items)


main()



import numpy
import pandas as pd
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
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
            "realData/poker_data.csv",
            "realData/weather_data.csv",
            "realData/rialto_data.csv"]

    realTargetFiles = ["realData/electric_targets.csv",
            "realData/poker_targets.csv",
            "realData/weather_targets.csv",
            "realData/rialto_targets.csv"]

    realDataComp = "realData/forestcover.csv"

    #Stream with synth generated data from generators, synth data stream that were used in other works and real data streams
    datastreams = [
        AGRAWALGenerator(random_state=112),
        ConceptDriftStream(stream = AGRAWALGenerator(random_state=112),
                           drift_stream = AGRAWALGenerator(random_state=112),
                           position = 100000,
                           width = 5000),
        HyperplaneGenerator(random_state=112,
                            n_drift_features=3),
        ConceptDriftStream(stream = HyperplaneGenerator(random_state=112,
                                                        n_drift_features=3,),
                           drift_stream = HyperplaneGenerator(random_state=112,
                                                        n_drift_features=3,),
                          position = 100000,
                          width = 5000),
        ConceptDriftStream(stream = HyperplaneGenerator(random_state=112,
                                                        n_drift_features=3,),
                           drift_stream = HyperplaneGenerator(random_state=112,
                                                        n_drift_features=3,),
                          alpha = 90,
                          position = 100000,
                          width = 5000),
        SineGenerator(random_state=112,has_noise=True),
        ConceptDriftStream(stream = SineGenerator(random_state=112,
                                                  has_noise=True),
                           drift_stream = SineGenerator(random_state=112,
                                                        has_noise=True),
                          position = 100000,
                          width = 5000),
        DataStream(pd.read_csv(usedSynthData[0]),pd.read_csv(usedSynthTargets[0])),
        DataStream(pd.read_csv(usedSynthData[1]),pd.read_csv(usedSynthTargets[1])),
        DataStream(pd.read_csv(usedSynthData[2]),pd.read_csv(usedSynthTargets[2])),
        DataStream(pd.read_csv(usedSynthData[3]),pd.read_csv(usedSynthTargets[3])),
        DataStream(pd.read_csv(usedSynthData[4]),pd.read_csv(usedSynthTargets[4])),
        DataStream(pd.read_csv(realDataFiles[0]),pd.read_csv(realTargetFiles[0])),
        DataStream(pd.read_csv(realDataFiles[1]),pd.read_csv(realTargetFiles[1])),
        DataStream(pd.read_csv(realDataFiles[2]),pd.read_csv(realTargetFiles[2])),
        DataStream(pd.read_csv(realDataFiles[3]),pd.read_csv(realTargetFiles[3])),
        ]

    #Name of the datastreams
    datastream_names = ["agrawal",
                        "agrawal_drift",
                        "hyperplane",
                        "hyperplane_drift",
                        "hyperplane_drift_90",
                        "sine",
                        "sine_drift",
                        "cess_data",
                        "hyper_data",
                        "mixeddrift_data",
                        "move_squares",
                        "sea_data",
                        "electric",
                        "poker",
                        "weather",
                        "rialto"]

    #varients for the rslvq that should be used
    rslvq = ["sgd","adadelta","rmsprop","adam"]

    #evaluation types
    eval = ["holdout","prequantial"]

    #evaluation metrics
    metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m', 'running_time']

    max_items = 500000

    custom_evaluation(datastreams, datastream_names, rslvq,eval, metrics, max_items)


main()



import numpy
import pandas as pd
from Algorithms.rslvq_all import RSLVQall
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.data.data_stream import DataStream
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.evaluation import EvaluatePrequential

filePath = "realData/"
fileType = ".csv"

realDataFiles = ["electric_data",
            "outdoor_data",
            "poker_data",
            "weather_data"]

realTargetFiles = ["electric_targets",
            "outdoor_targets",
            "poker_targets",
            "weather_targets"]

resultHoldoutFiles = ["electric_hold_result",
            "outdoor_hold_result",
            "poker_hold_result",
            "weather_hold_result"]

resultPreqFiles = ["electric_preq_result",
            "outdoor_preq_result",
            "poker_preq_result",
            "weather_preq_result"]

evalHoldoutPara = [[800,8000],[200,600],[10000,100000],[1000,2000]]

evalPrequPara = [[400,8000],[100,600],[2000,100000],[100,2000]]

rslvq=["sgd","adadelta","rmsprop","rmspropada"]

eval = ["holdout","prequantial"]

ev = 0

#Make a loop to go through each of the data sets to  automate the whole thing
#1.Loop to change the Testdata
for i in range(len(realDataFiles)):
    #Workflow from scikit_multiflow framework
    # 1. Load the data set as a stream
    X = pd.read_csv(filePath+realDataFiles[i]+fileType)
    Y = pd.read_csv(filePath+realTargetFiles[i]+fileType)

    X = X.to_numpy()
    Y = Y.to_numpy()

    stream = DataStream(X,Y)
    stream.prepare_for_use()

    #Second loop to change the classifier that is used on the dataset
    for j in range(len(rslvq)): 
        # 2. load the classifier that you want to use
        clf = RSLVQall(gradient_descent=rslvq[j])

        #Third loop to change the evaluator
        for k in range(len(eval)):
            # 3. Setup the evaluator
            if eval[k] == "holdout":
                evaluator_rslvq = EvaluateHoldout(show_plot = True,
                                                n_wait = evalHoldoutPara[i][0],
                                                max_samples = evalHoldoutPara[i][1],
                                                metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m'],
                                                output_file = filePath+resultHoldoutFiles[i]+"_"+rslvq[j]+fileType)
            else:
                evaluator_rslvq = EvaluatePrequential(show_plot = True,
                                                    n_wait = evalPrequPara[i][0],
                                                    max_samples = evalPrequPara[i][1],
                                                    pretrain_size = evalPrequPara[i][0],
                                                    metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m'],
                                                    output_file = filePath+resultPreqFiles[i]+"_"+rslvq[j]+fileType)
            # 4. Run evaluation
            stream.target_values
            evaluator_rslvq.evaluate(stream=stream, model=clf)

            print("Evaluation from Dataset "+realDataFiles[i]+" with clf " +rslvq[j]+ " and eval. Para. " +eval[k]+ " complete.")

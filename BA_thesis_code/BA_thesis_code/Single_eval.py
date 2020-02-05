import numpy
import pandas as pd
from Algorithms.RSLVQall import RSLVQall
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

# 1. Load the data set as a stream

X = pd.read_csv(filePath+realDataFiles[2]+fileType)
Y = pd.read_csv(filePath+realTargetFiles[2]+fileType)

print(X)
print(Y)

X = X.astype(int)
Y = Y.astype(int)

print(X)
print(Y)

#merged = xint.merge(yint, on='h')

#print(merged)

X = X.to_numpy()
Y = Y.to_numpy()

stream = DataStream(X,Y)
stream.prepare_for_use()

# 2. load the classifier that you want to use
clf = RSLVQall(gradient_descent = rslvq[0])

# 3. Setup the evaluator
if eval[0] == "holdout":
    evaluator_rslvq = EvaluateHoldout(show_plot = True,
                                      n_wait = evalHoldoutPara[2][0],
                                      max_samples = evalHoldoutPara[2][1],
                                      metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m'],
                                      output_file = filePath+resultHoldoutFiles[2]+"_"+rslvq[2]+fileType)
else:
    evaluator_rslvq = EvaluatePrequential(show_plot = True,
                                      n_wait = evalPrequPara[2][0],
                                      max_samples = evalPrequPara[2][1],
                                      pretrain_size = evalPrequPara[2][0],
                                      metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m'],
                                      output_file = filePath+resultPreqFiles[2]+"_"+rslvq[2]+fileType)

# 4. Run evaluation
evaluator_rslvq.evaluate(stream=stream, model=clf)



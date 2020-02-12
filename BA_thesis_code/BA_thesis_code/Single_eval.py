import numpy
import pandas as pd
from Algorithms.rslvq_all import RSLVQall
from Algorithms.rslvq_sgd import RSLVQSgd
from Algorithms.rslvq_adadelta import RSLVQAdadelta
from Algorithms.rslvq_rmsprop import RSLVQRMSprop
from Algorithms.rslvq_adam import RSLVQAdam
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.data.data_stream import DataStream
from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.evaluation import EvaluatePrequential

filePath = "realData/"
fileType = ".csv"

m_data = 'elec'

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

rslvq=["sgd","adadelta","rmsprop","adam"]

eval = ["holdout","prequantial"]

# 1. Load the data set as a stream

X = pd.read_csv(filePath+realDataFiles[0]+fileType)
Y = pd.read_csv(filePath+realTargetFiles[0]+fileType)

print(X.dtypes)
print(Y.dtypes)

X = X.drop(['a','b'], axis = 1)

print(X.dtypes)

#merged = xint.merge(yint, on='h')

#print(merged)

X = X.to_numpy()
Y = Y.to_numpy()

stream = DataStream(X,Y)
#stream = FileStream(filePath+m_data+fileType)
stream.prepare_for_use()

# 2. load the classifier that you want to use
clf = RSLVQall(gradient_descent = rslvq[3])

# 3. Setup the evaluator
#if eval[0] == "holdout":
evaluator_rslvq = EvaluateHoldout(show_plot = True,
                                      n_wait = 1000,
                                      max_samples = 10000,
                                      metrics =['accuracy','kappa', 'kappa_t', 'kappa_m'],
                                      output_file = filePath+resultHoldoutFiles[0]+"_"+rslvq[0]+fileType)
#else:
   # evaluator_rslvq = EvaluatePrequential(show_plot = True,
    #                                  n_wait = evalPrequPara[0][0],
     #                                 max_samples = evalPrequPara[0][1],
      #                                pretrain_size = evalPrequPara[0][0],
       #                               metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m'],
        #                              output_file = filePath+resultPreqFiles[0]+"_"+rslvq[0]+fileType)

# 4. Run evaluation
print(stream.target_values)
try:
    evaluator_rslvq.evaluate(stream=stream, model=clf)
except Exception as e:
    print(e)



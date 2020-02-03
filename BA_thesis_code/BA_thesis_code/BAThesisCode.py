import numpy
import pandas as pd
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.data.data_stream import DataStream
from skmultiflow.data import WaveformGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential

#Workflow from scikit_multiflow framework
# 1. Load the data set as a stream
#dfn = [["realData/electric_data.csv", " "], # 7 attributes
 #      ["realData/outdoor_data.csv", " "],  # 5 attributes
  #     ["realData/poker_data.csv", " "],    # 10 attributes
   #    ["realData/weather_data.csv", ","]]  # 8 attributes

#df = pd.read_csv(dfn[0][0],dfn[0][1])
#df = df.drop(df.columns[0], axis=1)



#dataStream = DataStream(df)
#dataStream.prepare_for_use()
stream = WaveformGenerator()
stream.prepare_for_use()

#1.1 retrieve the first sample 
#print("Data: ")
#print(dataStream.next_sample(5))

# 2. load the classifier that you want to use
#clf_base_rslvq = RSLVQ()
ht = HoeffdingTree()
# 3. Setup the evaluator
evaluator_rslvq = EvaluatePrequential(show_plot=True,
                                pretrain_size=200,
                                max_samples=20000)

# 4. Run evaluation
evaluator_rslvq.evaluate(stream=stream, model=ht)

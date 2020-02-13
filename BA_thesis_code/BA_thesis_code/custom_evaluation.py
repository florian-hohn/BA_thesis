import numpy
import pandas as pd
from Algorithms.rslvq_sgd import RSLVQSgd
from Algorithms.rslvq_adadelta import RSLVQAdadelta
from Algorithms.rslvq_rmsprop import RSLVQRMSprop
from Algorithms.rslvq_adam import RSLVQAdam
from Algorithms.rslvq_all import RSLVQall
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.metrics import ClassificationMeasurements

def custom_evaluation(datastreams, datastream_names, rslvq_varients, evaluator, metrics, stream_length):

        clfs_names = ['RSLVQ_SGD', 
                      'RSLVQ_Adadelta', 
                      'RSLVQ_RMSprop',
                      'RSLVQ_Adam']

        rslvq_sgd = RSLVQall(gradient_descent=rslvq_varients[0])
        rslvq_adadelta = RSLVQall(gradient_descent=rslvq_varients[1])
        rslvq_rmsprop = RSLVQall(gradient_descent=rslvq_varients[2])
        rslvq_adam = RSLVQall(gradient_descent=rslvq_varients[3])

        stream = datastreams[0]
        stream.prepare_for_use()

        rslvq_sgd = RSLVQall(gradient_descent=rslvq_varients[0])

        evaluator = EvaluateHoldout(n_wait = 10000,
                                                    max_samples = stream_length,
                                                    metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m', 'running_time'],
                                                    output_file = 'results/holdout_evaluation_test.csv')
        evaluator.evaluate(stream=stream, model=[rslvq_sgd, 
                                                 rslvq_adadelta, 
                                                 rslvq_rmsprop, 
                                                 rslvq_adam]) 
        print('Holdout evaluation test finished')

        for i in range(len(datastreams)):
            stream = datastreams[i]
            stream.prepare_for_use()

            rslvq_sgd = RSLVQall(gradient_descent=rslvq_varients[0])
            rslvq_adadelta = RSLVQall(gradient_descent=rslvq_varients[1])
            rslvq_rmsprop = RSLVQall(gradient_descent=rslvq_varients[2])
            rslvq_adam = RSLVQall(gradient_descent=rslvq_varients[3])

            l = 0
            for l in range(len(evaluator)):
                j=0
                k=0
                if evaluator[l] == "holdout":
                    for j in range(len(metrics)):
                        evaluator = EvaluateHoldout(n_wait = 10000,
                                                    max_samples = stream_length,
                                                    metrics = metrics[j],
                                                    output_file = 'results/holdout_evaluation_' + datastream_names[i] + '.csv',
                                                    restart_stream = True,
                                                   )
            
                        evaluator.evaluate(stream=stream, 
                                           model=[rslvq_sgd, 
                                                  rslvq_adadelta, 
                                                  rslvq_rmsprop, 
                                                  rslvq_adam]) 
                                          # model_names=clfs_names)
                        print('Holdout evaluation '+ datastream_names[i] +' finished')
                else:
                    for k in range(len(metrics)):
                        evaluator_rslvq = EvaluatePrequential(n_wait = 10000,
                                                              max_samples = stream_length,
                                                              pretrain_size = 10000,
                                                              metrics = metrics[k],
                                                              output_file = 'results/holdout_prequential_' + datastream_names[i] + '.csv',
                                                              )
                            
                        evaluator.evaluate(stream=stream, 
                                           model=[rslvq_sgd, 
                                                  rslvq_adadelta, 
                                                  rslvq_rmsprop, 
                                                  rslvq_adam])
                                          # model_names=clfs_names)
                        print('Prequential evaluation '+ datastream_names[i] +' finished')
import numpy
import pandas as pd
from Algorithms.rslvq_all import RSLVQall
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.metrics import ClassificationMeasurements

def custom_evaluation(datastreams, datastream_names, clfs, clfs_names, stream_length):

        ev=['holdout', 'prequel']
        
        eval_results_holdout = []
        eval_results_prequel = []

        mod = clfs

        stream = datastreams
        stream.prepare_for_use()

        #print(datastream_names[index])
        #print(stream.get_data_info())

        evaluator = EvaluateHoldout(n_wait = 10000,
                                                    max_samples = stream_length,
                                                    metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m'],
                                                    restart_stream = True)
        try:
            evaluator.evaluate(stream=stream, model=mod)
                                                        

            eval_results_holdout.append(evaluator.get_mean_measurements())
        except Exception as e:
            print(e)
            
        print('')
        print('Holdout evaluation for '+datastream_names+' stream finished')

        evaluator_rslvq = EvaluatePrequential(n_wait = 10000,
                                              max_samples = stream_length,
                                              pretrain_size = 10000,
                                              metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m']) 
        try:
            evaluator.evaluate(stream=stream, model=mod)
                                                         
                       
            eval_results_prequel.append(evaluator.get_mean_measurements())
        except Exception as e:
            print(e)
           
        print('')
        print('Prequential evaluation for '+datastream_names+' stream finished')
        
        
        for i, item in enumerate(eval_results_holdout, start=0):
            print('')
            print('Results for the ' + ev[0] + ' eval:')
            for j in range(len(item)):
                print('')
                print(clfs_names+' :')
                print('Performance: '+str(round(item[j].get_accuracy(), 4)))
                print('Kappa: '+str(round(item[j].get_kappa(), 4)))
                print('Kappa_m: '+str(round(item[j].get_kappa_m(), 4)))
                print('Kappa_t: '+str(round(item[j].get_kappa_t(), 4)))

        print('')

        for i, item in enumerate(eval_results_prequel, start=0):
            print('')
            print('Results for the ' + ev[1] + ' eval:')
            for j in range(len(item)):
                print('')
                print(clfs_names+' :')
                print('Performance: '+str(round(item[j].get_accuracy(), 4)))
                print('Kappa: '+str(round(item[j].get_kappa(), 4)))
                print('Kappa_m: '+str(round(item[j].get_kappa_m(), 4)))
                print('Kappa_t: '+str(round(item[j].get_kappa_t(), 4)))
        print('')
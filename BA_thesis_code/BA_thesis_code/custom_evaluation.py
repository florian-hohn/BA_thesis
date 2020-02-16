import numpy
import pandas as pd
from Algorithms.rslvq_all import RSLVQall
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.metrics import ClassificationMeasurements

def custom_evaluation(datastreams, clfs, stream_length, Prequential = False):

        eval_time = 0
        eval_results = []
        ev=['Holdout', 'Prequential']
        mod = clfs[0]

        stream = datastreams[0]
        stream.prepare_for_use()
        #print(stream.get_data_info())
        #print(datastream_names[index])
        
        
        if Prequential == True:
            evaluator = EvaluatePrequential(max_samples = stream_length,metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m', 'running_time']) 
            eval_text=ev[1]
        else:
            evaluator = EvaluateHoldout(max_samples = stream_length,
                                    metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m','running_time'])
            eval_text=ev[0]

        print('')
        print(eval_text+' evaluation for '+datastreams[1]+' stream:')
        try:
            evaluator.evaluate(stream=stream, model=mod)
                                                        
            eval_time = evaluator.running_time_measurements[0]._total_time
            eval_results.append(evaluator.get_mean_measurements())
        except Exception as e:
            print(e)
            
        print('')
        print(eval_text+' evaluation for '+datastreams[1]+' stream finished')

        #try:
        #    evaluator.evaluate(stream=stream, model=mod)
                                                         
                       
        #    eval_results_prequel.append(evaluator.get_mean_measurements())
        #except Exception as e:
        #    print(e)
           
        #print('')
        #print('Prequential evaluation for '+datastreams[1]+' stream finished')
        
        
        for i, item in enumerate(eval_results, start=0):
            print('')
            print('Results for the ' + eval_text + ' eval:')
            for j in range(len(item)):
                print('')
                print(clfs[1]+' :')
                print('Performance: '+str(round(item[j].get_accuracy(), 4)))
                print('Kappa: '+str(round(item[j].get_kappa(), 4)))
                print('Kappa_m: '+str(round(item[j].get_kappa_m(), 4)))
                print('Kappa_t: '+str(round(item[j].get_kappa_t(), 4)))
                print('Total comp. time: '+str(round(eval_time),2))
                #print('Total comp. time: '+ str(round(item[j].get_kappa_t(), 4)))

        print('')
import copy
import numpy
import pandas as pd
from Algorithms.rslvq_all import RSLVQall
from skmultiflow.evaluation import EvaluateHoldout
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.metrics import ClassificationMeasurements

def custom_evaluation(datastreams, clfs, stream_length, Prequential = False):

        eval_results=[]
        eval_time = 0
        eval_time = 0
        eval_acc = 0
        eval_kappa = 0
        eval_kappam = 0
        eval_kappat = 0
        ev=['Holdout', 'Prequential']
        mod = clfs[0]
        resultpath = ""
        rdf = []
        

        stream = datastreams[0]
        stream.prepare_for_use()
        #print(stream.get_data_info())
        #print(datastream_names[index])
        
        
        if Prequential == True:
            resultpath = "results/Prequential/"+ev[1]+"_"+datastreams[1]+"_"+clfs[1]+".csv"
            evaluator = EvaluatePrequential(max_samples = stream_length,metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m', 'running_time']) 
            eval_text=ev[1]
        else:
            resultpath ="results/Holdout/"+ev[0]+"_"+datastreams[1]+"_"+clfs[1]+".csv"
            evaluator = EvaluateHoldout(max_samples = stream_length,
                                    metrics = ['accuracy','kappa', 'kappa_t', 'kappa_m','running_time'])
            eval_text=ev[0]

        print('')
        print(eval_text+' evaluation for '+datastreams[1]+' stream:')
        try:
            evaluator.evaluate(stream=stream, model=mod)
                
            eval_results.append(evaluator.get_mean_measurements())
            eval_time = evaluator.running_time_measurements[0]._total_time
            
            for i, item in enumerate(eval_results, start=0):
                eval_acc = item[0].get_accuracy()
                eval_kappa = item[0].get_kappa()
                eval_kappam = item[0].get_kappa_m()
                eval_kappat = item[0].get_kappa_t()
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
        
        
        
        print('')
        print('Results for the ' + eval_text + ' eval:')
        print('')
        print(clfs[1]+' :')
        print('Accuracy: '+str(round(eval_acc, 4)))
        print('Kappa: '+str(round(eval_kappa, 4)))
        print('Kappa_m: '+str(round(eval_kappam, 4)))
        print('Kappa_t: '+str(round(eval_kappat, 4)))
        print('Total comp. time: '+str(round(eval_time,2)))

        try:
            #create dataframe with the ruslts for the datastream and the now active clf and save it as csv
            rdf_data = [[
                datastreams[1], 
                clfs[1], 
                str(round(eval_acc, 4)), 
                str(round(eval_kappa, 4)),
                str(round(eval_kappam, 4)),
                str(round(eval_kappat, 4)),
                str(round(eval_time,2))
                ]]
            rdf = pd.DataFrame(rdf_data,columns=['Stream','Clf', 'Accuracy', 'Kappa', 'Kappa_m', 'Kappa_t', 'total comp. time'])
            rdf.to_csv(resultpath, index = None, header=True)
        except Exception as e:
            print(e)

        #print('Total comp. time: '+ str(round(item[j].get_kappa_t(), 4)))

        print('')
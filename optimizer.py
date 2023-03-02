

from wallet import Wallet
from policy import Policy, Policy_01, Policy_02, Policy_03
from agent import Agent
from data import Data, loadData#, calculate_ema, computeRSI, createIndicatorDICO,createIndicator, addIndicator
from indicator import calculate_ema, computeRSI,createIndicator_bis, addIndicator, Init_indicator, findLastExtrema, extremaLocal
from Results import Results
import matplotlib.pyplot as plt
import optuna
from optuna import Trial
import pickle # Save and load data to and from storage
import random
import numpy as np 
from termcolor import colored
import datetime
import os
import time
import sys
import json
from scipy.signal import argrelextrema

def findLastId():
    current_directory = os.getcwd()
    result_dir = os.path.join(current_directory, 'Results')
    #print(result_dir)
    subdirs = [int(d.split('__')[0]) for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
    # Trouver le dernier identifiant en incrémentant de 1
    if subdirs:
        last_id = max(subdirs) + 1
    else:
        last_id = 1
    print(last_id)
    return last_id

class Optimizer() :

    """ Optimizes the input parameter space and output the best set found as well as intermediary visualization asset of the training """

    def __init__(self, data : Data, policy : type,ignoreTimer : int = 50,data_name : str="Results") :
        # Todo : Ensure variable types and values
        path=os.getcwd()
        self.folder_name=path+"/Results/"+data_name+"/"
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.text_file = open(self.folder_name+"log.txt",'a')
        self.text_file.close()
        
        
        self._data = data # The data to be used
        self._policyClass = policy # The policy class to optimize
        self._results = Results(self.folder_name)
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._study = optuna.create_study(direction="maximize") # The Optuna data structure to be used for the hyper-parameters tuning
        self.trainPerformances = [] # Save training performances over time. This is useful to know how much the model fits the training data.
        self.validPerformances = [] # Save validation performances over time. This is useful to know how much the model generalizes.
        self.testPerformances = [] # Save test performances over time. This is usefull to know how well the model would perform on new data.
        self.params = [] #Save params performances over time
        self.bestTrainedModel = -199999
        self.bestValidedModel = -199999
        self.bestTestedModel = -199999
        self.bestTrainParams= None
        self.bestValidParams= None
        self.bestTestParams= None
        self.ignoreTimer=ignoreTimer
        self.PRINTED = False
        self.countSincePrinted=0
        self.paramTemp=[[] for i in range(7)]
        self.paramImpact=[[] for i in range(7)]
        self.time=0
        path=os.getcwd()
        self.folder_name=path+"/Results/"+data_name+"/"
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.text_file = open(self.folder_name+"log.txt",'a')
        self.text_file.close()
        
        
    def print_save(self,text : str):
        self.text_file = open(self.folder_name+"log.txt",'a')
        self.text_file.write(text+"\n")
        self.text_file.close()
    
    
    def save_param(self):
        self.text_file = open(self.folder_name+"param.txt",'a')
        txtt="Trial: {} ==>  ".format(train_number)+"Last Train: {}%, std : {}".format(np.round(self._results.lastTrainPerf, decimals=3),np.round(self._results.lastTrainStd, decimals=4))+" ; Last Valid: {}%, std : {}".format(np.round(self._results.lastValidPerf, decimals=4),np.round(self._results.lastValidStd, decimals=4))+" => Test: {}%".format(np.round(self._results.lastTestPerf, decimals=4))
        self.text_file.write(txtt+"\n")
        self.text_file.write(json.dumps(self._results.params[-1]))
        self.text_file.write("\n\n")
        self.text_file.close()
        
        
    def runExperiment(self, params : dict, dataset : str = "train", sequenceLength : int = 1000, sav : bool = False):
            # Get the proper data
        if (dataset == "train") : closeSequences, highSequences, lowSequences, openSequences, volumeSequences, indics= self._data.trainSequences("close"), self._data.trainSequences("high"), self._data.trainSequences("low"), self._data.trainSequences("open"),self._data.trainSequences("volume"),self._data.trainSequences("indic")
        if (dataset == "valid") : closeSequences, highSequences, lowSequences, openSequences, volumeSequences, indics = self._data.validSequences("close"), self._data.validSequences("high"), self._data.validSequences("low"), self._data.validSequences("open"), self._data.validSequences("volume"),self._data.validSequences("indic")
        if (dataset == "test") : closeSequences, highSequences, lowSequences, openSequences, volumeSequences, indics = self._data.testSequences("close"), self._data.testSequences("high"), self._data.testSequences("low"), self._data.testSequences("open"), self._data.testSequences("volume"),self._data.testSequences("indic")
            # Compute the performance of the policy on all the sequences
        totalPerformance = 0 # Sum of the performances over all the sequences
        tot_array = []
        transac_array = []
        UnitCount=0
        if sav:
            count,wins,loss=0,0,0
        for closeSequence, highSequence, lowSequence, openSequence, volumeSequence, indic in zip(closeSequences, highSequences, lowSequences, openSequences, volumeSequences,indics) :
               # Instanciate an agent to run the policy of our data
            wallet = Wallet(fees=0.01)
            policy = self._policyClass()
            policy.params = params # Always use the same params provided as arguments (instead of sampling again)
            agent = Agent(wallet, policy,ignoreTimer=self.ignoreTimer)
                # Run the agent on the data
            
            for closeValue, highValue, lowValue, volumValue, indicator in zip(closeSequence, highSequence, lowSequence, volumeSequence,indic) :
                UnitCount+=1
                agent.act(indicator, closeValue, highValue, lowValue, volumValue)
                # TODO : What is sequenceLength for ? Max length of a single sequence or all the sequences ?
            #datas=np.array(agent.policy.val)
            #print(agent.policy.val)
            #print([np.mean(datas),np.max(datas),np.min(datas)])
            for elt in policy.weight:
                for i in range(7):
                    self.paramTemp[i].append(elt[i])
            if sav:
                wins += len(policy.wins)
                loss += len(policy.loss)
                count+=1
                name = "best_test_"+str(count)+".png"
                
                closeSeq, highSeq, lowSeq, openSeq=[], [], [], []
                for elt in closeSequence:
                    closeSeq.append(elt)
                for elt in highSequence:
                    highSeq.append(elt)
                for elt in lowSequence:
                    lowSeq.append(elt)
                for elt in openSequence:
                    openSeq.append(elt)
                policy.plot(closeSeq,openSeq,highSeq,lowSeq,self.folder_name,self._data.ratio,name=name,ignoreTimer = 0)
                #print("plot  +  "+ name)
            totalPerformance += agent.wallet.profit(closeValue)
            tot_array.append(agent.wallet.profit(closeValue))
            transac_array.append(agent.wallet.allTransactions())
        for i in range(len(tot_array)):
            tot_array[i] = tot_array[i]/UnitCount*self._data.perday
        if sav:
            #print(wins+loss)
            
            if wins+loss>0:
                wr=np.round(wins/(wins+loss)*100,decimals=1)
                
            else:
                wr=-1
            gain=np.round(totalPerformance/(UnitCount-self._data.numPartitions*self.ignoreTimer)*self._data.perday,decimals=3)
            tradeRate=np.round((wins+loss)/(UnitCount-self._data.numPartitions*self.ignoreTimer)*self._data.perday,decimals=2)
            print(colored("Résultat dans le test final : WR : {}%, gain quotidien {}%, nbr de trades quotidiens : {}".format(wr,gain, tradeRate),"green"))
        #print("totalunit : {}, total value : {}, resultat {}, perday {}".format(UnitCount,totalPerformance,totalPerformance/UnitCount*self._data.perday,self._data.perday))
        return totalPerformance/(UnitCount-self._data.numPartitions*self.ignoreTimer)*self._data.perday, tot_array, transac_array

    def optiPrint(self,study, trial):
        global train_number
        train_number+=1
        
        if not(self._results.PRINTED) :
            self._results.PRINTED=True
            self._results.countSincePrinted=0
            print(colored("Trial: {} ==>  ".format(train_number),"cyan"),colored("Train: {}%, std : {}".format(np.round(self._results.bestTrainedModel, decimals=3),np.round(self._results.bestTrainedStd, decimals=4)),self._results.PRINTED_Train),colored(" ; Valid: {}%, std : {}".format(np.round(self._results.bestValidedModel, decimals=4),np.round(self._results.bestValidStd, decimals=4)),self._results.PRINTED_Valid),colored(" => Test: {}%".format(np.round(self._results.bestTestedModel, decimals=4)), 'cyan'))
            txtt="Trial: {} ==>  ".format(train_number)+"Train: {}%, std : {}".format(np.round(self._results.bestTrainedModel, decimals=3),np.round(self._results.bestTrainedStd, decimals=4))+" ; Valid: {}%, std : {}".format(np.round(self._results.bestValidedModel, decimals=4),np.round(self._results.bestValidStd, decimals=4))+" => Test: {}%".format(np.round(self._results.bestTestedModel, decimals=4))
            self.print_save(txtt)
            self.save_param()
        # elif self._results.countSincePrinted>100:
        #     self._results.countSincePrinted=0
        #     dt=time.time()-self.time
        #     self.time=time.time()
        #     print("ecouled time : {}".format(dt))
        #     print(colored("Trial: {} ==>  Best Train: {}% ; Best Valid: {}% => Test: {}%".format(train_number,np.round(self._results.bestTrainedModel, decimals=3),np.round(self._results.bestValidedModel, decimals=4),np.round(self._results.bestTestedModel, decimals=4)), 'white'))
    
    def objective(self, trial : Trial) :
             # Sample a set of params to be tested
        policy = self._policyClass()
        policy.sampleFromTrial(trial) # TODO :: Make class method and remove policy instanciation
        params = policy.params
        self.params.append(params)
            # Compute performances
        self.paramTemp=[[] for i in range(7)] 
        
        trainPerformance,train_arr, train_transac = self.runExperiment(params, "train",False)
        validPerformance,valid_arr, valid_transac = self.runExperiment(params, "valid",False)
        testPerformance, test_arr, test_transac = self.runExperiment(params, "test",False)
        if(len(self.paramTemp[0])>0):
            for ind in range(7):
                self.paramImpact[ind].append(np.mean(self.paramTemp[ind]))
        
        self._results.saveExperiment(trainPerformance, validPerformance, testPerformance, params, train_arr,valid_arr, test_arr, train_transac, valid_transac, test_transac)

#             self.bestTestParams=params

            # Return training signal to Optuna
        #print(trainPerformance, np.std(np.array(train_arr)))
        return trainPerformance-2*np.std(np.array(train_arr))

    def fit(self, temps : float) :
        
        # #création des indicateurs pertinents pour la policy
        # indices,self._data.ratio=createIndicator_bis(self._data)#self._data
        
        # # for z in [3,5,7]:
        # #     for e in [3,5,7]:
        # #         hyperP = {
        # #             "Theta" : z,
        # #             "Theta_bis" : e,
        # #             "Theta_der" : 3,
        # #             "Theta_der2" : 3,
        # #             "Theta_RSI" : 14}
        # #         indices,self._data.ratio = Init_indicator(indices, data, hyperP)
        # #         indices=addIndicator(indices,self._data.ratio, hyperP)
        # # #suppression des self.ignoreTimer valeurs des data et de l'indicateur permettant d'avoir des moyennes stables
        # # indices=indices[self.ignoreTimer:]
        # # self._data.data=self._data.data[self.ignoreTimer:]        
        
        # for z in range(1,41,3):
        #         hyperP = {
        #             "Theta" : 3,
        #             "Theta_bis" : 3,
        #             "Theta_der" : 3,
        #             "Theta_der2" : 3,
        #             "Theta_RSI" : 14,
        #             "Theta_C" : z}
        #         indices,self._data.ratio = Init_indicator(indices, data, hyperP)
        #         indices=addIndicator(indices,self._data.ratio, hyperP)
        # #suppression des self.ignoreTimer valeurs des data et de l'indicateur permettant d'avoir des moyennes stables
        # indices=indices[self.ignoreTimer:]
        # self._data.data=self._data.data[self.ignoreTimer:]
        
        
        
        # #print(len(indices))
        # #print(len(self._data.data))
        
        # for j in range(len(self._data.data)):
        #     self._data.data[j].indic=indices[j]
        
        print(colored("Study launch for {} minutes, with {} partitions ".format(np.round(temps/60,decimals=2),self._data.numPartitions),"green"))
        self.print_save("Study launch for {} minutes, with {} partitions ".format(np.round(temps/60,decimals=2),self._data.numPartitions))
        debut=datetime.datetime.now()
        sf=int((debut.second+temps)%60)
        mf=(int((debut.second+temps)/60)+debut.minute)%60
        hf=int((int((debut.second+temps)/60)+debut.minute)/60+debut.hour)%24
        D=int(int((int((debut.second+temps)/60)+debut.minute)/60+debut.hour)/24)
        if D==0:
            print('Fin à {}h {}m {}s'.format(hf,mf,sf))
        elif D==1:
            print('Fin demain à {}h {}m {}s'.format(hf,mf,sf))
        # <budget> : Time allocated to the fit of the models in s # TODO : Handle budget
            # Optimize the objective
        optuna.logging.disable_default_handler()
        self._study.optimize(self.objective, timeout=temps,callbacks=[self.optiPrint])#temps
            # After fitting, plot the data to visualize the training
        #self.plotPerformances()
        self._results.plotPerformances(self.folder_name)
        self._results.plotParams(self.folder_name)
        param_pese=["derivé","dérivé seconde","RSI","derive_diff_close","sign_diff_moy","volume","croisement_moyennes"]

        compt=0
        for i,tab in enumerate(self.paramImpact):
            plt.figure(figsize=(20,11))
            plt.title("influence de : "+param_pese[i]+" au cours des trades"+" moy="+str(np.round(np.mean(np.abs(tab)),decimals=5)))
            plt.grid()
            plt.plot(tab)
            compt+=1
            plt.savefig(self.folder_name+"poids_"+param_pese[i],bbox_inches='tight')
            plt.close()
            
        self.runExperiment(self._results.bestValidParams,"test",sav=True)


    def getBestTrainModel(self) :
        return [self.bestTrainedModel,self.bestTrainParams] # TODO : Retrieve best model on train from study


    def getBestValidModel(self) :
        return [self.bestValidedModel,self.bestValidParams] # TODO : Retrieve best model on valid from study


    def saveState(self, folderPath : str = "./_data/optimizerStates/") :
        with open(f"{folderPath}opt_{random.random()}", "wb") as saveFile:
            pickle.dump(self, saveFile, protocol=pickle.HIGHEST_PROTOCOL)


    def loadState(self, savePath : str) :
        with open(savePath, "rb") as readFile:
            return pickle.load(readFile)


if __name__ == "__main__" :
        # Get data to feed to optimizer
    for t in [5]:#,15]:,3,5
        ignoreTimer=150
        #data = loadData(paire="BTCBUSD", sequenceLength=24*30*4*10*3, interval_str="{}m".format(t), numPartitions=3, reload=True,ignoreTimer=ignoreTimer)
        data = loadData(paire="BTCBUSD", sequenceLength=24*30*4, interval_str="{}m".format(t), numPartitions=10,trainProp = 0.6, validProp = 0.25, testProp  = 0.15, reload=True,ignoreTimer=ignoreTimer)
        data.plot() # and plot it
        print("création des indices ....")
        #création des indicateurs pertinents pour la policy
        indices,data.ratio=createIndicator_bis(data)#self._data
        
        # for z in [1,3]:
        #     for e in [5,7.5,10]:
        #         hyperP = {
        #             "Theta" : 3,
        #             "Theta_bis" : 3,
        #             "Theta_der" : 3,
        #             "Theta_der2" : 3,
        #             "Theta_RSI" : 14,
        #             "Theta_C" : 200,
        #             "SL_max" : e/10.,
        #             "SL_min" : z/10}
        #         indices,data.ratio = Init_indicator(indices, data, hyperP)
        #         indices=addIndicator(indices,data.ratio, hyperP)
        # #suppression des self.ignoreTimer valeurs des data et de l'indicateur permettant d'avoir des moyennes stables
        # indices=indices[ignoreTimer:]
        # data.data=data.data[ignoreTimer:]  

        hyperP = {
            "Theta" : 3,
            "Theta_bis" : 3,
            "Theta_der" : 3,
            "Theta_der2" : 3,
            "Theta_RSI" : 14,
            "Theta_C" : 200,
            "SL_max" : 1,
            "SL_min" : 0.2}
        indices,data.ratio = Init_indicator(indices, data, hyperP)
        indices=addIndicator(indices,data.ratio, hyperP)
        #suppression des self.ignoreTimer valeurs des data et de l'indicateur permettant d'avoir des moyennes stables
        indices=indices[ignoreTimer:]
        data.data=data.data[ignoreTimer:]
            

        for j in range(len(data.data)):
            data.data[j].indic=indices[j]
        print("Indices créés")
    
        for j in range(1):
            duree_min = 4
            plt.close("all")
            train_number=0
            new_id = findLastId()
            opti_name="{}__SL_clean_tf_{}m_dur_{}".format(new_id,t,duree_min)
            #opti_name="test"
            optimizer = Optimizer(data, Policy_02, ignoreTimer=ignoreTimer,data_name=opti_name)
            optimizer.fit(60*duree_min)
            print("Fin algo : {} executions".format(train_number))
            optimizer.print_save("Fin algo : {} executions".format(train_number))

    #optimizer.runExperiment(optimizer.bestTestParams,"test",sav=True)
sys.exit()

#%%
seq = np.array(data.testSequence("open")[:60])
maxi = argrelextrema(seq,np.greater)
maxi_val = seq[maxi[0]]
mini = argrelextrema(seq,np.less)
mini_val = seq[mini[0]]
plt.plot(seq)
plt.scatter(maxi,maxi_val, c = 'green')
plt.scatter(mini,mini_val, c = 'red')

#%%
plt.close("all")
from matplotlib.patches import Rectangle


fig, ax1 = plt.subplots(figsize=(20,11))
#ax2 = ax1.twinx()
#ax2.plot(np.array(data.getValueStream_indic(hyperP, "std_A"))/np.array(data.getValueStream_indic(hyperP, "std_B")),color='gray')
#ax1.plot(data.getValueStream("close"))
close = np.array(data.getValueStream("close"))
ope = np.array(data.getValueStream("open"))
high = np.array(data.getValueStream("high"))
low = np.array(data.getValueStream("low"))

close = np.array(data.getValueStream_indic(hyperP, "closeV"))
ope = np.array(data.getValueStream_indic(hyperP, "openV"))
high = np.array(data.getValueStream_indic(hyperP, "highV"))
low = np.array(data.getValueStream_indic(hyperP, "lowV"))


# RSI = np.array(data.getValueStream_indic(hyperP, "RSI_stoch"))
# MACD_cross = np.array(data.getValueStream_indic(hyperP, "MACD_crossing"))
# MACD = np.array(data.getValueStream_indic(hyperP, "MACD"))
# MACD_s = np.array(data.getValueStream_indic(hyperP, "MACD_signal"))
# trend = np.array(data.getValueStream_indic(hyperP, "close_moy_C"))



TP_max = np.array(data.getValueStream_indic(hyperP, "maxi_proche"))/100.*close + close
TP_min = -np.array(data.getValueStream_indic(hyperP, "mini_proche"))/100.*close + close

unpourcent = close*1.0075
unpourcent_b = close*1.003

x = np.arange(len(close))+0.5


for j in range(len(high)):
    if close[j]>ope[j]:
        c='green'
    else:
        c='red'
    ax1.plot([j+0.5,j+0.5],[low[j],high[j]], c)
    ax1.add_patch(Rectangle((j, min(close[j],ope[j])), 1, max(close[j],ope[j])-min(close[j],ope[j]),facecolor =c))
#ax1.plot(TP_max, 'r')#ls ='--'
ax1.plot(x,TP_max,c = 'green', marker = "x", ls = "none")
ax1.plot(x,TP_min,c = 'red', marker = "x", ls = "none")
#ax1.plot(unpourcent, 'b')
plt.fill_between(x, unpourcent,unpourcent_b, color='b', alpha = 0.3)
# ax1.plot(stdB, 'r', ls ='--')
# ax2.plot(MACD, 'blue' )
# ax2.plot(MACD_s, 'orange' )
#ax2.plot(MACD_cross, 'orange' )
# ax2.plot(rsi, 'red' )
# ax1.plot(close + stdB, 'g', ls ='--')
# ax1.plot(close - stdB, 'g', ls ='--')
# ax1.plot(np.array(data.getValueStream_indic(hyperP, "close_moy_A"))*ratio,color='y')
# ax1.plot(np.array(data.getValueStream_indic(hyperP, "close_moy_B"))*ratio,color='r')
plt.title("comparaison indicatieur")
ax1.set_xlabel('Unité de temps',fontsize=20)
ax1.set_ylabel('Prix ($)',fontsize=20)
plt.grid()


#%%
plt.close("all")
def findLastExtrema(array, val_min, val_max, maxim : bool = True, n : int = 2):
    array_inverted_0 = np.copy(array)#[::-1]
    indice = 0
    found_at_least_one = False
    pos_extrema = False
    best_one = False
    for j in range(n):
        if array_inverted_0[j]>val_max :
            return val_max, pos_extrema
        else :
            if  (array_inverted_0[j]>np.max([val_min,best_one]) and found_at_least_one) or (array_inverted_0[j]>val_min and not(found_at_least_one)):
                best_one = array_inverted_0[j]
                found_at_least_one = True
                pos_extrema = j
    indice = n
    while (array_inverted_0[indice]<=val_max) and indice<len(array_inverted_0)-1-n:
        if (array_inverted_0[indice]>np.max([best_one,val_min])and found_at_least_one) or  (array_inverted_0[indice]>val_min and not(found_at_least_one)):
            if extremaLocal(array_inverted_0[indice-n:indice+n+1], maxim):
                best_one = array_inverted_0[indice]
                found_at_least_one = True
                pos_extrema = j
        indice += 1
    if not(found_at_least_one):
        return val_max, pos_extrema
    else:
        return best_one, pos_extrema

def extremaLocal(array, maxim : bool = True):
    indice_milieu = int(np.round(len(array)/2))
    arr = np.concatenate((array[:indice_milieu] , array[indice_milieu+1:]))
    val_milieu = array[indice_milieu]
    res = 1
    if maxim :
        for elt in arr:
            if elt >= val_milieu :
                res *=0
    else :
        for elt in arr:
            if elt <= val_milieu :
                res *=0
    return res == 1
n = 2
fig, ax1 = plt.subplots(figsize=(20,11))

close = np.array(data.getValueStream_indic(hyperP, "closeV"))*1000
ope = np.array(data.getValueStream_indic(hyperP, "openV"))*1000
high = np.array(data.getValueStream_indic(hyperP, "highV"))*1000
low = np.array(data.getValueStream_indic(hyperP, "lowV"))*1000

x = np.arange(len(close))+0.5

maxi = close + np.array(data.getValueStream_indic(hyperP, "maxi_proche"))/100 * close 
mini = close - np.array(data.getValueStream_indic(hyperP, "mini_proche"))/100 * close 

plt.grid()
plt.show()
plt.plot(x,maxi,c = 'green', marker = "x", ls = "none")
plt.plot(x,mini,c = 'red', marker = "x", ls = "none")

plt.title("indicateur",fontsize=30, fontweight = 'bold')

for j in range(len(high)):
    if close[j]>ope[j]:
        c='green'
    else:
        c='red'
    ax1.plot([j+0.5,j+0.5],[low[j],high[j]], c)
    ax1.add_patch(Rectangle((j, min(close[j],ope[j])), 1, max(close[j],ope[j])-min(close[j],ope[j]),facecolor =c))
    

fig, ax1 = plt.subplots(figsize=(20,11))   
 

plt.title("local",fontsize=30, fontweight = 'bold')

for j in range(len(high)):
    if close[j]>ope[j]:
        c='green'
    else:
        c='red'
    ax1.plot([j+0.5,j+0.5],[low[j],high[j]], c)
    ax1.add_patch(Rectangle((j, min(close[j],ope[j])), 1, max(close[j],ope[j])-min(close[j],ope[j]),facecolor =c))
    
   
l = 107
last_extrema_pos = False
for i, elt in enumerate(close):
    if i > l and i < len(close)-n-2:
        tp_min = -(low[i-l:i+1]-elt)/elt*100   
        tp_max = (high[i-l:i+1]-elt)/elt*100   
        tp_min = tp_min[::-1]
        tp_max = tp_max[::-1]
        lastEX_min, pos = findLastExtrema(tp_min, 0.2, 1, maxim = True, n = 2)
        lastEX_max, pos = findLastExtrema(tp_max, 0.2, 1, maxim = True, n = 2)
        plt.scatter(i+0.5,+lastEX_max/100*elt+elt, c = 'green', marker = 'x')
        plt.scatter(i+0.5,-lastEX_min/100*elt+elt, c = 'red', marker = 'x')
plt.grid()
plt.show()


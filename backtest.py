

from wallet import Wallet
from policy import Policy, Policy_01, Policy_02, Policy_03
from agent import Agent
from data import Data, loadData#, calculate_ema, computeRSI, createIndicatorDICO,createIndicator, addIndicator
from indicator import calculate_ema, computeRSI,createIndicator_bis, addIndicator, Init_indicator
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

   



plt.close("all")
ignoreTimer=50
data_name="backtest_01fees_RR3_stddiv_test"
data = loadData(paire="BTCBUSD", sequenceLength=20*24*30*4, interval_str="3m", numPartitions=9, reload=True,ignoreTimer=ignoreTimer)
data.plot() # and plot it



path=os.getcwd()
folder_name=path+"/Results/"+data_name+"/"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
policyClass = Policy_03




params = {'TP': 1.5,
 'SL': 0.6,
 'weight_0': -0.4183485221852743,
 'weight_1': -0.5660605060891415,
 'weight_2': -0.8891577335759198,
 'weight_3': -0.4660225530118107,
 'weight_4': 0.480878283386754,
 'weight_5': 0.5705812174523972,
 'weight_6': 0.5729945885443585,
 'Theta': 7,
 'Theta_bis': 5,
 'normFactor': 4.949489013188023}


params = {"TP": 3, "SL": 0.5, "weight_0": 0.5925377451557796, "weight_1": 0, "weight_2": -0.28148749244735105, "weight_3": 0, "weight_4": -0.7002733979273887, "weight_5": 0, "weight_6": 0.8545965859654405, "Theta": 3, "Theta_bis": 3, "Theta_RSI": 14, "normFactor": 51.8352677444534}


hyperP = {
    "Theta" : params["Theta"],
    "Theta_bis" : params["Theta_bis"],
    "Theta_der" : 3,
    "Theta_der2" : 3,
    "Theta_RSI" : params["Theta_RSI"]}


#création des indicateurs pertinents pour la policy
#indices,self._data.ratio=createIndicatorDICO(self._data, hyperP)


indices,ratio=createIndicator_bis(data)#self._data


indices,ratio = Init_indicator(indices, data, hyperP)
indices=addIndicator(indices,ratio, hyperP)
#suppression des self.ignoreTimer valeurs des data et de l'indicateur permettant d'avoir des moyennes stables
indices=indices[ignoreTimer:]
data.data=data.data[ignoreTimer:]



for j in range(len(data.data)):
    data.data[j].indic=indices[j]

print(colored("Study launch with {} partitions ".format(data.numPartitions),"green"))

# <budget> : Time allocated to the fit of the models in s # TODO : Handle budget
    # Optimize the objective


dataset = "train"    # Get the proper data
if (dataset == "train") : closeSequences, highSequences, lowSequences, volumeSequences, indics= data.trainSequences("close"), data.trainSequences("high"), data.trainSequences("low"),data.trainSequences("volume"),data.trainSequences("indic")
# Compute the performance of the policy on all the sequences
totalPerformance = 0 # Sum of the performances over all the sequences
tot_array = []
UnitCount=0
paramTemp=[[] for i in range(7)] 
paramImpact=[[] for i in range(7)]
count,wins,loss=0,0,0
for closeSequence, highSequence, lowSequence, volumeSequence, indic in zip(closeSequences, highSequences, lowSequences, volumeSequences,indics) :
       # Instanciate an agent to run the policy of our data
    wallet = Wallet(fees=0.08)
    policy = policyClass()
    policy.params = params # Always use the same params provided as arguments (instead of sampling again)
    agent = Agent(wallet, policy,ignoreTimer=ignoreTimer)
        # Run the agent on the data
    
    for closeValue, highValue, lowValue, volumValue, indicator in zip(closeSequence, highSequence, lowSequence, volumeSequence,indic) :
        UnitCount+=1
        agent.act(indicator, closeValue, highValue, lowValue, volumValue)
        # TODO : What is sequenceLength for ? Max length of a single sequence or all the sequences ?
    for elt in policy.weight:
        for i in range(7):
            paramTemp[i].append(elt[i])

    wins += len(policy.wins)
    loss += len(policy.loss)
    count+=1
    name = "best_test_"+str(count)+".png"
    closeSeq=[]
        
        
        
            
    for elt in closeSequence:
        closeSeq.append(elt)
    
    policy.plot(closeSeq,folder_name,ratio,name=name,ignoreTimer = 0)
    #print("plot  +  "+ name)
    totalPerformance += agent.wallet.profit(closeValue)
    tot_array.append(agent.wallet.profit(closeValue))
for i in range(len(tot_array)):
    tot_array[i] = tot_array[i]/UnitCount*data.perday

    
if wins+loss>0:
    wr=np.round(wins/(wins+loss)*100,decimals=1)
    
else:
    wr=-1
tot_array = np.array(tot_array)
gain=np.round(totalPerformance/(UnitCount-data.numPartitions*ignoreTimer)*data.perday,decimals=3)
tradeRate=np.round((wins+loss)/(UnitCount-data.numPartitions*ignoreTimer)*data.perday,decimals=2)
print(colored("Résultat dans le test final : WR : {}%, gain quotidien {}%, nbr de trades quotidiens : {}\nmoyenne des partition : {}, std : {}".format(wr,gain, tradeRate,np.round(np.mean(tot_array),decimals=3), np.round(np.std(tot_array),decimals=4)),"green"))
#print("totalunit : {}, total value : {}, resultat {}, perday {}".format(UnitCount,totalPerformance,totalPerformance/UnitCount*self._data.perday,self._data.perday))
out = totalPerformance/(UnitCount-data.numPartitions*ignoreTimer)*data.perday 



# param_pese=["derivé","dérivé seconde","RSI","derive_diff_close","sign_diff_moy","volume","croisement_moyennes"]
# compt=0
# for i,tab in enumerate(paramTemp):
#     plt.figure(figsize=(20,11))
#     plt.title("influence de : "+param_pese[i]+" au cours des trades"+" moy="+str(np.round(np.mean(np.abs(tab)),decimals=5)))
#     plt.grid()
#     plt.plot(tab)
#     compt+=1
#     plt.savefig(folder_name+"poids_"+param_pese[i],bbox_inches='tight')


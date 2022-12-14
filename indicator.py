
import numpy as np
import matplotlib.pyplot as plt




def key_hyper_param_influents(hyper_params : dict):
    key="key_"
    for cle,value in hyper_params.items():
        key = key + cle + '_' + str(value) + '_'
    return key
        

def calculate_ema(newPrice : float, oldPrice : float, days : int = 10, smoothing : float = 2):
    return (newPrice * (smoothing / (1 + days))) + oldPrice * (1 - (smoothing / (1 + days)))

class indicator():
    
    def __init__(self):#,param : dict,tabKeys : list,tabValues : list):
        #parametres influant le calcul des indicateurs
        self.param_dict={}
        
# =============================================================================
#         key_param=key_hyper_param_influents(param)
#         self.param_dict[key_param]={}
#         for key,value in zip(tabKeys,tabValues):
#             self.param_dict[key_param][key]=value
# =============================================================================
        
    def addIndicator(self,param : dict, value ,key : str):
        key_param=key_hyper_param_influents(param)
        if (key_param not in self.param_dict):
            self.param_dict[key_param]={}
        self.param_dict[key_param][key]=value
        
    def get_Indicator(self,param : dict ,key : str):
        key_param=key_hyper_param_influents(param)
        return self.param_dict[key_param][key]
    
    def calculated(self, param : dict, key : str):
        key_param=key_hyper_param_influents(param)
        return key in self.param_dict[key_param]
    
    def getParamsKeys(self):
        keys=[]
        for key in self.param_dict:
            keys.append(key)
        return keys
    
def computeRSI(der, d : int = 14) :
    if len(der) < d :
        return 50
    gp = 0
    lp = 0
    for k in range(d):
        if der[-(1 + k)] > 0 :
            gp += der[-(1 + k)]
        elif der[-(1 + k)] < 0 :
            lp -= der[-(1 + k)]
    if lp <= 0 :
        return 100
    else :
        RS = gp / lp
        return 100 - 100 / (1 + RS)
    
def createIndicator(data,params):
    
    closeData = data.getValueStream('close')
    volumeData = data.getValueStream('volume')
    openData = data.getValueStream('open')
    highData = data.getValueStream('high')
    lowData = data.getValueStream('low')
    ratio =np.max(closeData)
    closeData /= ratio
    highData /= ratio
    openData /= ratio
    lowData /= ratio
    indices = {}
    indicateurs = []#0price regularized, 1volume, 
    #2deriv?? 1, 3d??riv?? 2, 4prix moyenn?? 1, 5prix moyenn?? 2, 6RSI, 7signe diff moy, 8d??riv?? diff moy
    #9d??riv?? 1 moy, 10d??riv?? 2 moy,11 signe ??cart
    
    
    for closeV, highV, lowV, volumeV in zip(closeData, highData, lowData, volumeData) :
        indicateurs.append(indicator(params,["closeV","highV","lowV","volume"],[closeV,highV,lowV,np.sqrt(np.sqrt(np.abs(volumeV))) * np.sign(volumeV)]))        
        indices=indicateurs[-1]
    indicateurs.append(indices)   
    return indicateurs,ratio  
   
def Init_indicator(indicateur,data,params):
    
    closeData = data.getValueStream('close')
    volumeData = data.getValueStream('volume')
    openData = data.getValueStream('open')
    highData = data.getValueStream('high')
    lowData = data.getValueStream('low')
    ratio =np.max(closeData)
    closeData /= ratio
    highData /= ratio
    openData /= ratio
    lowData /= ratio    
    
    for i in range(len(closeData)) :
        volumeV=volumeData[i]
        indicateur[i].addIndicator(params,closeData[i] ,"closeV")
        indicateur[i].addIndicator(params,highData[i] ,"highV")
        indicateur[i].addIndicator(params,lowData[i] ,"lowV")
        indicateur[i].addIndicator(params,np.sqrt(np.sqrt(np.abs(volumeV))) * np.sign(volumeV) ,"volume")    
    return indicateur,ratio 

def createIndicator_bis(data):
    
    closeData = data.getValueStream('close')
    indices = {}
    indicateurs = []
    
    
    for closeV in closeData :
        indicateurs.append(indicator())        
        indices=indicateurs[-1]
    indicateurs.append(indices)   
    return indicateurs,np.max(closeData) 
  
def addIndicator(indicateurs,ratio ,hyperP : dict()):
    if not(indicateurs[-1].calculated(hyperP,'deriv??')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=1:
                indicateurs[i].addIndicator(hyperP,0,"deriv??")
            else:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")-indicateurs[i-1].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"deriv??")
    
    if not(indicateurs[-1].calculated(hyperP,'d??riv??2')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=1:
                indicateurs[i].addIndicator(hyperP,0,"d??riv??2")
            else:
                value=indicateurs[i].get_Indicator(hyperP,"deriv??")-indicateurs[i-1].get_Indicator(hyperP,"deriv??")
                indicateurs[i].addIndicator(hyperP,value,"d??riv??2")    
                
    if not(indicateurs[-1].calculated(hyperP,'close_moy_A')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=1:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"close_moy_A")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"closeV"), indicateurs[i-1].get_Indicator(hyperP,"close_moy_A"), hyperP["Theta"])
                indicateurs[i].addIndicator(hyperP,value,"close_moy_A")

    if not(indicateurs[-1].calculated(hyperP,'close_moy_B')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=1:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"close_moy_B")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"closeV"), indicateurs[i-1].get_Indicator(hyperP,"close_moy_B"), hyperP["Theta"]*hyperP["Theta_bis"])
                indicateurs[i].addIndicator(hyperP,value,"close_moy_B")
                
    if not(indicateurs[-1].calculated(hyperP,'close_moy_C')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=1:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"close_moy_C")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"closeV"), indicateurs[i-1].get_Indicator(hyperP,"close_moy_C"), hyperP["Theta_C"])
                indicateurs[i].addIndicator(hyperP,value,"close_moy_C")
    
    if not(indicateurs[-1].calculated(hyperP,'der_moy_C')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=1:
                indicateurs[i].addIndicator(hyperP,0,"der_moy_C")
            else:
                value=indicateurs[i].get_Indicator(hyperP,"close_moy_C")-indicateurs[i-1].get_Indicator(hyperP,"close_moy_C")
                indicateurs[i].addIndicator(hyperP,value,"der_moy_C")   

        
    if not(indicateurs[-1].calculated(hyperP,'TPV')):
        for i in range(len(indicateurs)):
            value = (indicateurs[i].get_Indicator(hyperP,"closeV")+indicateurs[i].get_Indicator(hyperP,"highV")+indicateurs[i].get_Indicator(hyperP,"lowV"))/3
            indicateurs[i].addIndicator(hyperP,value,"TPV")              
        
    if not(indicateurs[-1].calculated(hyperP,'TP_moy_20')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=20:
                value=indicateurs[i].get_Indicator(hyperP,"TPV")
                indicateurs[i].addIndicator(hyperP,value,"TP_moy_20")
            else:
                array = [indicateurs[i-j].get_Indicator(hyperP,"TPV") for j in range(20)]
                value = np.mean(np.array(array))
                indicateurs[i].addIndicator(hyperP,value,"TP_moy_20")
    
    if not(indicateurs[-1].calculated(hyperP,'bollinger_high')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=20:
                value = indicateurs[i].get_Indicator(hyperP,"TP_moy_20")
            else:
                array = [indicateurs[i-j].get_Indicator(hyperP,"TPV") for j in range(20)]
                value = indicateurs[i].get_Indicator(hyperP,"TP_moy_20") + 2*np.std(np.array(array))
            indicateurs[i].addIndicator(hyperP,value,"bollinger_high")

    
    if not(indicateurs[-1].calculated(hyperP,'bollinger_low')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=20:
                value = indicateurs[i].get_Indicator(hyperP,"TP_moy_20")
            else:
                array = [indicateurs[i-j].get_Indicator(hyperP,"TPV") for j in range(20)]
                value = indicateurs[i].get_Indicator(hyperP,"TP_moy_20") - 2*np.std(np.array(array))
            indicateurs[i].addIndicator(hyperP,value,"bollinger_low")
    
    if not(indicateurs[-1].calculated(hyperP,'touch_bollinger')):
        for i in range(len(indicateurs)):
            #definition deriv?? 1
            if i<=25:
                value = 0
            else:
                if indicateurs[i].get_Indicator(hyperP,"closeV")<indicateurs[i].get_Indicator(hyperP,"bollinger_low"):
                    value = -1
                elif indicateurs[i].get_Indicator(hyperP,"closeV")>indicateurs[i].get_Indicator(hyperP,"bollinger_high"):
                    value = 1
                else :
                    value = 0
            indicateurs[i].addIndicator(hyperP,value,"touch_bollinger")
    
    

    if not(indicateurs[-1].calculated(hyperP,'sign_diff_moy')):
        for i in range(len(indicateurs)):
            value=(np.sign(indicateurs[i].get_Indicator(hyperP,"close_moy_B")-indicateurs[i].get_Indicator(hyperP,"close_moy_A")))
            indicateurs[i].addIndicator(hyperP,value,"sign_diff_moy")
            
    if not(indicateurs[-1].calculated(hyperP,'derive_diff_close')):
        for i in range(len(indicateurs)):
            if i<=0:
                value=0
            else:
                value=(indicateurs[i-1].get_Indicator(hyperP,"close_moy_B")+indicateurs[i].get_Indicator(hyperP,"close_moy_A")-indicateurs[i].get_Indicator(hyperP,"close_moy_B")-indicateurs[i-1].get_Indicator(hyperP,"close_moy_A"))
            indicateurs[i].addIndicator(hyperP,value,"derive_diff_close")  
            
    if not(indicateurs[-1].calculated(hyperP,'derive_moy_1')):
        for i in range(len(indicateurs)):
            if i<=0:
                value=indicateurs[i].get_Indicator(hyperP,"deriv??")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"deriv??"), indicateurs[i-1].get_Indicator(hyperP,"derive_moy_1"), hyperP["Theta_der"])
            indicateurs[i].addIndicator(hyperP,value,"derive_moy_1")
            
    if not(indicateurs[-1].calculated(hyperP,'derive_moy_2')):
        for i in range(len(indicateurs)):
            if i<=0:
                value=indicateurs[i].get_Indicator(hyperP,"deriv??")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"deriv??"), indicateurs[i-1].get_Indicator(hyperP,"derive_moy_2"), hyperP["Theta_der2"])
            indicateurs[i].addIndicator(hyperP,value,"derive_moy_2")

    if not(indicateurs[-1].calculated(hyperP,'std_A')):
        for i in range(len(indicateurs)):
            if i<=6:
                value=1
            else:
                array = [indicateurs[i-j].get_Indicator(hyperP,"closeV") for j in range(6)]
                value = np.std(np.array(array))
                
            indicateurs[i].addIndicator(hyperP,value,"std_A")
            
            
    if not(indicateurs[-1].calculated(hyperP,'std_B')):
        for i in range(len(indicateurs)):
            if i<=45:
                value=1
            else:
                array = [indicateurs[i-j].get_Indicator(hyperP,"closeV") for j in range(45)]
                value = np.std(np.array(array))
                
            indicateurs[i].addIndicator(hyperP,value,"std_B")

    if not(indicateurs[-1].calculated(hyperP,'RSI')):
        for i in range(len(indicateurs)):
            if i<=0:
                value=50
            else:
                value=(computeRSI([indicateurs[i-j].get_Indicator(hyperP,"deriv??") for j in range(hyperP["Theta_RSI"])],d=hyperP["Theta_RSI"])- 50)
                if value >35:
                    value=(value-20)*2
                elif value >=20:
                    value=(value-20)
                elif value >-20:
                    value=0
                elif value >=-35:
                    value=(value+20)
                else:
                    value=(value+20)*2
            indicateurs[i].addIndicator(hyperP,value/6,"RSI"),#/6
            
    if not(indicateurs[-1].calculated(hyperP,'croisement_moyennes')):
        for i in range(len(indicateurs)):
            if i<=5:
                value=0
            else:
                dernier_croisement=[indicateurs[i-j].get_Indicator(hyperP,"croisement_moyennes") for j in range(1,5)]
                if (indicateurs[i].get_Indicator(hyperP,"sign_diff_moy")*indicateurs[i-2].get_Indicator(hyperP,"sign_diff_moy")==-1 and np.sum(np.abs(dernier_croisement))==0):
                    value=(10*indicateurs[i].get_Indicator(hyperP,"sign_diff_moy"))
                else:
                    value=(0)
            indicateurs[i].addIndicator(hyperP,value,"croisement_moyennes")
  
    return indicateurs



class indices():
    
    def __init__(self):
        #parametres influant le calcul des indicateurs
        
        self.list=[]
        
    def add_hyper_param(self,param : dict,data):
        idd,ratio=createIndicator(data)

        
params_1 = {
            "Theta" : 4,
            "Theta_bis" : 5,
            "Theta_der" : 3,
        }

params_2 = {
            "Theta" : 2,
            "Theta_bis" : 5,
            "Theta_der" : 3,
        }

params_3 = {
            "Theta" : 5,
            "Theta_bis" : 5,
            "Theta_der" : 6,
        }
params_4 = {
            "Theta" : 1,
            "Theta_bis" : 1,
            "Theta_der" : 3,
        }


# =============================================================================
# idd = ['RSI','moy1','moy2','derive']
# val = [1,2,3,4]
# indicateur=indicator(params_1,idd,val)
# for i in range(4):
#     indicateur.addIndicator(params_2, idd[i], val[i]+1)
#     indicateur.addIndicator(params_3, idd[i], val[i]/23)
#     indicateur.addIndicator(params_4, idd[i], val[i]*10)
# 
# keys=indicateur.getParamsKeys()
# print(len(keys))
# print(keys)
# 
# 
# =============================================================================

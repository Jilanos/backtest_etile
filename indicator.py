
import numpy as np
import matplotlib.pyplot as plt


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

def findLastExtrema(array, val_min, val_max, maxim : bool = True, n : int = 2):
    array_inverted_0 = np.copy(array)#[::-1]
    indice = 0
    found_at_least_one = False
    best_one = False
    for j in range(n):
        if array_inverted_0[j]>val_max :
            return val_max
        else :
            if  (array_inverted_0[j]>np.max([val_min,best_one]) and found_at_least_one) or (array_inverted_0[j]>val_min and not(found_at_least_one)):
                best_one = array_inverted_0[j]
                found_at_least_one = True
    indice = n
    while (array_inverted_0[indice]<=val_max) and indice<len(array_inverted_0)-1-n:
        if (array_inverted_0[indice]>np.max([best_one,val_min])and found_at_least_one) or  (array_inverted_0[indice]>val_min and not(found_at_least_one)):
            if extremaLocal(array_inverted_0[indice-n:indice+n+1], maxim):
                best_one = array_inverted_0[indice]
                found_at_least_one = True
        indice += 1
    if not(found_at_least_one):
        return val_max
    else:
        return best_one

    
    

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
    #2derivé 1, 3dérivé 2, 4prix moyenné 1, 5prix moyenné 2, 6RSI, 7signe diff moy, 8dérivé diff moy
    #9dérivé 1 moy, 10dérivé 2 moy,11 signe écart
    
    
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
    ratio = np.max(closeData)
    closeData /= ratio
    highData /= ratio
    openData /= ratio
    lowData /= ratio    
    
    for i in range(len(closeData)) :
        volumeV=volumeData[i]
        indicateur[i].addIndicator(params,closeData[i] ,"closeV")
        indicateur[i].addIndicator(params,highData[i] ,"highV")
        indicateur[i].addIndicator(params,lowData[i] ,"lowV")
        indicateur[i].addIndicator(params,openData[i] ,"openV")
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
    if not(indicateurs[-1].calculated(hyperP,'derivé')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                indicateurs[i].addIndicator(hyperP,0,"derivé")
            else:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")-indicateurs[i-1].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"derivé")
    
    if not(indicateurs[-1].calculated(hyperP,'dérivé2')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                indicateurs[i].addIndicator(hyperP,0,"dérivé2")
            else:
                value=indicateurs[i].get_Indicator(hyperP,"derivé")-indicateurs[i-1].get_Indicator(hyperP,"derivé")
                indicateurs[i].addIndicator(hyperP,value,"dérivé2")    
                
    if not(indicateurs[-1].calculated(hyperP,'close_moy_A')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"close_moy_A")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"closeV"), indicateurs[i-1].get_Indicator(hyperP,"close_moy_A"), hyperP["Theta"])
                indicateurs[i].addIndicator(hyperP,value,"close_moy_A")

    if not(indicateurs[-1].calculated(hyperP,'close_moy_B')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"close_moy_B")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"closeV"), indicateurs[i-1].get_Indicator(hyperP,"close_moy_B"), hyperP["Theta"]*hyperP["Theta_bis"])
                indicateurs[i].addIndicator(hyperP,value,"close_moy_B")
                                
    if not(indicateurs[-1].calculated(hyperP,'ema12')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"ema12")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"closeV"), indicateurs[i-1].get_Indicator(hyperP,"ema12"), 12)
                indicateurs[i].addIndicator(hyperP,value,"ema12")

    if not(indicateurs[-1].calculated(hyperP,'ema26')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"ema26")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"closeV"), indicateurs[i-1].get_Indicator(hyperP,"ema26"), 26)
                indicateurs[i].addIndicator(hyperP,value,"ema26")
                
                
    if not(indicateurs[-1].calculated(hyperP,'MACD')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            indicateurs[i].addIndicator(hyperP,indicateurs[i].get_Indicator(hyperP,"ema12")-indicateurs[i].get_Indicator(hyperP,"ema26"),"MACD")
                
                            
    if not(indicateurs[-1].calculated(hyperP,'MACD_signal')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                value=indicateurs[i].get_Indicator(hyperP,"MACD")
                indicateurs[i].addIndicator(hyperP,value,"MACD_signal")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"MACD"), indicateurs[i-1].get_Indicator(hyperP,"MACD_signal"), 9)
                indicateurs[i].addIndicator(hyperP,value,"MACD_signal")
                 
                            
    if not(indicateurs[-1].calculated(hyperP,'MACD_crossing')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=5:
                value=0
            else:
                dernier_croisement=[indicateurs[i-j].get_Indicator(hyperP,"MACD_crossing") for j in range(1,5)]
                if ((indicateurs[i].get_Indicator(hyperP,"MACD_signal")-indicateurs[i].get_Indicator(hyperP,"MACD"))<0 and (indicateurs[i-2].get_Indicator(hyperP,"MACD_signal")-indicateurs[i-2].get_Indicator(hyperP,"MACD"))>0):
                    value=(1)
                elif ((indicateurs[i].get_Indicator(hyperP,"MACD_signal")-indicateurs[i].get_Indicator(hyperP,"MACD"))>0 and (indicateurs[i-2].get_Indicator(hyperP,"MACD_signal")-indicateurs[i-2].get_Indicator(hyperP,"MACD"))<0):
                    value=(-1)
                else:
                    value=(0)
            indicateurs[i].addIndicator(hyperP,value,"MACD_crossing")
                
            
                
    if not(indicateurs[-1].calculated(hyperP,'close_moy_C')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                value=indicateurs[i].get_Indicator(hyperP,"closeV")
                indicateurs[i].addIndicator(hyperP,value,"close_moy_C")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"closeV"), indicateurs[i-1].get_Indicator(hyperP,"close_moy_C"), hyperP["Theta_C"])
                indicateurs[i].addIndicator(hyperP,value,"close_moy_C")
    
    if not(indicateurs[-1].calculated(hyperP,'der_moy_C')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                indicateurs[i].addIndicator(hyperP,0,"der_moy_C")
            else:
                value=indicateurs[i].get_Indicator(hyperP,"close_moy_C")-indicateurs[i-1].get_Indicator(hyperP,"close_moy_C")
                indicateurs[i].addIndicator(hyperP,value,"der_moy_C")   

        
    if not(indicateurs[-1].calculated(hyperP,'maxi_proche')):
        for i in range(len(indicateurs)):
            if i<110 : 
                value = (hyperP["SL_max"])
            else :
                val = indicateurs[i].get_Indicator(hyperP,"closeV")
                value_int = [(indicateurs[i-j].get_Indicator(hyperP,"highV")-val)/val*100 for j in range(108)]
                value = findLastExtrema(value_int, hyperP["SL_min"], hyperP["SL_max"], maxim = True, n = 2)

            indicateurs[i].addIndicator(hyperP,value,"maxi_proche")   
            
        
    if not(indicateurs[-1].calculated(hyperP,'mini_proche')):
        for i in range(len(indicateurs)):
            if i<110 : 
                value = (hyperP["SL_max"])
            else :
                val = indicateurs[i].get_Indicator(hyperP,"closeV")
                value_int = [-(indicateurs[i-j].get_Indicator(hyperP,"lowV")-val)/val*100 for j in range(108)]
                value = findLastExtrema(value_int, hyperP["SL_min"], hyperP["SL_max"], maxim = True, n = 2)

            indicateurs[i].addIndicator(hyperP,value,"mini_proche")               
   
    
    # if not(indicateurs[-1].calculated(hyperP,'maxi_proche')):
    #     for i in range(len(indicateurs)):
    #         close_act = indicateurs[i].get_Indicator(hyperP,"closeV")
    #         value_int = np.max([indicateurs[i-j].get_Indicator(hyperP,"highV") for j in range(24)])
    #         value = abs(value_int-close_act)/close_act*100
    #         if value > hyperP["SL_max"] or value < hyperP["SL_min"]:
    #             value = (hyperP["SL_max"] + hyperP["SL_min"])/2
    #         indicateurs[i].addIndicator(hyperP,value,"maxi_proche")   
     
        
    # if not(indicateurs[-1].calculated(hyperP,'mini_proche')):
    #     for i in range(len(indicateurs)):
    #         close_act = indicateurs[i].get_Indicator(hyperP,"closeV")
    #         value_int = np.min([indicateurs[i-j].get_Indicator(hyperP,"lowV") for j in range(24)])
    #         value = abs(value_int-close_act)/close_act*100
    #         if value > hyperP["SL_max"] or value < hyperP["SL_min"]:
    #             value = (hyperP["SL_max"] + hyperP["SL_min"])/2
    #         indicateurs[i].addIndicator(hyperP,value,"mini_proche")   
            
        
    if not(indicateurs[-1].calculated(hyperP,'TPV')):
        for i in range(len(indicateurs)):
            value = (indicateurs[i].get_Indicator(hyperP,"closeV")+indicateurs[i].get_Indicator(hyperP,"highV")+indicateurs[i].get_Indicator(hyperP,"lowV"))/3
            indicateurs[i].addIndicator(hyperP,value,"TPV")      
            
    if not(indicateurs[-1].calculated(hyperP,'TR')):
        for i in range(len(indicateurs)):
            if i<=2:
                value = 0
            else:
                v1 = indicateurs[i].get_Indicator(hyperP,"highV")-indicateurs[i].get_Indicator(hyperP,"lowV")
                v2 = abs(indicateurs[i].get_Indicator(hyperP,"highV")-indicateurs[i-1].get_Indicator(hyperP,"closeV"))
                v3 = abs(indicateurs[i].get_Indicator(hyperP,"lowV")-indicateurs[i-1].get_Indicator(hyperP,"closeV"))
                value = max(v1,v2,v3)#*ratio
            indicateurs[i].addIndicator(hyperP,value,"TR")              
        
    if not(indicateurs[-1].calculated(hyperP,'TR_moy_14')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=15:
                value=indicateurs[i].get_Indicator(hyperP,"TR")
            elif i==16:
                array = [indicateurs[i-j].get_Indicator(hyperP,"TR") for j in range(14)]
                value = np.sum(np.array(array))
            else:
                value = indicateurs[i].get_Indicator(hyperP,"TR") + (13./14)*indicateurs[i-1].get_Indicator(hyperP,"TR_moy_14")
            indicateurs[i].addIndicator(hyperP,value,"TR_moy_14")         
        
    if not(indicateurs[-1].calculated(hyperP,'TP_moy_20')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=20:
                value=indicateurs[i].get_Indicator(hyperP,"TPV")
                indicateurs[i].addIndicator(hyperP,value,"TP_moy_20")
            else:
                array = [indicateurs[i-j].get_Indicator(hyperP,"TPV") for j in range(20)]
                value = np.mean(np.array(array))
                indicateurs[i].addIndicator(hyperP,value,"TP_moy_20")
    
    if not(indicateurs[-1].calculated(hyperP,'bollinger_high')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=20:
                value = indicateurs[i].get_Indicator(hyperP,"TP_moy_20")
            else:
                array = [indicateurs[i-j].get_Indicator(hyperP,"TPV") for j in range(20)]
                value = indicateurs[i].get_Indicator(hyperP,"TP_moy_20") + 2*np.std(np.array(array))
            indicateurs[i].addIndicator(hyperP,value,"bollinger_high")

    
    if not(indicateurs[-1].calculated(hyperP,'bollinger_low')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=20:
                value = indicateurs[i].get_Indicator(hyperP,"TP_moy_20")
            else:
                array = [indicateurs[i-j].get_Indicator(hyperP,"TPV") for j in range(20)]
                value = indicateurs[i].get_Indicator(hyperP,"TP_moy_20") - 2*np.std(np.array(array))
            indicateurs[i].addIndicator(hyperP,value,"bollinger_low")
    
    if not(indicateurs[-1].calculated(hyperP,'touch_bollinger')):
        for i in range(len(indicateurs)):
            #definition derivé 1
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
            
    if not(indicateurs[-1].calculated(hyperP,'MFI')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=14:
                value = 50
            else:
                array_TPV = np.array([indicateurs[i-j].get_Indicator(hyperP,"TPV") for j in range(14)])
                array_sig_TPV = np.array([np.sign(indicateurs[i-j].get_Indicator(hyperP,"TPV")-indicateurs[i-j-1].get_Indicator(hyperP,"TPV")) for j in range(14)])
                array_vol = np.array([indicateurs[i-j-1].get_Indicator(hyperP,"volume") for j in range(14)])
                money_flow = array_TPV*array_sig_TPV*array_vol
                pos, neg = 0, 0
                for j in range(14):
                    if money_flow[j] >0:
                        pos += money_flow[j]
                    else:
                        neg -= money_flow[j]
                value = 100 - 100/(1+ (pos/neg))
            indicateurs[i].addIndicator(hyperP,value,"MFI")  
            
    if not(indicateurs[-1].calculated(hyperP,'CCI')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=22:
                value = 0
            else:
                array_TPV = np.array([indicateurs[i-j].get_Indicator(hyperP,"TPV") for j in range(20)])
                moy = indicateurs[i].get_Indicator(hyperP,"TP_moy_20")
                std = np.std(array_TPV-moy)
                value = (indicateurs[i].get_Indicator(hyperP,"TPV")-moy)/(0.015*std)
            indicateurs[i].addIndicator(hyperP,value,"CCI")
                 
    if not(indicateurs[-1].calculated(hyperP,'DM+')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=1:
                DM_p_fin = 0
                DM_m_fin = 0
            else:
                DM_p = indicateurs[i].get_Indicator(hyperP,"highV")-indicateurs[i-1].get_Indicator(hyperP,"highV")
                DM_m = indicateurs[i-1].get_Indicator(hyperP,"lowV")-indicateurs[i].get_Indicator(hyperP,"lowV")
                DM_p_fin = 0
                DM_m_fin = 0
                for j in range(14):
                    if DM_p>DM_m and DM_p>0 :
                        DM_p_fin = DM_p
                    if DM_p<DM_m and DM_m>0 :
                        DM_m_fin = DM_m 
            indicateurs[i].addIndicator(hyperP,DM_p_fin,"DM+")
            indicateurs[i].addIndicator(hyperP,DM_m_fin,"DM-")
    
    if not(indicateurs[-1].calculated(hyperP,'DM_moy-_14')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=15:
                DM_moyM=indicateurs[i].get_Indicator(hyperP,"DM-")
                DM_moyP=indicateurs[i].get_Indicator(hyperP,"DM+")
            elif i==16:
                array = [indicateurs[i-j].get_Indicator(hyperP,"DM-") for j in range(14)]
                DM_moyM = np.sum(np.array(array))
                array = [indicateurs[i-j].get_Indicator(hyperP,"DM+") for j in range(14)]
                DM_moyP = np.sum(np.array(array))
            else:
                DM_moyM = indicateurs[i].get_Indicator(hyperP,"DM-") + (13./14)*indicateurs[i-1].get_Indicator(hyperP,"DM_moy-_14")
                DM_moyP = indicateurs[i].get_Indicator(hyperP,"DM+") + (13./14)*indicateurs[i-1].get_Indicator(hyperP,"DM_moy+_14")
            indicateurs[i].addIndicator(hyperP,DM_moyP,"DM_moy+_14")       
            indicateurs[i].addIndicator(hyperP,DM_moyM,"DM_moy-_14")  
            
         
            
    if not(indicateurs[-1].calculated(hyperP,'DI+')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i <=17 :
                valueP = 1
                valueM = 1
            else:
                valueP = indicateurs[i].get_Indicator(hyperP,"DM_moy+_14")/indicateurs[i].get_Indicator(hyperP,"TR_moy_14")*100
                valueM = indicateurs[i].get_Indicator(hyperP,"DM_moy-_14")/indicateurs[i].get_Indicator(hyperP,"TR_moy_14")*100
            indicateurs[i].addIndicator(hyperP,valueP,"DI+")  
            indicateurs[i].addIndicator(hyperP,valueM,"DI-")   
            
    if not(indicateurs[-1].calculated(hyperP,'DX')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            value = abs(indicateurs[i].get_Indicator(hyperP,"DI+")-indicateurs[i].get_Indicator(hyperP,"DI-"))/abs(indicateurs[i].get_Indicator(hyperP,"DI+")+indicateurs[i].get_Indicator(hyperP,"DI-"))*100
            indicateurs[i].addIndicator(hyperP,value,"DX")              
    
            
    if not(indicateurs[-1].calculated(hyperP,'ADX')):
        for i in range(len(indicateurs)):
            #definition derivé 1
            if i<=14:
                value = 0
            elif i==15:
                value = np.mean(np.array([indicateurs[i-j].get_Indicator(hyperP,"DX") for j in range(14)]))
            else:
                value = (indicateurs[i-1].get_Indicator(hyperP,"ADX")*13+indicateurs[i].get_Indicator(hyperP,"DX"))/14
            indicateurs[i].addIndicator(hyperP,value,"ADX")
            
            
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
                value=indicateurs[i].get_Indicator(hyperP,"derivé")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"derivé"), indicateurs[i-1].get_Indicator(hyperP,"derive_moy_1"), hyperP["Theta_der"])
            indicateurs[i].addIndicator(hyperP,value,"derive_moy_1")
            
    if not(indicateurs[-1].calculated(hyperP,'derive_moy_2')):
        for i in range(len(indicateurs)):
            if i<=0:
                value=indicateurs[i].get_Indicator(hyperP,"derivé")
            else:
                value=calculate_ema(indicateurs[i].get_Indicator(hyperP,"derivé"), indicateurs[i-1].get_Indicator(hyperP,"derive_moy_2"), hyperP["Theta_der2"])
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
                value=(computeRSI([indicateurs[i-j].get_Indicator(hyperP,"derivé") for j in range(hyperP["Theta_RSI"])],d=hyperP["Theta_RSI"])- 50)
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


            
    if not(indicateurs[-1].calculated(hyperP,'average_gain')):
        for i in range(len(indicateurs)):
            if i<=14:
                value_gain=0
                value_loss=0
            elif i <=17 :
                d = [indicateurs[i-j].get_Indicator(hyperP,"derivé") for j in range(14)]
                value_gain = 0
                value_loss = 0
                for elt in d:
                    if elt>0:
                        value_gain += elt
                    else :
                        value_loss -= elt
            else:
                d = indicateurs[i].get_Indicator(hyperP,"derivé")
                if d > 0:
                    value_gain = (indicateurs[i-1].get_Indicator(hyperP,"average_gain") *13. +d )/14.
                    value_loss = indicateurs[i-1].get_Indicator(hyperP,"average_perte")*13./14.
                else :
                    value_gain = (indicateurs[i-1].get_Indicator(hyperP,"average_gain") *13.)/14.
                    value_loss = (indicateurs[i-1].get_Indicator(hyperP,"average_perte")*13. - d)/14.
            indicateurs[i].addIndicator(hyperP,value_gain,"average_gain")
            indicateurs[i].addIndicator(hyperP,value_loss,"average_perte")
    
    if not(indicateurs[-1].calculated(hyperP,'RSI_true')):
        for i in range(len(indicateurs)):
            if i<=18:
                value=50
            else:
                value=100 - 100./(1+indicateurs[i].get_Indicator(hyperP,"average_gain")/indicateurs[i].get_Indicator(hyperP,"average_perte"))
            indicateurs[i].addIndicator(hyperP,value,"RSI_true"),#/6
            
    if not(indicateurs[-1].calculated(hyperP,'RSI_stoch')):
        for i in range(len(indicateurs)):
            if i<=30:
                value = 0
            else:
                array = [indicateurs[i-j].get_Indicator(hyperP,"RSI_true") for j in range(14)]
                mini = min(array)
                maxi = max(array)
                RSI = indicateurs[i].get_Indicator(hyperP,"RSI_true")
                value = (RSI-mini)/(maxi-mini)
            indicateurs[i].addIndicator(hyperP,value,"RSI_stoch")


    if not(indicateurs[-1].calculated(hyperP,'RSI_k')):
        for i in range(len(indicateurs)):
            if i<=30:
                value = 0
            elif i==31:
                array = [indicateurs[i-j].get_Indicator(hyperP,"RSI_stoch") for j in range(2)]
                value = np.mean(np.array(array))
            else:
                value = calculate_ema(indicateurs[i].get_Indicator(hyperP,"RSI_stoch"), indicateurs[i-1].get_Indicator(hyperP,"RSI_k"), 1)
            indicateurs[i].addIndicator(hyperP,value,"RSI_k")
            

    if not(indicateurs[-1].calculated(hyperP,'RSI_d')):
        for i in range(len(indicateurs)):
            if i<=30:
                value = 0
            elif i==31:
                array = [indicateurs[i-j].get_Indicator(hyperP,"RSI_stoch") for j in range(3)]
                value = np.mean(np.array(array))
            else:
                value = calculate_ema(indicateurs[i].get_Indicator(hyperP,"RSI_stoch"), indicateurs[i-1].get_Indicator(hyperP,"RSI_d"), 3)
            indicateurs[i].addIndicator(hyperP,value,"RSI_d")
            
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

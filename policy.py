
import numpy as np
import abc # Abstract Basic Class
import random
from bet import Bet # For type checking
from optuna import Trial # for type checking
from regularizer import Regularizer_EMA, Regularizer_AvgStd, Derivator
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Policy() :

    """ Parametrized policies define what action should be taken """

    params = None # All parametrized policies must have parameters

    @abc.abstractmethod
    def sample(self) : # All parametrized policies must be sampleable uniformely ...
        pass


    @abc.abstractmethod
    def sample(self, trial : Trial) : # ... and by an Optuna trial
        pass


    @abc.abstractmethod
    def betToMake(self) : # All policies must predict what bet to make
        pass



class Policy_01(Policy) :

    """ First serious policy implemented """

    def __init__(self) :
        self.ratio=None
        self.val = []
        self.count=0
        self.entryPrice=[]
        self.wins=[]
        self.loss=[]
        self.moneyToBet = 100
        self.params= {}
        
        self.weight=[]
        


    def sampleFromTrial(self, trial : Trial) :
            # Sample the policy space and return the associated params

        self.params = {
            "TP" : trial.suggest_int("TP", 5, 25)/10,#trial.suggest_float('TP',0.2,5),
            "SL" : trial.suggest_int("SL", 2, 8)/10,#trial.suggest_float('SL',1,8),
            "revert" : trial.suggest_int("revert", -1, 1,2),
            # "weight_0" : 0,#trial.suggest_float('weight_0', 0, 1),#0,
            # "weight_1" : 0,#trial.suggest_float('weight_1', -1, 1),
            # "weight_2" :0,#ttrial.suggest_float('weight_2', -1, 1),
            # "weight_3" : 0,#trial.suggest_float('weight_3', -1, 1),
            # "weight_4" : 0,#trial.suggest_float('weight_4', -1, 1),
            # "weight_5" : 0,#trial.suggest_float('weight_5', -1, 1),
            # "weight_6" : 0,#trial.suggest_float('weight_6', -1, 1),
            # "Theta":    3,#trial.suggest_int("Theta", 3, 7,2),
            # "Theta_bis":    3,#trial.suggest_int("Theta_bis", 3, 7,2),
            # "Theta_RSI":    14,
            "Theta_C" :    trial.suggest_int("Theta_C", 5, 105,10),
            # "normFactor" : 0,#trial.suggest_float('normFactor', 0.001, 100),
        }


    def betToMake(self, priceRegularized : float,indic, highestRegularized : float, lowestRegularized : float, volume : float, bet : Bet) :
        self.count+=1
        if (bet != None) :
            return "", None, None, None
        #print(indic)
# =============================================================================
#         key_derive_diff_moy='derive_diff_close'+str(self.params["Theta"])+'_'+str(self.params["Theta_bis"])
#         key_signe_diff_moy='sign_diff_moy'+str(self.params["Theta"])+'_'+str(self.params["Theta_bis"])
#         key_croisement_moyennes='croisement_moyennes'+str(self.params["Theta"])+'_'+str(self.params["Theta_bis"])
# =============================================================================
        param = {
            "Theta" : 3,
            "Theta_bis" : 3,
            "Theta_der" : 3,
            "Theta_der2" : 3,
            "Theta_RSI" : 14,
            "Theta_C" : self.params["Theta_C"]}
        val = 0
        if np.sign(indic.get_Indicator(param, "touch_bollinger")) ==self.params["revert"] and np.sign(indic.get_Indicator(param, "der_moy_C")) ==1:
            #print("try long : ",indic.get_Indicator(param, "der_moy_C"),self.params["seuil_der"])
            #if abs(indic.get_Indicator(param, "der_moy_C"))*1000>self.params["seuil_der"]:
            val = 1
                #print("Sucess Long")
        elif np.sign(indic.get_Indicator(param, "touch_bollinger")) ==-self.params["revert"] and np.sign(indic.get_Indicator(param, "der_moy_C")) ==-1:
            #print("try long : ",indic.get_Indicator(param, "der_moy_C"),self.params["seuil_der"])
            #if abs(indic.get_Indicator(param, "der_moy_C"))*1000>self.params["seuil_der"]:
            val= -1
                #print("Sucess Short")

        param_pese=["derivé","dérivé2","RSI","derive_diff_close","sign_diff_moy","volume","croisement_moyennes"]
        
        poids=[1 for i in range(7)]

        self.val.append(val)
        if val == 1 :
            self.entryPrice = [self.count-1, priceRegularized]
            self.weight.append(poids)
            return "Long", self.moneyToBet, self.params["TP"], self.params["SL"]
        elif val == -1:
            self.entryPrice = [self.count-1, priceRegularized]
            self.weight.append(poids)
            return "Short", self.moneyToBet, self.params["TP"], self.params["SL"]
        else :
            return "", None, None, None
        print("priceRegularized, highestRegularized, lowestRegularized : ", priceRegularized, highestRegularized, lowestRegularized)


    def addTrade(self, win : bool, val : float) :
        if win == True :
            self.wins.append([self.entryPrice[0], self.entryPrice[1], self.count-1, val])
        else :
            self.loss.append([self.entryPrice[0], self.entryPrice[1], self.count-1, val])
            
    def plot(self, closeSequence,openSequence,highSequence,lowSequence,folder_name,ratio,name:str, ignoreTimer : int = 0):
        closeSequence = np.array(closeSequence)
        openSequence = np.array(openSequence)
        highSequence = np.array(highSequence)
        lowSequence = np.array(lowSequence)
        fig, ax1 = plt.subplots(figsize=(20,11))
        plt.xlabel('TF',fontsize=20)
        plt.ylabel('Price',fontsize=20)
        plt.title("Bitcoin Price",fontsize=30, fontweight = 'bold')
        # print("len : ",len(highSequence))
        # print("close : ",closeSequence[:30])
        # print("open : ",openSequence[:30])
        for j in range(len(highSequence)):
            if closeSequence[j]>openSequence[j]:
                c='green'
            else:
                c='red'
            plt.plot([j+0.5,j+0.5],[lowSequence[j],highSequence[j]], c)
            ax1.add_patch(Rectangle((j, min(closeSequence[j],openSequence[j])), 1, max(closeSequence[j],openSequence[j])-min(closeSequence[j],openSequence[j]),facecolor =c))

        #plt.plot(np.array(closeSequence[ignoreTimer:]), label='Closing Prices')
# =============================================================================
#         print("WINS")
#         print(self.wins)
#         print("LOSS")
#         print(self.loss)
# =============================================================================
        for a in self.wins:
            plt.scatter(a[0]-ignoreTimer,a[1],s=60,marker='x',c='g')#length_includes_head=True,head_width=5,head_length=30
            plt.arrow(a[0]-ignoreTimer,a[1],a[2]-a[0],a[3]-a[1],color='g')
        for a in self.loss:
            plt.scatter(a[0]-ignoreTimer,a[1],s=60,marker='x',c='r')
            plt.arrow(a[0]-ignoreTimer,a[1],a[2]-a[0],a[3]-a[1],color='r')
        plt.savefig(folder_name+name)
        #plt.close()



class Policy_02(Policy) :

    """ First serious policy implemented """

    def __init__(self) :
        self.ratio=None
        self.val = []
        self.count=0
        self.entryPrice=[]
        self.wins=[]
        self.loss=[]
        self.moneyToBet = 100.
        self.riskPourcent = 1.
        self.params= {}
        
        self.weight=[]
        


    def sampleFromTrial(self, trial : Trial) :
            # Sample the policy space and return the associated params

        self.params = {
            "TP" : trial.suggest_int("TP", 8, 35)/10.,#trial.suggest_float('TP',0.2,5),
            "RR" : trial.suggest_int("RR", 12, 35)/10.,#trial.suggest_float('TP',0.2,5),
            #"SL" : trial.suggest_int("SL", 2, 8)/10,#trial.suggest_float('SL',1,8),
            #"buy_adx" : trial.suggest_int("buy_adx", 20, 50),#IntParameter(20, 50, default=32, space='buy')
            #"buy_fastd" : trial.suggest_int("buy_fastd", 15, 45),#buy_fastd = IntParameter(15, 45, default=30, space='buy')
            #"buy_fastk" : trial.suggest_int("buy_fastk", 15, 45),#buy_fastk = IntParameter(15, 45, default=26, space='buy')
            #"buy_mfi" :    trial.suggest_int("buy_mfi", 10, 25),
            #"sell_adx" :    trial.suggest_int("sell_adx", 50, 100),#sell_adx = IntParameter(50, 100, default=53, space='sell')
            #"sell_cci" :    trial.suggest_int("sell_cci", 100, 200),#    sell_cci = IntParameter(100, 200, default=183, space='sell')
            #"sell_fastd" :    trial.suggest_int("sell_fastd", 50, 100),#    sell_fastd = IntParameter(50, 100, default=79, space='sell')
            # "SL_min" :    trial.suggest_int("SL_min", 1, 3, 2)/10.,#   sell_fastk = IntParameter(50, 100, default=70, space='sell')
            # "SL_max" :    trial.suggest_int("SL_max", 50, 100, 25)/100.,#   sell_fastk = IntParameter(50, 100, default=70, space='sell')
            "SL_min" :    0.2,#   sell_fastk = IntParameter(50, 100, default=70, space='sell')
            "SL_max" :    1,#   sell_fastk = IntParameter(50, 100, default=70, space='sell')
            #"sell_fastk" :    trial.suggest_int("sell_fastk", 50, 100),#   sell_fastk = IntParameter(50, 100, default=70, space='sell')
            "buy_RSI" :    trial.suggest_int("buy_RSI", 0, 100),#   sell_fastk = IntParameter(50, 100, default=70, space='sell')
            "sell_RSI" :    trial.suggest_int("sell_RSI", 0, 100)#   sell_fastk = IntParameter(50, 100, default=70, space='sell')
            #"sell_mfi" :    trial.suggest_int("sell_mfi", 75, 100)#    sell_mfi = IntParameter(75, 100, default=92, space='sell')
        }


    def betToMake(self, priceRegularized : float,indic, highestRegularized : float, lowestRegularized : float, volume : float, bet : Bet) :
        self.count+=1
        if (bet != None) :
            return "", None, None, None
        #print(indic)
# =============================================================================
#         key_derive_diff_moy='derive_diff_close'+str(self.params["Theta"])+'_'+str(self.params["Theta_bis"])
#         key_signe_diff_moy='sign_diff_moy'+str(self.params["Theta"])+'_'+str(self.params["Theta_bis"])
#         key_croisement_moyennes='croisement_moyennes'+str(self.params["Theta"])+'_'+str(self.params["Theta_bis"])
# =============================================================================
        param = {
            "Theta" : 3,
            "Theta_bis" : 3,
            "Theta_der" : 3,
            "Theta_der2" : 3,
            "Theta_RSI" : 14,
            "Theta_C" : 200,
            "SL_max" : self.params["SL_max"],
            "SL_min" : self.params["SL_min"]}
        val = 0
        if indic.get_Indicator(param, "MACD_crossing")==1 and indic.get_Indicator(param, "RSI_stoch")< self.params["buy_RSI"]:
            if indic.get_Indicator(param, "closeV")> indic.get_Indicator(param, "close_moy_C"):
                val = 1
                #print("Sucess Long")
        elif indic.get_Indicator(param, "MACD_crossing")==-1 and indic.get_Indicator(param, "RSI_stoch")> self.params["sell_RSI"]:
            if indic.get_Indicator(param, "closeV")< indic.get_Indicator(param, "close_moy_C"):
                val= -1
                #print("Sucess Short")

        param_pese=["derivé","dérivé2","RSI","derive_diff_close","sign_diff_moy","volume","croisement_moyennes"]
        
        poids=[1 for i in range(7)]

        self.val.append(val)
        if val == 1 :
            self.entryPrice = [self.count-1, priceRegularized]
            self.weight.append(poids)
            return "Long", self.riskPourcent*100/indic.get_Indicator(param, "mini_proche"), self.params["RR"]*indic.get_Indicator(param, "mini_proche"), indic.get_Indicator(param, "mini_proche")
            #return "Long", self.moneyToBet, self.params["TP"], self.params["SL"]
        elif val == -1:
            self.entryPrice = [self.count-1, priceRegularized]
            self.weight.append(poids)
            return "Short", self.riskPourcent*100/indic.get_Indicator(param, "maxi_proche"), self.params["RR"]*indic.get_Indicator(param, "maxi_proche"), indic.get_Indicator(param, "maxi_proche")
            #return "Short", self.moneyToBet, self.params["TP"], self.params["SL"]
        else :
            return "", None, None, None
        print("priceRegularized, highestRegularized, lowestRegularized : ", priceRegularized, highestRegularized, lowestRegularized)


    def addTrade(self, win : bool, val : float, val_other : float) :
        if win == True :
            self.wins.append([self.entryPrice[0]+0.5, self.entryPrice[1], self.count+0.5, val, val_other])
        else :
            self.loss.append([self.entryPrice[0]+0.5, self.entryPrice[1], self.count+0.5, val, val_other])
            
    def plot(self, closeSequence,openSequence,highSequence,lowSequence,folder_name,ratio,name:str, ignoreTimer : int = 0):
        closeSequence = np.array(closeSequence)
        openSequence = np.array(openSequence)
        highSequence = np.array(highSequence)
        lowSequence = np.array(lowSequence)
        fig, ax1 = plt.subplots(figsize=(20,11))
        plt.xlabel('TF',fontsize=20)
        plt.ylabel('Price',fontsize=20)
        plt.title("Bitcoin Price",fontsize=30, fontweight = 'bold')
        # print("len : ",len(highSequence))
        # print("close : ",closeSequence[:30])
        # print("open : ",openSequence[:30])
        for j in range(len(highSequence)):
            if closeSequence[j]>openSequence[j]:
                c='green'
            else:
                c='red'
            plt.plot([j+0.5,j+0.5],[lowSequence[j],highSequence[j]], c)
            ax1.add_patch(Rectangle((j, min(closeSequence[j],openSequence[j])), 1, max(closeSequence[j],openSequence[j])-min(closeSequence[j],openSequence[j]),facecolor =c))

        #plt.plot(np.array(closeSequence[ignoreTimer:]), label='Closing Prices')
# =============================================================================
#         print("WINS")
#         print(self.wins)
#         print("LOSS")
#         print(self.loss)
# =============================================================================
        for a in self.wins:
            plt.scatter(a[0]-ignoreTimer,a[1],s=60,marker='x',c='g')#length_includes_head=True,head_width=5,head_length=30
            plt.arrow(a[0]-ignoreTimer,a[1],a[2]-a[0],a[3]-a[1],color='g')
            ax1.add_patch(Rectangle((a[0]-ignoreTimer, a[1]), a[2]-a[0], a[3]-a[1], facecolor = 'green', alpha = 0.4))
            ax1.add_patch(Rectangle((a[0]-ignoreTimer, a[1]), a[2]-a[0], a[4]-a[1], facecolor = 'red', alpha = 0.4))
                
        for a in self.loss:
            plt.scatter(a[0]-ignoreTimer,a[1],s=60,marker='x',c='r')
            plt.arrow(a[0]-ignoreTimer,a[1],a[2]-a[0],a[3]-a[1],color='r')
            ax1.add_patch(Rectangle((a[0]-ignoreTimer, a[1]), a[2]-a[0], a[4]-a[1], facecolor = 'green', alpha = 0.4))
            ax1.add_patch(Rectangle((a[0]-ignoreTimer, a[1]), a[2]-a[0], a[3]-a[1], facecolor = 'red', alpha = 0.4))
        plt.savefig(folder_name+name)
        #plt.close()


class Policy_03(Policy) :

    """ First serious policy implemented """

    def __init__(self) :
        self.ratio=None
        self.val = []
        self.count=0
        self.entryPrice=[]
        self.wins=[]
        self.loss=[]
        self.moneyToBet = 100
        self.params= {}
        
        self.weight=[]
        


    def sampleFromTrial(self, trial : Trial) :
            # Sample the policy space and return the associated params

        self.params = {
            "TP" : 1.5,#trial.suggest_float('TP',0.2,5),
            "SL" : 0.5,#trial.suggest_float('SL',1,8),
            "weight_0" : trial.suggest_float('weight_0', 0, 1),#0,
            "weight_1" : 0,#trial.suggest_float('weight_1', -1, 1),
            "weight_2" : trial.suggest_float('weight_2', -1, 1),
            "weight_3" : 0,#trial.suggest_float('weight_3', -1, 1),
            "weight_4" : trial.suggest_float('weight_4', -1, 1),
            "weight_5" : 0,#trial.suggest_float('weight_5', -1, 1),
            "weight_6" : trial.suggest_float('weight_6', -1, 1),
            "Theta":    trial.suggest_int("Theta", 3, 7,2),
            "Theta_bis":    trial.suggest_int("Theta_bis", 3, 7,2),
            "Theta_RSI":    14,
            "normFactor" : trial.suggest_float('normFactor', 0.001, 100),
        }


    def betToMake(self, priceRegularized : float,indic, highestRegularized : float, lowestRegularized : float, volume : float, bet : Bet) :
        self.count+=1
        if (bet != None) :
            return "", None, None, None
        #print(indic)
# =============================================================================
#         key_derive_diff_moy='derive_diff_close'+str(self.params["Theta"])+'_'+str(self.params["Theta_bis"])
#         key_signe_diff_moy='sign_diff_moy'+str(self.params["Theta"])+'_'+str(self.params["Theta_bis"])
#         key_croisement_moyennes='croisement_moyennes'+str(self.params["Theta"])+'_'+str(self.params["Theta_bis"])
# =============================================================================
        param = {
            "Theta" : self.params["Theta"],
            "Theta_bis" : self.params["Theta_bis"],
            "Theta_der" : 3,
            "Theta_der2" : 3,
            "Theta_RSI" : self.params["Theta_RSI"]}
        
        val = self.params["weight_1"] *  indic.get_Indicator(param, "dérivé2")
        val += self.params["weight_2"] * indic.get_Indicator(param, "RSI")
        val += self.params["weight_3"] * indic.get_Indicator(param, "derive_diff_close")
        val += self.params["weight_4"] *  indic.get_Indicator(param, "sign_diff_moy")
        val += self.params["weight_5"] *  indic.get_Indicator(param, "volume")
        val += self.params["weight_6"] *  indic.get_Indicator(param, "croisement_moyennes")
        param_pese=["derivé","dérivé2","RSI","derive_diff_close","sign_diff_moy","volume","croisement_moyennes"]
        
        poids=[100*(self.params["weight_"+str(i)] *  indic.get_Indicator(param,param_pese[i]))/val for i in range(7)]
        val /= self.params["weight_0"] *   indic.get_Indicator(param, "std_A")/indic.get_Indicator(param, "std_B")
        val /= self.params["normFactor"]
        self.val.append(val)
        if val > 1 :
            self.entryPrice = [self.count-1, priceRegularized]
            self.weight.append(poids)
            return "Long", self.moneyToBet, self.params["TP"], self.params["SL"]
        elif val < -1:
            self.entryPrice = [self.count-1, priceRegularized]
            self.weight.append(poids)
            return "Short", self.moneyToBet, self.params["TP"], self.params["SL"]
        else :
            return "", None, None, None
        print("priceRegularized, highestRegularized, lowestRegularized : ", priceRegularized, highestRegularized, lowestRegularized)


    def addTrade(self, win : bool, val : float) :
        if win == True :
            self.wins.append([self.entryPrice[0], self.entryPrice[1], self.count-1, val])
        else :
            self.loss.append([self.entryPrice[0], self.entryPrice[1], self.count-1, val])
            
    def plot(self, closeSequence,folder_name,ratio,name:str, ignoreTimer : int = 0):
        plt.figure(figsize=(16,9))
        plt.xlabel('Minutes')
        plt.ylabel('Price')
        plt.plot(np.array(closeSequence[ignoreTimer:]), label='Closing Prices')
# =============================================================================
#         print("WINS")
#         print(self.wins)
#         print("LOSS")
#         print(self.loss)
# =============================================================================
        for a in self.wins:
            plt.scatter(a[0]-ignoreTimer,a[1],s=60,marker='x',c='g')#length_includes_head=True,head_width=5,head_length=30
            plt.arrow(a[0]-ignoreTimer,a[1],a[2]-a[0],a[3]-a[1],color='g')
        for a in self.loss:
            plt.scatter(a[0]-ignoreTimer,a[1],s=60,marker='x',c='r')
            plt.arrow(a[0]-ignoreTimer,a[1],a[2]-a[0],a[3]-a[1],color='r')
        plt.savefig(folder_name+name)
        #plt.close()



if __name__ == "__main__" :
    policy = Policy_01()
    policy.sample()

    policy = Policy_02()
    policy.sample()

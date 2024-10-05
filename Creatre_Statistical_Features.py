'''
    Written By: Mehdi Touyserkani - Aug 2024.
    https://ir-bestpro.com.
    https://www.linkedin.com/in/bestpro-group/
    https://github.com/irbestpro/
    ir_bestpro@yahoo.com
    BESTPRO SOFTWARE ENGINEERING GROUP

'''

import numpy as np
from scipy.stats import skew,kurtosis

class Statistical_Features():

    def __init__(self , Data): # Class Constructor
        self.Data = Data
        self.X = []

    def Create_Features(self):
        for i in range(0,len(self.Data)):
            temp = self.Data[i]
            self.X.append(np.mean(temp))
            self.X.append(skew(temp , bias=True))
            self.X.append(kurtosis(temp, bias=True))
            self.X.append(np.sqrt(np.mean(temp**2)))
            
            #self.X.append(np.std(temp))
            #self.X.append(np.sum(temp**2))
            #self.X.append(np.sqrt(np.mean(temp**2)) / np.sum(temp**2))
            #self.X.append(np.mean(temp) / np.std(temp))
        
        return self.X

'''
    These code are written By Mehdi Touyserkani.
    Email Address: ir_bestrpo@yahoo.com.
    Personal Website: Https://www.ir-bestpro.com
    LinkedIn: https://www.linkedin.com/in/bestpro-group/
    
'''

import pywt
import numpy as np
import pandas as pd
from mspca import mspca
from PyEMD import EMD
import Creatre_Statistical_Features

#__________________Loading Dataset________________________

def loadDataSets():

    fileNames = ['A' , 'B' , 'C' , 'D' , 'E']
    DataSets = pd.read_csv( './DataSets/A' + '.csv').to_numpy()
    Labels = np.zeros((len(fileNames) * 99 ,1))
    for i in range(1, len(fileNames)):
        DataSets = np.concatenate((DataSets , pd.read_csv( './DataSets/'+ fileNames[i] + '.csv').to_numpy()) , axis= 0)
        Labels[i * 99 : (i+1) * 99] = np.ones((99,1)) * i
    
    return DataSets , Labels

#______Start DWT Feature Extractor and Deep Learning_________

if __name__ == '__main__':
    
    Dataset , Labels = loadDataSets()

    #________Denoise Process with MSPCA Algorithm_____________

    MSPCA = mspca.MultiscalePCA() # create denoising object (MSPCA)
    Dataset = MSPCA.fit_transform(Dataset, wavelet_func='db4', threshold=0.5)[0:len(Dataset) , :] # Denoised Dataset
    #pd.DataFrame(Dataset).to_csv('./Features/Denoised.csv')

    emd = EMD() # Create Emperical Mode Decomposition Object
    Coefficients = [] # combination of DWT and EMD Coefficients

    #______________Applying DWT Here__________________________

    for freq in range(0,len(Dataset)):

        print('Extracting DWT and EMD Features From Sample #%d '%freq)
        DWT_Coeffs = pywt.wavedec(Dataset[freq , :] , 'db4' , level=6) # all DWT Coefficients and details(D1,D2,D3,D4,D5,D6,A)
        IMF = emd(Dataset[0,:] , max_imf=7)[7] # 7 Levels of Decomposition
        #D1 = pywt.waverec(coeffs=Coeffs , wavelet='db4') # Signal Reconstruction

        temp_coef = np.zeros((1 , 4137))
        c = 0
        for i in range(0,len(DWT_Coeffs)): # Vertical Concatenation of DWT Coefficients
            temp_coef [0 , c: c + len(DWT_Coeffs[i])] = DWT_Coeffs[i]
            c += len(DWT_Coeffs[i])
        
        obj = Creatre_Statistical_Features.Statistical_Features([temp_coef[0] , IMF])
        temp = obj.Create_Features()
        Coefficients.append(temp)

    pd.DataFrame(Coefficients).to_csv('./Features/DWT_EMD_Features.CSV')
    pd.DataFrame(Labels).to_csv('./Features/Labels.CSV')
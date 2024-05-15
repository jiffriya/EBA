
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


from M12.Testconfig12 import (
(
    ROOT,
    TEST_SET_LIST,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    LL_FEATURE_PATH,
    LL_LENGTH,
    ANGLE_FEATURE_PATH,
    AngLENGTH,
    
    )



LL_FEATURE=17



class CustomDataset12(Dataset):
    
    
   
    def __init__(self, pid_path: Path ):
   
        
        all_pids = np.loadtxt(fname=str(TEST_SET_LIST), dtype='str').tolist()

       
        self.ll_data = np.zeros((len(all_pids), LL_LENGTH,LL_FEATURE)) 
        self.angle_data = np.zeros((len(all_pids), AngLENGTH))
        
        self.y_labels = []
        
     
        
       
        
        affinity = {}
        affinity_df = pd.read_csv(ROOT / "affinity_data.txt", delimiter='\t')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        #self.affinity = affinity
        affinity = affinity
        
        
        
        for i, pid in enumerate(all_pids):
            print(pid)
            #self.y_labels.append(affinity[pid])
            self.y_labels.append(affinity[pid])
            
            #ligand_smi=label_smiles(self.smi[pid], max_smi_len)
            #self.ll_data[i]= ligand_smi
            
            with open(f"{LL_FEATURE_PATH.absolute()}/{pid}.pkl", "rb") as dif:
                ll_info = pickle.load(dif)
            #self.ll_data[i] = ll_info['smile_features'][:LL_LENGTH]
            if ll_info.shape[0] > LL_LENGTH :
                self.ll_data[i, :, :] = ll_info[ :LL_LENGTH, :]
            else:
                self.ll_data[i, :ll_info.shape[0], :] = ll_info[:,:]
            
            
            
            
            
            with open(f"{ANGLE_FEATURE_PATH.absolute()}/{pid}_angle_info.pkl", "rb") as dif:
                self.angle_info = pickle.load(dif)   #pl_angle Dim 40 is ok or pl_angle_1D
                #self.LenangleBin.append(len(self.angle_info['BinAngle1D']))
                if self.angle_info['BinAngle1D'].shape[0] > AngLENGTH:
                    self.angle_data[i, :] = self.angle_info['BinAngle1D'][:AngLENGTH]
                else:
                    self.angle_data[i, :self.angle_info['BinAngle1D'].shape[0]] = self.angle_info['BinAngle1D']
            
           
                    
       
           
        np.savetxt( CHECKPOINT_PATH1 / "Test2016_290label.lst",  self.y_labels, delimiter='\t', fmt='%f')
        #np.savetxt( CHECKPOINT_PATH1 / "lengthDistan.lst",  self.LenDistBin, delimiter='\t', fmt='%f')
        #*************************************************



    def __getitem__(self, idx):
        
        al = np.int32(self.angle_data[idx, :])
        ll = np.float32(self.ll_data[idx, :])
        return ( al, ll), self.y_labels[idx]

    def __len__(self):
        return len(self.y_labels)
    

#a=CustomDataset1(TEST_SET_LIST)            


        
            
        
        
   

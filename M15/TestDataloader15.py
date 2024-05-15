

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset



from M15.Testconfig15 import (
    ROOT,
    LL_FEATURE_PATH,
    PK_PATH,
    max_pkt_len,
    PK_FEATURE_SIZE,
    
    TEST_SET_LIST,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    LL_LENGTH,)




LL_FEATURE=17
#max_pkt_len=63

            

            
class CustomDataset1(Dataset):
    
    
    
    def __init__(self, pid_path: Path ):
        #print("Loading data")
        
       
        
        all_pids: list = np.loadtxt(fname=str(pid_path.absolute()), dtype='str').tolist()
        
        self.max_pkt_len=63
        #self.ll_data = np.zeros((len(all_pids), max_smi_len))  # ll_info['smile_features'].shape[0]
        self.ll_data = np.zeros((len(all_pids), LL_LENGTH,LL_FEATURE)) 
        self.pk_data = np.zeros((len(all_pids), max_pkt_len, PK_FEATURE_SIZE))
        
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
            #with open(f"{LL_FEATURE_PATH}\{pid}.pkl", "rb") as dif:
                ll_info = pickle.load(dif)
            #self.ll_data[i] = ll_info['smile_features'][:LL_LENGTH]
            if ll_info.shape[0] > LL_LENGTH :
                self.ll_data[i, :, :] = ll_info[ :LL_LENGTH, :]
            else:
                self.ll_data[i, :ll_info.shape[0], :] = ll_info[:,:]
            
            
            _pkt_tensor = pd.read_csv(f"{PK_PATH.absolute()}/{pid}.csv" , index_col=0)
            if "idx" in _pkt_tensor.columns: _pkt_tensor = _pkt_tensor.drop(['idx'], axis=1).values[:self.max_pkt_len] 
            else:_pkt_tensor = _pkt_tensor.values[:self.max_pkt_len] 
            
            pkt_tensor = np.zeros((self.max_pkt_len, PK_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor
            self.pk_data[i] =  pkt_tensor   
            
            
            
           
                    
       
           
        np.savetxt( CHECKPOINT_PATH1 / "Test2016_290label.lst",  self.y_labels, delimiter='\t', fmt='%f')
        #np.savetxt( CHECKPOINT_PATH1 / "lengthDistan.lst",  self.LenDistBin, delimiter='\t', fmt='%f')
        #*************************************************



    def __getitem__(self, idx):
        ll = np.float32(self.ll_data[idx, :])
        pk = np.int32(self.pk_data[idx, :])
        return ( ll, pk), self.y_labels[idx]

    def __len__(self):
        return len(self.y_labels)
    
    
    

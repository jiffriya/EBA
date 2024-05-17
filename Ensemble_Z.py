
#E5-1

from pathlib import Path
import os
import sys
import torch
from torch.utils.data import DataLoader

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt

sys.path.append('../M12/Testconfig12') 
sys.path.append('../M12/TestDataloader12')
sys.path.append('../M12/model12')

sys.path.append('../M24/Testconfig24') 
sys.path.append('../M24/TestDataloader24')
sys.path.append('../M24/model24')

sys.path.append('../M234/Testconfig234') 
sys.path.append('../M234/TestDataloader234')
sys.path.append('../M234/model234')

sys.path.append('../M135/Testconfig135') 
sys.path.append('../M135/TestDataloader135')
sys.path.append('../M135/model135')

sys.path.append('../M345/Testconfig345')  
sys.path.append('../M345/TestDataloader345')
sys.path.append('../M345/model345')

sys.path.append('../M1234/Testconfig1234') 
sys.path.append('../M1234/TestDataloader1234')
sys.path.append('../M1234/model1234')

sys.path.append('../M1345/Testconfig1345') 
sys.path.append('../M1345/TestDataloader1345')
sys.path.append('../M1345/model1345')

sys.path.append('../M1234/Testconfig2345') 
sys.path.append('../M1234/TestDataloader2345')
sys.path.append('../M1234/model2345')

sys.path.append('../45/Testconfig45')
sys.path.append('../M45/TestDataloader45')
sys.path.append('../M45/model45')
###########################################################

from M12.Testconfig12 import (
    ROOT,
    TEST_SET_LIST,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    LL_FEATURE_PATH,
    LL_LENGTH,
    ANGLE_FEATURE_PATH,
    AngLENGTH
    )
    
from M12.TestDataloader12 import CustomDataset12
from M12.model12 import Model12
#***********************************************

from M24.Testconfig24 import (
    ROOT,
    TEST_SET_LIST,
    BATCH_SIZE,
    max_smi_len,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    ANGLE_FEATURE_PATH,
    AngLENGTH,
    SMI_PATH
    )

from M24.TestDataloader24 import CustomDataset24
from M24.model24 import Model24
#***********************************************
from M234.Testconfig234 import (
    ROOT,
    TEST_SET_LIST,
    BATCH_SIZE,
    PT_FEATURE_SIZE,
    max_seq_len,
    max_smi_len,
    PP_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    ANGLE_FEATURE_PATH,
    AngLENGTH,
    SMI_PATH
    )




from M234.TestDataloader234 import CustomDataset234
from M234.model234 import Model234
#*******************************************


from M135.Testconfig135 import (
    ROOT,
    PK_PATH,
    PK_FEATURE_SIZE,
    max_pkt_len,
    TEST_SET_LIST,
    BATCH_SIZE,
    PT_FEATURE_SIZE,
    max_seq_len,
    PP_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    LL_FEATURE_PATH,
    LL_LENGTH,
    )

from M135.TestDataloader135 import CustomDataset135
from M135.model135 import Model135

#**********************************************
from M345.Testconfig345 import (
    ROOT,
    PK_PATH,
    PK_FEATURE_SIZE,
    max_pkt_len,
    TEST_SET_LIST,
    BATCH_SIZE,
    PT_FEATURE_SIZE,
    max_seq_len,
    max_smi_len,
    PP_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    SMI_PATH
    
    )

from M345.TestDataloader345 import CustomDataset345
from M345.model345 import Model345 

#***********************************************
from M1234.Testconfig1234 import (
    ROOT,
    TEST_SET_LIST,
    BATCH_SIZE,
    PT_FEATURE_SIZE,
    max_seq_len,
    max_smi_len,
    PP_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    LL_FEATURE_PATH,
    LL_LENGTH,
    ANGLE_FEATURE_PATH,
    AngLENGTH,
    SMI_PATH
    )

from M1234.TestDataloader1234 import CustomDataset1234
from M1234.model1234 import Model1234 
#*******************************************
from M1345.Testconfig1345 import (
    ROOT,
    PK_PATH,
    PK_FEATURE_SIZE,
    max_pkt_len,
    TEST_SET_LIST,
    BATCH_SIZE,
    PT_FEATURE_SIZE,
    max_seq_len,
    max_smi_len,
    PP_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    LL_FEATURE_PATH,
    LL_LENGTH,
    #ANGLE_FEATURE_PATH,
    #AngLENGTH,
    SMI_PATH
    )


from M1345.TestDataloader1345 import CustomDataset1345
from M1345.model1345 import Model1345

#*************************************
from M2345.Testconfig2345 import (
    ROOT,
    PK_PATH,
    PK_FEATURE_SIZE,
    max_pkt_len,
    TEST_SET_LIST,
    BATCH_SIZE,
    PT_FEATURE_SIZE,
    max_seq_len,
    max_smi_len,
    PP_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    ANGLE_FEATURE_PATH,
    AngLENGTH,
    SMI_PATH
    )

from M2345.TestDataloader2345 import CustomDataset2345
from M2345.model2345 import Model2345


#**********************************************
from M45.Testconfig45 import (
    ROOT,
    PK_PATH,
    PK_FEATURE_SIZE,
    max_pkt_len,
    TEST_SET_LIST,
    BATCH_SIZE,
    max_smi_len,
    CHECKPOINT_PATH,
    CHECKPOINT_PATH1,
    TOTAL_EPOCH,
    SMI_PATH
    )

from M45.TestDataloader45 import CustomDataset45
from M45.model45 import Model45


####################################################


#**********************************************
model12='model12.pt'
model24='model24.pt'
model234='model234.pt'
model135='model135.pt'
model345='model345.pt'
model1234='model1234.pt'
model1345='model1345.pt'
model45='model45.pt'
model2345='model2345.pt'




file_names=[model12,model24,model234,model135,model345,model1234,model1345,model2345,model45]

#**************************************************************************************
def forward_pass(model, x, device):
    model.eval()
    for i in range(len(x)):
        x[i] = x[i].to(device)
    return model(x)

def predict():
    weight_file_path1 =   file_names[0]
    weight_file_path2 =   file_names[1]
    weight_file_path3 =   file_names[2]
    weight_file_path4 =  file_names[3]
    weight_file_path5 = file_names[4]
    weight_file_path6 = file_names[5]
    weight_file_path7 = file_names[6]
    weight_file_path8 = file_names[7]
    #weight_file_path9 = file_names[4]
    weight_file_path10 = file_names[8]
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device_name}")
    device = torch.device(device_name)
    model1 = Model12().to(device)
    model2 = Model24().to(device)
    model3 = Model234().to(device)
    model4 = Model135().to(device)
    model5 = Model345().to(device)
    model6 = Model1234().to(device)
    model7 = Model1345().to(device)
    model8 = Model2345().to(device)
    #model9 = Model12345().to(device)
    model10 = Model45().to(device)
    
    
    model1.load_state_dict(torch.load(weight_file_path1, map_location=torch.device('cpu')))
    print(f"Model1 weights loaded in {device_name}")
     
    model2.load_state_dict(torch.load(weight_file_path2, map_location=torch.device('cpu')))
    print(f"Model2 weights loaded in {device_name}")
    
    model3.load_state_dict(torch.load(weight_file_path3, map_location=torch.device('cpu')))
    print(f"Model3 weights loaded in {device_name}")
    
    model4.load_state_dict(torch.load(weight_file_path4, map_location=torch.device('cpu')))
    print(f"Model4 weights loaded in {device_name}")

    model5.load_state_dict(torch.load(weight_file_path5, map_location=torch.device('cpu')))
    print(f"Model5 weights loaded in {device_name}")
    
    model6.load_state_dict(torch.load(weight_file_path6, map_location=torch.device('cpu')))
    print(f"Model6 weights loaded in {device_name}")
    
    model7.load_state_dict(torch.load(weight_file_path7, map_location=torch.device('cpu')))
    print(f"Model7 weights loaded in {device_name}")
    
    model8.load_state_dict(torch.load(weight_file_path8, map_location=torch.device('cpu')))
    print(f"Model8 weights loaded in {device_name}")
    
    model10.load_state_dict(torch.load(weight_file_path10, map_location=torch.device('cpu')))
    print(f"Model10 weights loaded in {device_name}")
    
    #dataset=CustomDataset(pid_path=TRAIN_SET_LIST)
    
    print(TEST_SET_LIST)
    model1_value=[]
    model2_value=[]
    model3_value=[]
    model4_value=[]
    model5_value=[]
    model6_value=[]
    model7_value=[]
    model8_value=[]
    #model9_value=[]
    model10_value=[]
    
    dataloader1 = DataLoader(
        dataset=CustomDataset12(pid_path = TEST_SET_LIST) ,
        batch_size=1)
    
  
    #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
    with open(CHECKPOINT_PATH1 / "PredictedModel1.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader1:
                prediction = forward_pass(model1, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:        	
                    #print(value)
                    rounded_value1 = round(float(value), 2) 
                    model1_value.append(rounded_value1)
                    
                    file.write(f"{rounded_value1}\n")
    print("Model1 done")
    
    #**********************
    dataloader2 = DataLoader(
        dataset=CustomDataset24(pid_path=TEST_SET_LIST ) ,
        batch_size=1)
    
    #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
    
    with open(CHECKPOINT_PATH1 / "PredictedModel2.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader2:
                prediction = forward_pass(model2, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:        	
                    #print(value)
                    rounded_value2 = round(float(value), 2)
                    model2_value.append(rounded_value2)
                    file.write(f"{rounded_value2}\n")
    print("Model2 done")
    
    #**************************************
    dataloader3 = DataLoader(
        dataset=CustomDataset234(pid_path=TEST_SET_LIST ) ,
        batch_size=1)
    
    #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
    
    with open(CHECKPOINT_PATH1 / "PredictedModel3.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader3:
                prediction = forward_pass(model3, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:        	
                    #print(value)
                    rounded_value3 = round(float(value), 2)
                    model3_value.append(rounded_value3)
                    file.write(f"{rounded_value3}\n")
    print("Model3 done")
    #******************************
    
    dataloader4 = DataLoader(
        dataset=CustomDataset135(pid_path = TEST_SET_LIST) ,
        batch_size=1)
    
  
    #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
    with open(CHECKPOINT_PATH1 / "PredictedModel4.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader4:
                prediction = forward_pass(model4, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:        	
                    #print(value)
                    rounded_value4 = round(float(value), 2) 
                    model4_value.append(rounded_value4)
                    
                    file.write(f"{rounded_value4}\n")
    print("Model4 done")
    
#*********************************
    dataloader5 = DataLoader(
        dataset=CustomDataset345(pid_path = TEST_SET_LIST) ,
        batch_size=1)
    
  
    #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
    with open(CHECKPOINT_PATH1 / "PredictedModel5.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader5:
                prediction = forward_pass(model5, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:        	
                    #print(value)
                    rounded_value5 = round(float(value), 2) 
                    model5_value.append(rounded_value5)
                    
                    file.write(f"{rounded_value5}\n")
    print("Model5 done")
    
    #*********************************
    dataloader6 = DataLoader(
        dataset=CustomDataset1234(pid_path = TEST_SET_LIST) ,
        batch_size=1)
    
  
    #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
    with open(CHECKPOINT_PATH1 / "PredictedModel6.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader6:
                prediction = forward_pass(model6, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:        	
                    #print(value)
                    rounded_value6 = round(float(value), 2) 
                    model6_value.append(rounded_value6)
                    
                    file.write(f"{rounded_value6}\n")
    print("Model6 done")
    
    #**************************
    
    dataloader7 = DataLoader(
        dataset=CustomDataset1345(pid_path = TEST_SET_LIST) ,
        batch_size=1)
    
  
    #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
    with open(CHECKPOINT_PATH1 / "PredictedModel7.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader7:
                prediction = forward_pass(model7, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:        	
                    #print(value)
                    rounded_value7 = round(float(value), 2) 
                    model7_value.append(rounded_value7)
                    
                    file.write(f"{rounded_value7}\n")
    print("Model7 done")
    
    #**************************
    
    dataloader8 = DataLoader(
        dataset=CustomDataset2345(pid_path = TEST_SET_LIST) ,
        batch_size=1)
    
  
    #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
    with open(CHECKPOINT_PATH1 / "PredictedModel8.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader8:
                prediction = forward_pass(model8, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:        	
                    #print(value)
                    rounded_value8 = round(float(value), 2) 
                    model8_value.append(rounded_value8)
                    
                    file.write(f"{rounded_value8}\n")
    print("Model8 done")
    
    #**************************
    
    
    dataloader10 = DataLoader(
        dataset=CustomDataset45(pid_path = TEST_SET_LIST) ,
        batch_size=1)
    
  
    #with open(CHECKPOINT_PATH1 / "BinPredicted2016.lst", "w") as file:
    with open(CHECKPOINT_PATH1 / "PredictedModel10.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader10:
                prediction = forward_pass(model10, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:        	
                    #print(value)
                    rounded_value10 = round(float(value), 2) 
                    model10_value.append(rounded_value10)
                    
                    file.write(f"{rounded_value10}\n")
    print("Model10 done")
    
    model1_value = np.array( model1_value)
    model2_value = np.array( model2_value)
    model3_value = np.array( model3_value)
    model4_value = np.array( model4_value)
    model5_value = np.array( model5_value)
    model6_value = np.array( model6_value)
    model7_value = np.array( model7_value)
    model8_value = np.array( model8_value)
    model10_value = np.array( model10_value)
    
    
    Avg_value=(model1_value+model2_value+model3_value+model4_value+model5_value+model6_value+model7_value+model8_value+model10_value)/9
    
    
    
    Avg_value=np.round( Avg_value, 2)
    Avg_value_list =  Avg_value.tolist()
    with open(CHECKPOINT_PATH1 / "FinalPredicted.lst", "w") as file:
        for item in Avg_value_list:
            file.write(f"{ item}\n")
            
        
#*********************************
    
    predicted=[]
    
    try:
        with open(CHECKPOINT_PATH1 / "FinalPredicted.lst", 'r') as file:
        #with open('/home/mac/Research2023/JNewAngle/Result/Predicted2016.lst', 'r') as file:
            for line in file:
                # Process each line as needed
                line.strip()
                predicted.append(line)
                  # Prints each line after stripping newline characters
    except FileNotFoundError:
        print(f"Predicted File not found:")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
    Actual=[]
    try:
       
        with open( CHECKPOINT_PATH1 / "Test2016_290label.lst", 'r') as file:
            for line in file:
                # Process each line as needed
                line.strip()
                Actual.append(line)
                  # Prints each line after stripping newline characters
    except FileNotFoundError:
        print(f"Test label File not found:")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
    actual_score =np.array(Actual, dtype='float32') 
    predicted_score = np.array(predicted, dtype='float32')
    
    print("\nlength",len(actual_score), len(predicted_score))
    ## Pearson correlation coefficient (R)
    r = stats.pearsonr(actual_score, predicted_score)[0]
    
    mse = mean_squared_error(actual_score, predicted_score)
    rmse = sqrt(mse)

     
    mae = mean_absolute_error(actual_score, predicted_score)

     
    sd = np.sqrt(sum((np.array(actual_score)-np.array(predicted_score))**2)/(len(actual_score)-1))

    CI=c_index(actual_score, predicted_score) 

    print("Pearson correlation coefficient = ", np.around( r, 3))
    print("Root mean squared error = ", np.around( rmse, 3))
    print("Mean absolute error = ", np.around( mae, 3))
    print("Mean square error = ", np.around( mse, 3))
    print("standard deviation = ", np.around( sd, 3))
    print("Concordance Index  = " ,np.around( CI, 3))
    #print("Model Weight  =", file_names[i])"""

    #with open('/home/jiffriya/Dataset2016/MSfCNN/TrainingData/Figure2013/EMetrics2013.txt', 'a') as f:
    #with open('/home/mac/Dataset/myprogram/Newscfnn/AngleGenerate/EMetrics2013.txt', 'a') as f:
    with open(CHECKPOINT_PATH1 / "Evaluate_Final.txt", 'a') as f:
        f.write("Pearson correlation coefficient = %f "%(np.around( r, 3)))
        f.write("\nRoot mean squared error = %f" %np.around( rmse, 3))
        f.write("\nMean absolute error = %f" %np.around( mae, 3))
        f.write("\nMean square error = %f" %np.around( mse, 3))
        f.write("\nstandard deviation = %f" % np.around( sd, 3))
        f.write("\nConcordance Index  = %f" % np.around( CI, 3))
        #f.write("\nModel Weight  = %s\n\n" %file_names[i])
        #f.write("\nTest Loss%f " % loss1)
        #f.write("Test MAE = %f " % mae1)
            


def c_index(y_true, y_pred):
    
    #y_true=test_y1 
   
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1
    if pair != 0:
        result= summ / pair
    else:
        result=0
    return result
    
    

        
    
      
    
    


if __name__ == "__main__":
    predict()

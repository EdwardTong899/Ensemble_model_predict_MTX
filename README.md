# 基於Ensemble learning預測MTX

 
   - Data取自於群益api (60000分鐘k棒)
   - 使用兩種分類器進行投票 (AdaBoost & XGBoost)
   - Predict Dataset : training Dataset = 2 : 8
   - 分類方式: 20分鐘內漲幅大於25點 = 1, 20分鐘內跌幅大於25點 = 2, 20分鐘內價格在+-20點之間 = 0.


# 執行流程
1. 使用Creat_indicator_v2.py 
   - import MTX00_data.txt產生對應indicator
   - 輸出MTX00_result_data.csv

2. 開啟colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WZg6YNu3J_lN-ngcHXIK6Jhp1px7cwZU#scrollTo=t7EpuquYF9r3)

3. Import data  
    - MTX00_result_data.csv
      
4. 資料前處理
  ```shell
stock_data = []
dataframe = import_data()
inital_index = cac_inital_index(dataframe) # 計算從哪一列開始，避開空白位置
Bollinger_Band_index = cac_Bollinger_Band_index(dataframe,inital_index) # 計算Bollinger_Band

slope_5days = cac_slope_5days(dataframe,inital_index) # 計算5日斜率

slope_60mins = cac_slope_60mins(dataframe,inital_index)# 計算60分鐘
slope_5mins = cac_slope_5mins(dataframe,inital_index)# 計算5分鐘

time_divisor = cac_time_divisor(dataframe,inital_index)# 計算時間區隔

quantity_divisor = cac_quantity_divisor(dataframe,inital_index) # 計算數量區隔
# show_value_distributed(quantity_divisor)
y = get_y(dataframe, inital_index)
show_value_distributed(y)

# 寫入feature
stock_data.append(Bollinger_Band_index)
stock_data.append(slope_5days)
stock_data.append(slope_60mins)
stock_data.append(slope_5mins)
stock_data.append(time_divisor)
stock_data.append(quantity_divisor)
stock_data.append(y)

```  
5. 資料切割 
  ```shell
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(df, train_size=0.8, shuffle=False)
print('訓練資料有%s筆' %len(train_data))
print('驗證資料有%s筆' %len(val_data))

X_train_df = train_data.drop([6], axis=1) #,"L1-dcache-stores","dTLB-loads","L1-dcache-loads","branch-instructions"
X_train = np.array(X_train_df).astype(float)
Y_train_df = train_data[6]
Y_train = np.array(Y_train_df).astype(int)

X_valid_df = val_data.drop([6], axis=1) #,"L1-dcache-stores","dTLB-loads"
X_valid = np.array(X_valid_df).astype(float)
Y_valid_df = val_data[6]

```  
  
6. XGBboost預測
  ```shell
# xgboost
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc
import numpy as np
import time
ts = time.time()




# Confusion Matrix
mat = confusion_matrix(Y_valid, xg1_val)
sns.heatmap(mat,square= True, annot=True, cbar= False, fmt='d')
plt.xlabel("predicted value")
plt.ylabel("true value")
plt.show()

# Accuracy
accuracy = metrics.accuracy_score(Y_valid, xg1_val)
print('Valdation accuracy:', accuracy)

# precision, recall, f1-score
#target_names = ['0','1','2','3']
target_names = ['0','1','2']

print("report:\n",classification_report(Y_valid, xg1_val, target_names=target_names))

print('Total time took: {0}s'.format(time.time()-ts))

accuracy_score(Y_valid,xg1_val)
## Valdation accuracy: 0.8923076923076924
## Test accuracy: 0.41203281677301734
```  
    
7. 使用2種模型投票結果   
  ```shell
value = 0
value_list = []
for i in range(len(ensemble_val)):
  if(ensemble_val[i] == 1 and Y_valid[i] == 1):
    value = value + 23
  elif(ensemble_val[i] == 2 and Y_valid[i] == 2):
    value = value + 23
  elif(ensemble_val[i] == 1 and Y_valid[i] == 0):
    value = value - 2
  elif(ensemble_val[i] == 2 and Y_valid[i] == 0):
    value = value - 2
  elif(ensemble_val[i] == 0):
    value = value + 0
  else:
    value = value -27
  value_list.append(value)

print(value)
``` 




# 2nd-ML100Days
   
Day_001: 定義 Mean Squared Error 及 Mean Absolute Error  
Day_002: 讀取資料、查看資料  
Day_003: 讀取網頁資料(txt)、txt 轉成 Dataframe、Try-Except    
Day_004: One-Hot Encoder( get_dummies() 函數 )    
Day_005: 計算平均值、標準差、最大值、最小值 及 繪製直方圖  
Day_006: 篩選數值型欄位、Outliers( 盒鬚圖、ECDF、直方圖 )  
Day_007: Outlier的處理( 補 分位數(Quantile) )  
Day_008: 資料分組離散化( cut() 函數 : 等距分組 )  
Day_009: 計算相關係數( Correlation Coefficient )、散佈圖( Scatter Plot )檢視相關性  
Day_010: 計算相關係數( Correlation Coefficient )、散佈圖( Scatter Plot )或盒鬚圖( Box Plot )檢視相關性  
Day_011: 資料分組離散化( cut() : 等距分組 + np.linspace() 函數 )、KDE、長條圖( Bar Plot )  
Day_012: 資料分組離散化( cut() : 等距分組 、 qcut() : 等頻分組 )  
Day_013: 資料分組離散化( cut() : 等距分組 、 qcut() : 等頻分組 )  
Day_014: 繪圖排版 subplots   
Day_015: Heatmap、Gridplot、隨機矩陣( np.random.random : 隨機小數均勻分布；np.random.randn : 常態分布 )   
Day_016: 匯出儲存成 csv 檔  
Day_017: 篩選類別型的欄位，將其轉成數值型態欄位( LabelEncoder()、MinMaxScaler() )  
Day_023: Label Encoder 標籤編碼( LabelEncoder() 函數 ) 及 Mean Encoder 均值編碼( 利用 groupby 函數執行 )  
Day_024: Counting Encoder 計數編碼( 利用 groupby 函數執行 ) 及 Feature Hash 特徵雜湊( hash() 函數 )  
Day_025: 時間特徵分解( 年、月、日、時、分、秒 ) 及 週期循環特徵( 利用 sin 或 cos 函數執行 )  
Day_026: 特徵組合( ex: 經緯度座標 )  
Day_027: 特徵組合( Group by Encoding 群聚編碼 : 合成類別特徵與數值特徵 )  
Day_028: 特徵篩選    
         (1) 相關係數過濾法 : 利用 corr() + list() + pop() 函數   
         (2) L1-Embedding(Lasso Regression Embedding) : 利用 Lasso( alpha = ) 函數  
         (3) GDBT(梯度提升樹) Embedding    
  
Day_029: 樹狀模型的特徵重要性( estimator.feature_importances_ )  
Day_030: 分類預測模型的特徵優化( 隨機森林 Random Forset + 葉編碼 Leaf Encoding + Logistic Regression )  
Day_034: 切分 訓練集/測試集 資料    
         (1) 資料切分 : train_test_split() 函數   
         (2) 交叉採樣 : KFold( n_splits = ,shuffle = False ) 函數   
         ＊＊ n_splits 為等份數；shuffle = False 表示每次劃分的結果相同  
         註：當樣本不均衡時，須搭配運用的函數 np.where()、np.concatenate()  

Day_035: 模型的選擇 by 預測類型  
         (1) 迴歸問題：預測的目標為實數  
         (2) 分類問題：預測的目標為類別  
         (3) 二元分類(binary-class): 瑕疵 vs 正常  
         (4) 多元分類(multi-class): 手寫辨識1~9 ；    
         (5) 多標籤(multi-label): 如天氣預測多雲時晴  
         (6) 迴歸問題可轉化成分類問題
  
Day_036: 模型的評估指標  
         (1) 迴歸問題：MAE(Mean Absolute Error)、MSE(Mean Square Error)、R-Square  
         (2) 分類問題：AUC(Area Under Curve)、Precision、Recall、F1-Score  
  
Day_037: 線性迴歸(Linear Regression) v.s. 邏輯斯迴歸(Logistic Regression)  
Day_038: 線性迴歸 ＆ 邏輯斯迴歸 - 程式碼    
         from sklearn import linear_model
         (1) 線性：Linear = linear_model.LinearRegression( )  
         (2) 邏輯斯：Logistic = linear_model.LogisticRegression( )   
         
Day_039: Lasso & Ridge Regression  
         (1) 正則化(Regularization): 避免模型過擬合(over-fitting)  
         (2) 正則化方法：Lasso(L1)、Ridge Regression(L2)   

Day_040: Lasso & Ridge Regression - 程式碼 
         from sklearn import linear_model
         (1) Lasso = linear_model.Lasso( alpha =  )  
         (2) Ridge = linear_model.Ridge( alpha =  )  

  


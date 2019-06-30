# 2nd-ML100Days


   
#### Day_001: 定義 Mean Squared Error 及 Mean Absolute Error   
   * MSE  

         程式碼：  
         def mean_squared_error( y, y_hat ):
             MSE = np.sum(  ( y - y_hat )**2 ) / len( y )
             return MSE
   * MAE

         程式碼： 
         def mean_squared_error( y, y_hat ):
             MAE = np.sum(  np.abs( y - y_hat ) ) / len( y )
             return MSE


#### Day_002: 讀取資料、檢視資料資訊  
   * 資料讀取

         程式碼：  
         dir_data = ''                                # 資料存放路徑
         data_name = os.path.join( dir_data, '檔名' )  # 資料檔名
         Data = pd.read_csv( data_name )              # 讀取資料

   * 檢視資料的樣態  

         程式碼：  
         data.shape       # 資料 row 與 column 的數目
         data.columns     # 資料的欄位名稱
         data.info( )     # 資料欄位資訊( 可知欄位有無 Missing Values )
         data.types       # 資料欄位的型態( object, int, float )


#### Day_003: 建立新的 Dataframe、Try-Except 例外處理  
   * 建立 dataframe  
    
         程式碼：
         Data = pd.DataFrame( data, columns )
          
   * Try-Except 例外處理  
   
         程式碼：
         try : 
            執行原本工作
         except :
            若上述原本工作無法執行時，則執行此例外處理
            
   Example : 
         
         a = 
         b = 
         try : 
            if a < b : 
               print( 'b - a = ' + str( b-a ) )
         except :
               print( 'a >= b ' )


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
  
Day_041: 決策樹(Decision Tree)     
Day_042: 決策樹(Decision Tree) - 程式碼     
         from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    
         (1) 迴歸問題： DTR = DecisionTreeRegressor( )  
         (2) 分類問題： DTC = DecisionTreeClassifier( )  
  
   
集成學習(Ensemble Learning): 可避免當決策樹足夠深時，容易導致過擬合(overfitting)的缺點   
(i) Bagging 從原資料集中隨機做取後放回的採樣，分別在採樣的子集訓練模型 - 隨機森林(Random Forest)   
(ii) Boosting 透過迭代訓練一系列的模型，下一個模型補強前一個模型的不足，每個模型的訓練樣本分佈由前一個模型的結果產生 - 梯度提升決策數(Gradient Boosting Decision Tree)

Day_043: 隨機森林(Random Forest)        
Day_044: 隨機森林(Random Forest) - 程式碼   
         from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor    
         (1) 迴歸問題： RFR = RandomForestRegressor( )  
         (2) 分類問題： RFC = RandomForestClassifier( ) 
         
Day_045: 梯度提升機(Grandient Boosting Machine)       
Day_046: 梯度提升機(Grandient Boosting Machine) - 程式碼   
         from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor   
         (1) 迴歸問題： GBR = GradientBoostingRegressor( )  
         (2) 分類問題： GBC = GradientBoostingClassifie( ) 
                  
Day_047: 參數調整(Fine-Tuning)  
         from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  
         (1) 網格搜尋(Grid Search)：GridSearchCV( )  
         (2) 隨機搜尋(Random Search)：RandomizedSearchCV( )    
         
Day_049: 混合泛化(Blending)  
         對於相同的訓練集資料，結合多個/多種不同的分類模型，將每個模型的預測值加權合成，得出新的預測結果。  
         若每個模型的權重相同，則亦稱投票泛化(Voting)。  
  
註：於分類問題中，若我們想得到 lebal = 1 的預測機率，則可用 .predict_proba( data )[ :, 1 ]  得到機率值！！  
  
         
Day_50: 堆疊泛化(Stacking)  
        不僅將多個/多種不同模型結合，且使用每個模型的預測值，作為新的特徵值。  
  

分群(Clustering)  
Day_55: K-Mean  
        將資料分成 k 個群簇(cluster)，使得在同一群簇中的資料彼此盡量相近，且不同群簇的資料盡量不同。  
        
        程式碼：
        from sklearn.cluster import KMeans
        KMeans = KMeans( n_clusters, init, n_init )
        
        
Day_56: 輪廓分析(Silhouette Analysis)  
        用來評估資料分群的適當性。  
        (1) 輪廓分數(Silhouette Score) = (b_i - a_i)/max{ b_i, a_i },   
        其中 a_i : 對任一筆資料，與其同一群簇資料的平均距離；b_i : 對任一筆資料，不同群簇的資料與其平均距離的最大值。  
        ＊ 輪廓分數越大，表示越能表現每個群簇中資料點越近，且不同群簇彼此相距越遠的效果！！
   
   
Day_57: 階層分群(Hierarchical Clustering)  
        可在不定義分群個數只定義距離的情況做分群(Bottom-down)，但不適合應用於大量的資料上。 
        
        程式碼：  
        from sklearn.cluster import AgglomerativeClustering
        HC = AgglomerativeClustering( n_clusters, linkage )  
        HC.fix( data )

         
Day_58: 2D 樣版資料集(2D Toy Dataset)  
        利用圖形的視覺呈現而非量化(如：輪廓分析)，來評估非監督式模型的表現。  
        
        
Day_59: 主成份分析(Principal Component Analysis, PCA)    
        一種將資料降維到特定維度的方法，有助於加速機器學習演算法；降維後新特徵為舊特徵的線性組合，係一種線性降維的方法。  
        註：於監督式學習中，不建議在一開使就用 PCA ，因可能造成失去重要的特徵導致模型 underfitting。  
  
        程式碼：  
        from sklearn import decomposition
        PCA = decomposition.PCA( n_components ) 
        PCA.fit( data )
        New_Features = PCA.transform( data )
        
        
Day_60: 手寫辨識資料集(Modified National Institute of Standards and Technology Datasets, MNIST)    

Day_61: t-SEN  
        亦是一種降維方法，對於高維度的資料用 Gaussian 分布、低維度資料用 t 分布來近似，再透過 KL Divergence 計算其相似度，以梯度下降(Gradient Descent)求最佳解。
        
        程式碼：  
        from sklearn import manifold
        tSEN = manifold.TSNE( n_components, early_exaggeration, random_state, init = 'pca', learning_rate ) 
        data_tSEN = tSEN.fit_transform( data)
        
        
Day_62: 流形還原  
將高維度中相近的點，對應至低維度空間中，盡可能保持資料點彼此間的距離關係。比方說，若資料結構像瑞士卷一般，則流形還原就是將它攤開且鋪平。  
  
  
深度學習(Deep Learning)  
Day_63 ~ Day_065:       
    (1) 批次大小(Batch Size): 越小，學習曲線較震盪，但收斂速度較快。    
    (2) 學習率(Learnng Rate): 越大，學習曲線較震盪，但收斂速度較快；但選擇過大時，可能造成無法收斂的情況。  
    (3) 正規化(Regularization): 在 L1/L2 正規化非深度學習上的效果較為明顯，而正規化參數較小才有效果。  
    (4) 隱藏層(Hidden Layer)的層數不多時，啟動函數(Activation Function)選用 Sigmoid / Tanh 的效果較 Relu 好。但實務上，Relu 所需的計算時間短，而 Sigmoid 需大量的計算時間。  
    註：深度學習體驗平台 TensorFlow Playground  https://playground.tensorflow.org    
    
    
    
    

        
        
        
        
        
        
        

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
         Data.shape              # 資料 row 與 column 的數目  
         Data.columns            # 資料的欄位名稱  
         Data.types              # 資料欄位的型態( object, int, float )
         Data.info( )            # 資料欄位資訊( 可知欄位有無 Missing Values )  
         Data.isnull( ).sum( )   # 資料各欄位的 Missing Values 總數


#### Day_003: 建立新的 Dataframe、Try-Except 例外處理  
   * 建立 dataframe  
    
         程式碼：
         DataFrame = pd.DataFrame( Data, columns )
          
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


#### Day_004: 獨熱編碼(OneHot Encoder)、標籤編碼(Label Encoder)  * 參考 Day_022 說明
當類別(/離散)型特徵的取值之間無大小關係時(如：星期)，可利用 OneHot Encoder 將特徵扁平化 :

         程式碼：  
         from sklearn.preprocessing import OneHotEncoder
         Col = ''     # 輸入欄位名稱
         Subset_Data = Data[ [ Col ] ]  
         OneHot_Data = pd.get_dummies( Subset_Data )

當類別(/離散)型特徵的取值之間有大小關係時(如：尺寸)，可利用 Label Encoder 將特徵扁平化 :  
   * 方法一：

         程式碼：  
         from sklearn.preprocessing import LabelEncoder
         Col = ''     # 輸入欄位名稱
         Subset_Data = Data[ Col ]  # 篩選資料
         Label_Data = LabelEncoder( ).fit_transform( Subset_Data )

   * 方法二：

         程式碼：  
         mapping = { 'XL': 3, 'L': 2, 'M': 1 }
         Data[ '欄位名稱_2' ] = df[ '欄位名稱_1' ].map( mapping )

Reference: https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621


#### Day_005: 平均值、標準差、最大值、最小值 及 直方圖  

         程式碼：  
         Data[ [ '欄位名稱' ] ].mean( )   # 平均值
         Data[ [ '欄位名稱' ] ].std( )    # 標準差
         Data[ [ '欄位名稱' ] ].min( )    # 最小值
         Data[ [ '欄位名稱' ] ].max( )    # 最大值
         
         # 繪製直方圖
         Data[ '欄位名稱' ].hist( bins , label = '', color )  
         plt.title( '' )     # 直方圖標題
         plt.xlabel( '' )    # x 軸標籤
         plt.ylabel( '' )    # y 軸標籤
         plt.legend( )
         plt.show( )


#### Day_006: 篩選數值型欄位、利用圖形尋找 Outliers( 盒鬚圖、ECDF、直方圖 )  
   * Dataframe 的欄位型態統計  
   
         程式碼：  
         Col_Type = Data.dtypes.to_frame( ).reset_index( )
         Col_Type = pd.value_counts( Col_Type[ 'Type' ] ).reset_index( )
         for i in np.arange( Col_Type.shape[0] ):
             print( str( Col_Type.iloc[ i, 0 ] ) + ' 型態的欄位有 ' + str( Col_Type.iloc[ i, 1 ] ) + ' 欄' )

   * 篩選數值型態的欄位
   
         程式碼： 
         # 篩選數值型態的欄位
         dtype_select = [ np.dtype( int ), np.dtype( float ) ]
         numeric_cols = list( Data.columns[ list( Data.dtypes.isin( dtype_select ) ) ] )
         # 排除只有 2 個值的欄位( 例如：0 跟 1 ) 
         numeric_cols = list( data[ numeric_cols ].columns[ list( data[ numeric_cols ].apply( lambda x :len( x.unique() )!= 2 ) ) ] )


#### Day_007: Outlier的處理( 以平均值、中位數、最小值、最大值或眾數 取代  )  
   * 百分位數(Quantile)
         
         程式碼：
         # 計算第 0 到 第 100 的百分位數
         q_all = [ np.percentile( data[ ~data[ Missing_col ].isnull( ) ][ Missing_col ], q = i ) for i in np.arange( 0, 101 ) ] 
         Qth_Percentile = pd.DataFrame( { 'q': list( range( 101 ) ), 'value': q_all } )
         Qth_Percentile
         
   * 眾數(Mode)
   
         程式碼：
         from collections import defaultdict  
         Col = ' '    # 輸入計算眾數的欄位名稱
         
         mode_dict = defaultdict( lambda : 0 )
         for value in data[ ~data[ Missing_col ].isnull( ) ][ Missing_col ] :
             mode_dict[value] += 1
         Mode = sorted( mode_dict.items( ), key = lambda kv : kv[ 1 ], reverse = True )
         
         print( 'Mode of ' + str( Col ) + ' = ' + str( Mode[ 0 ][ 0 ] ) )


#### Day_008: 資料分組離散化  
   * 等距分組  
   
         程式碼：
         Col = ' '    # 輸入等距分組的欄位名稱
         
         Maximum = Data[ Col ].values.max( )
         cut_rule = [ 0, 2, 4, Maximum ]    # 分割區間：( 0, 2 ]、( 2, 4 ]、( 4, Maximum ]
         Data[ str(Col)+'Group' ] = pd.cut( Data[ Col ].values, cut_rule, include_lowest = True, precision = 1 )
         Data[ str(Col)+'Group' ].value_counts( )


#### Day_009: 計算相關係數( Correlation Coefficient )、散佈圖( Scatter Plot )檢視相關性  
   * 相關係數
   
         程式碼：
         Data.corr( )  # 資料欄位彼此的相關係數


#### Day_010: 計算相關係數( Correlation Coefficient )、散佈圖( Scatter Plot )或盒鬚圖( Box Plot )檢視相關性  

         程式碼：
         Data.corr( )['TARGET']  # 目標欄位(TARGET)與所有欄位的相關係數  
         
         # 相關係數最大的前 5 個欄位
         Data.corr( )[ 'TARGET' ].sort_values( ascending = False ).head(  )


#### Day_011: 資料分組離散化、長條圖( Bar Plot )  
   * 等距分組
   
         程式碼：
         Col = ' '    # 輸入等距分組的欄位名稱
         
         bin_cut =  np.linspace( 0, 50, num = 11, dtype = 'int64' )                # 分割區間 = [0 15 20 25 30 35 40 45 50]
         Data[ str(Col)+'Group' ] = pd.cut( Data[ Col ], bins = bin_cut )  
         Group_Counts = Data[ str(Col)+'Group' ].value_counts( ).reset_index( )    # 計算每個分組中的資料總數
         Group_Counts.columns = [ str(Col)+'Group', 'Counts' ]

   * 長條圖(Bar Plot)
   
         程式碼：
         x = Group_Counts[ str(Col)+'Group' ]
         y = Group_Counts[ 'Counts' ]
         
         plt.figure( figsize = ( 8, 8 ) )
         sns.barplot( x, y )
    
         plt.title('')     # 長條圖的標題
         plt.xticks( rotation = 75 ) 
         plt.xlabel( '' )
         plt.ylabel( '' )
         

#### Day_012: 資料分組離散化
   * 等頻分組（ 可得知 0%、25%、50%、75%、100% 百分位數 ）
   
         程式碼：
         Col = ' '    # 輸入等頻分組的欄位名稱
         
         Data[ str(Col)+'Group' ] = pd.qcut( Data[ Col ], 4 )  # 依照 quartiles [0, 0.25, 0.5, 0.75, 1] 分割
         Data[ str(Col)+'Group' ].value_counts( )              # 每個分組的資料總數皆會相同！！


#### Day_013: 資料分組離散化  

#### Day_014: 繪圖排版 subplots   

#### Day_015: Heatmap、Gridplot  
Heatmap 常用於呈現特徵間的相關性，也可用於呈現不同條件下，數量的大小關係。

         # 將欄位彼此的相關係數，利用 Heatmap 視覺化
          
         程式碼：
         plt.figure( figsize = ( 10, 10 ) )
         sns.heatmap( Data.corr( ), cmap = plt.cm.RdYlBu_r, annot = True, fmt = '.1f', ax = axs )
         plt.show( )
          
   註 1：隨機生成數值落於 ( a, b ) 的隨機矩陣
         
         程式碼：
         Matrix = ( b - a ) * np.random.random( ( nrow, ncol ) ) + a 
      
   註 2：隨機生成符合常態分配的隨機矩陣
         
         程式碼：
         Matrix = np.random.randn( nrow, ncol ) 
        
         
#### Day_016: 匯出儲存成 csv 檔  
 
         程式碼：
         Data.to_csv( '檔名.csv', index = False )
         
 
#### Day_017: 特徵工程(Feature Engineering)簡介 
(1) 數值型特徵
   * Step 1 : 填補缺漏值(Missing Values)
   * Step 2 : 去除或調整離群值(Outliers)
   * Step 3 : 去除偏態  
   * Step 4 : 特徵縮放(標準化、最小最大化)  
   * Step 5 : 特徵組合
   * Step 6 : 特徵篩選與評估  
     
(2) 類別型特徵  
   * 標籤編碼(Label Encoding) : 將類別依序編上編號
   * 獨熱編碼(OneHot Encoding) : 將每個不同的類別分獨立為一欄 
   * 均值編碼(Mean Encoding) : 利用目標值(Target)的平均值，取代類別型特徵   
     註：程式碼參考 Day_023  
   * 計數編碼(Counting Encoding) : 若類別的目標均值與其總比數正(\負)相關，則可利用每個類別的總筆數，取代類別型特徵  
   * 特徵雜湊(Feature Hash) : 相異類別的數量非常龐大時使用(例如：姓名)   

(3) 時間型特徵


#### Day_018: 特徵的類型
(1) 數值型  
(2) 類別型  
(3) 二元特徵：可視為數值型也可視為類別型(例：True = 1/ False = 0)  
(4) 排序型：例如名次、百分等級等有大小關係，通常以數值型特徵處理，因若視為類別型處理，將會失去排序的資訊。  


#### Day_019: 數值型特徵  
填補 Missing Values ( 盡量不要改變資料的分佈情況！ )
   * 平均值：資料偏態不明顯時  
   * 中位數：資料有很明顯偏態時  
   
標準化(Standard Scaler)及最大最小化(MinMax Scaler) 
   * 假定資料符合常態分配，適合使用標準化做特徵縮放。  
   * 假定資料符合均勻分配，適合使用最大最小化做特徵縮放。  
   * 標準化較不易受極端值的影響。
   * 若使用最大最小化，需注意資料是否有極端值。  
   * 樹狀模型(如：決策樹、隨機森林、梯度提升機)：標準化及最大最小化後對預測不會有影響。
   * 非樹狀模型(如：線性迴歸、邏輯斯迴歸、類神經網絡等)：標準化及最大最小化後對預測會有影響。

         程式碼：
         from sklearn.preprocessing import StandardScaler, MinMaxScaler
         Col = ''   # 輸入資料標準化(/最大最小化)的欄位名稱
         
         # 標準化
         Data[ Col ] = StandardScaler( ).fit_transform( Data[ Col ].values.reshape( -1, 1 ) )
         
         # 最大最小化
         Data[ Col ] = MinMaxScaler( ).fit_transform( Data[ Col ].values.reshape( -1, 1 ) )
         

#### Day_020: 離群值的處理 
捨去離群值：若離群值僅有少數幾筆資料時，此方法不至於對原始資料的分佈造成變化。

         程式碼：
         Col = ''            # 輸入欄位名稱
         Upper_Bound =       # 資料的上限
         Lower_Bound =       # 資料的下限
         
         keep_indexs = ( Data[ Col ] < Upper_Bound ) & ( Data[ Col ] > Lower_Bound )
         Data = Data[ keep_indexs ]

調整離群值：

         程式碼：
         Col = ''            # 輸入欄位名稱
         
         Data[ Col ] = Data[ Col ].clip( Lower_Bound, Upper_Bound ) # 將大於上限的數值調整成上限，小於下限的數值調整成下限
         

#### Day_021: 去偏態 
偏態常出現於非負且可能為 0 的欄位(例如：價格、計數等)  

自然對數(log)去偏：

         程式碼：
         Col = ''            # 輸入欄位名稱
         Data[ Col ] = np.log1p( Data[ Col ] )     # y = log( 1 + x ) 可將 0 對應到 0

分布去偏(boxcox)：

         程式碼：
         from scipy import stats
         
         Col = ''            # 輸入欄位名稱
         Data[ Col ] = stats.boxcox( Data[ Col ], lambda )   
         
         # lambda = 0   : logarithmic transformation  log(X)  
         # lambda = 0.5 : square root transformation  sqrt(X)          

Reference: https://www.itread01.com/content/1543890427.html  


#### Day_022: 標籤編碼(Label Encoding)、獨熱編碼(OneHot Encoding)  * 程式碼參考 Day_004
類別型特徵建議採用標籤編碼，若該特徵重要性高且類別值少，則可考慮使用獨熱編碼。


#### Day_023: 均值編碼(Mean Encoding) 
利用目標值(Target)的平均值，取代類別型特徵。容易造成模型 overfitting!!

         程式碼：
         target_col = ''     # 輸入目標欄位名稱
         Col = ''            # 輸入均值編碼的欄位名稱
         
         target_mean = Data.groupby( [ Col ] )[ 'Survived' ].mean( ).reset_index( )
         target_mean = [ Col, f'{c}_mean' ]                                 


#### Day_024: 計數編碼(Counting Encoding)、特徵雜湊(Feature Hash)   

#### Day_025: 時間特徵分解( 年、月、日、時、分、秒 ) 及 週期循環特徵( 利用 sin 或 cos 函數執行 )  

#### Day_026: 特徵組合( ex: 經緯度座標 )  

#### Day_027: 特徵組合( Group by Encoding 群聚編碼 : 合成類別特徵與數值特徵 )  

#### Day_028: 特徵篩選    
(1) 相關係數過濾法 : 利用 corr() + list() + pop() 函數   
(2) L1-Embedding(Lasso Regression Embedding) : 利用 Lasso( alpha = ) 函數  
(3) GDBT(梯度提升樹) Embedding    
  
#### Day_029: 樹狀模型的特徵重要性( estimator.feature_importances_ )  


#### Day_030: 分類預測模型的特徵優化( 隨機森林 Random Forset + 葉編碼 Leaf Encoding + Logistic Regression )  


#### Day_034: 切分 訓練集/測試集 資料    
(1) 資料切分 : train_test_split() 函數   
(2) 交叉採樣 : KFold( n_splits = ,shuffle = False ) 函數   
   * n_splits 為等份數；shuffle = False 表示每次劃分的結果相同  
 
 註：當樣本不均衡時，須搭配運用的函數 np.where()、np.concatenate()  


#### Day_035: 模型的選擇 by 預測類型  
(1) 迴歸問題：預測的目標為實數  
(2) 分類問題：預測的目標為類別  
(3) 二元分類(binary-class): 瑕疵 vs 正常  
(4) 多元分類(multi-class): 手寫辨識1~9 ；    
(5) 多標籤(multi-label): 如天氣預測多雲時晴  
(6) 迴歸問題可轉化成分類問題


#### Day_036: 模型的評估指標  
(1) 迴歸問題：MAE(Mean Absolute Error)、MSE(Mean Square Error)、R-Square  
(2) 分類問題：AUC(Area Under Curve)、Precision、Recall、F1-Score  
  
  
#### Day_037: 線性迴歸(Linear Regression) & 邏輯斯迴歸(Logistic Regression)  

         程式碼    
         from sklearn import linear_model  
         (1) 線性：Linear = linear_model.LinearRegression( )  
         (2) 邏輯斯：Logistic = linear_model.LogisticRegression( )   
     
     
#### Day_039: Lasso & Ridge Regression  
(1) 正則化(Regularization): 避免模型過擬合(over-fitting)  
(2) 正則化方法：Lasso(L1)、Ridge Regression(L2)   

         程式碼     
         from sklearn import linear_model  
         (1) Lasso = linear_model.Lasso( alpha =  )  
         (2) Ridge = linear_model.Ridge( alpha =  )  
  
  
#### Day_041: 決策樹(Decision Tree)     
 
         程式碼     
         from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    
         (1) 迴歸問題： DTR = DecisionTreeRegressor( )  
         (2) 分類問題： DTC = DecisionTreeClassifier( )  
  
   
### 集成學習(Ensemble Learning): 
可避免當決策樹足夠深時，容易導致過擬合(overfitting)的缺點   
> (i) Bagging 從原資料集中隨機做取後放回的採樣，分別在採樣的子集訓練模型 - 隨機森林(Random Forest)   
> (ii) Boosting 透過迭代訓練一系列的模型，下一個模型補強前一個模型的不足，每個模型的訓練樣本分佈由前一個模型的結果產生 - 梯度提升決策數(Gradient Boosting Decision Tree)


#### Day_043: 隨機森林(Random Forest)        

         程式碼   
         from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor    
         (1) 迴歸問題： RFR = RandomForestRegressor( )  
         (2) 分類問題： RFC = RandomForestClassifier( ) 


#### Day_045 ~ Day_046: 梯度提升機(Grandient Boosting Machine)       
         
         程式碼   
         from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor   
         (1) 迴歸問題： GBR = GradientBoostingRegressor( )  
         (2) 分類問題： GBC = GradientBoostingClassifie( ) 
 
 
#### Day_047: 參數調整(Fine-Tuning)  
         
         程式碼：
         from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  
         
         # 網格搜尋(Grid Search)
         GridSearchCV( )   
         
         # 隨機搜尋(Random Search) 
         RandomizedSearchCV( )  
         
         
#### Day_049: 混合泛化(Blending)  
對於相同的訓練集資料，結合多個/多種不同的分類模型，將每個模型的預測值加權合成，得出新的預測結果。若每個模型的權重相同，則亦稱投票泛化(Voting)。  
   註：於分類問題中，若我們想得到 lebal = 1 的預測機率，則可用 .predict_proba( data )[ :, 1 ]  得到機率值！！  
  
         
#### Day_050: 堆疊泛化(Stacking)  
不僅將多個/多種不同模型結合，且使用每個模型的預測值，作為新的特徵值。  
  

### 分群(Clustering)  
#### Day_055: K-Mean  
將資料分成 k 個群簇(cluster)，使得在同一群簇中的資料彼此盡量相近，且不同群簇的資料盡量不同。  
        
        程式碼：
        from sklearn.cluster import KMeans
        KMeans = KMeans( n_clusters, init, n_init )
        
        
#### Day_056: 輪廓分析(Silhouette Analysis)  
用來評估資料分群的適當性。  
輪廓分數(Silhouette Score) = (b_i - a_i)/max{ b_i, a_i }，其中 a_i : 對任一筆資料，與其同一群簇資料的平均距離；b_i : 對任一筆資料，不同群簇的資料與其平均距離的最大值。  
   * 輪廓分數越大，表示越能表現每個群簇中資料點越近，且不同群簇彼此相距越遠的效果！！
   
   
#### Day_057: 階層分群(Hierarchical Clustering)  
可在不定義分群個數只定義距離的情況做分群(Bottom-down)，但不適合應用於大量的資料上。 
        
        程式碼：  
        from sklearn.cluster import AgglomerativeClustering
        HC = AgglomerativeClustering( n_clusters, linkage )  
        HC.fix( data )

         
#### Day_058: 2D 樣版資料集(2D Toy Dataset)  
利用圖形的視覺呈現而非量化(如：輪廓分析)，來評估非監督式模型的表現。  
        
        
#### Day_059: 主成份分析(Principal Component Analysis, PCA)    
一種將資料降維到特定維度的方法，有助於加速機器學習演算法；降維後新特徵為舊特徵的線性組合，係一種線性降維的方法。   
註：於監督式學習中，不建議在一開使就用 PCA ，因可能造成失去重要的特徵導致模型 underfitting。  
  
        程式碼：  
        from sklearn import decomposition
        PCA = decomposition.PCA( n_components ) 
        PCA.fit( data )
        New_Features = PCA.transform( data )
        
        
#### Day_060: 手寫辨識資料集(Modified National Institute of Standards and Technology Datasets, MNIST)    

#### Day_061: t-SEN  
亦是一種降維方法，對於高維度的資料用 Gaussian 分布、低維度資料用 t 分布來近似，再透過 KL Divergence 計算其相似度，以梯度下降(Gradient Descent)求最佳解。
        
        程式碼：  
        from sklearn import manifold
        tSEN = manifold.TSNE( n_components, early_exaggeration, random_state, init = 'pca', learning_rate ) 
        data_tSEN = tSEN.fit_transform( data)
        
        
#### Day_062: 流形還原  
將高維度中相近的點，對應至低維度空間中，盡可能保持資料點彼此間的距離關係。比方說，若資料結構像瑞士卷一般，則流形還原就是將它攤開且鋪平。  
  
  
### 深度學習(Deep Learning)  
#### Day_063 ~ Day_065:       
(1) 批次大小(Batch Size): 越小，學習曲線較震盪，但收斂速度較快。    
(2) 學習率(Learnng Rate): 越大，學習曲線較震盪，但收斂速度較快；但選擇過大時，可能造成無法收斂的情況。  
(3) 正規化(Regularization): 在 L1/L2 正規化非深度學習上的效果較為明顯，而正規化參數較小才有效果。  
(4) 隱藏層(Hidden Layer)的層數不多時，啟動函數(Activation Function)選用 Sigmoid / Tanh 的效果較 Relu 好。但實務上，Relu 所需的計算時間短，而 Sigmoid 需大量的計算時間。  
   * 註：深度學習體驗平台 [TensorFlow Playground](https://playground.tensorflow.org)    


#### Day_066: Keras  
安裝流程 for macOS 
   * Step 1 : 開啟終端機  
   * Step 2 : 輸入  export PATH=~/anaconda/bin:$PATH ，顯示 Anaconda 環境變量  
   * Step 3 : 輸入  conda create -n tensorflow python=3.7 anaconda ，建立 Anaconda 虛擬環境
   * Step 4 : 輸入  activate tensorflow ，啟動 Anaconda 虛擬環境  
   * Step 5 : 輸入  pip install tensorflow==2.0.0-beta1 ，安裝 Tensorflow
   * Step 6 : 輸入  pip install --ignore-installed --upgrade keras ，安裝 Keras  
   * Step 7 : 輸入  conda env list ，確認環境   
   
Reference :   
(1) [TensorFlow](https://www.tensorflow.org)  
(2) [Keras](https://keras.io)


#### Day_067: Keras 內建資料集  


#### Day_068: Keras 模型搭建  
卷積神經網絡(CNN)  

        範例程式碼：  
        import keras
        from keras.datasets import cifar10  # Keras 內建資料集
        from keras.utils import np_utils    # OneHot Encoding  

        # Sequential( ) : 空的模型物件，用於建立一系列模型
        from keras.models import Sequential, load_model              

        # Conv2D : 平面的卷積模組
        from keras.layers import Conv2D                     

        # MaxPooling2D : 平面池化模組
        from keras.layers import MaxPooling2D            

        # Flatten：為了全連接層運算，建立平坦層
        from keras.layers import Flatten                              

        # Dense : 建立全連接層(fully-connected layer)；Activation：激活函數；Dropout：隨機拋棄避免過擬合
        from keras.layers import Dense,  Activation, Dropout  
        
        
        # 資料預處理
        # step 1 : 陣列資料轉換
        # num1 = 資料圖片的張數；num2 = 解析度；num3 = 解析度；num4 = 色版數目( 例如：RGB 為 3 )  
        x_train = x_train_image.reshape( num1, num2, num3, num4 ).astype( 'float32' )  
        x_test = x_test_image.reshape( num1, num2, num3, num4 ).astype( 'float32' )  
        
        # step 2 : 資料標準化(Normalization)  
        x_train_normalize = x_train / 255  # 除以畫素最大值 255
        x_test_normalize = x_test / 255    # 除以畫素最大值 255
        
        # step 3 : 獨熱編碼(OneHot Encoding)  
        y_train_onehot = np_utils.to_categorical( y_train_label )
        y_test_onehot = np_utils.to_categorical( y_test_label )
        
        
        # 建立 CNN 模型
        model = Sequential( )

        model.add( Conv2D( filters,                        # filter( 又稱 kernel ) 的個數  
                           kernel_size = ( , ),            # filter( 又稱 kernel ) 的個尺寸
                           padding = 'same',               # padding : 邊界周圍補 0 且 filter 的步伐(stride) 為 1
                           input_shape = x_train.shape[ 1: ] )
               )
        
        model.add( MaxPooling2D( pool_size = ( 2, 2 ) ) )  # 最大池化層( 池化核心 2 * 2)

        model.add( Flatten( ) )              # 資料扁平化，為了後續全連接層(fully-connected layer)運算
        model.add( Dense( 512 ) )            # 建立有 512 個神經元的隱藏層
        model.add( Activation( 'relu' ) )    # 激活函數
        model.add( Dropout( 0.5 ) )          # 隨機停止 50% 的神經元運作
        model.add( Dense( 256 ) )            # 建立有 256 個神經元的隱藏層
        model.add( Activation( 'tanh' ) )    # 激活函數
        model.add( Dropout( 0.7 ) )          # 隨機停止 70% 的神經元運作
        model.add( Dense( num_classes ) )    # num_classes : 分類數目
        model.add( Activation( 'softmax' ) ) # 標準化指數層(softmax layer)
        
        
Reference :   
(1) [Convolutional Layers](https://keras.io/layers/convolutional/)  
(2) [Pooling Layers](https://keras.io/layers/pooling/)  
(3) [Activations](https://keras.io/activations/)  
(4) [Using the Keras Flatten Operation in CNN Models with Code Examples](https://missinglink.ai/guides/deep-learning-frameworks/using-keras-flatten-operation-cnn-models-code-examples/)        
(5) [使用 Keras 卷積神經網路 (CNN) 辨識手寫數字](http://yhhuang1966.blogspot.com/2018/04/keras-cnn.html)   


#### Day_069: Keras 函數式 API  
定義複雜模型(例如：多個輸出模型、具有共享層的模型等)的方法  

        範例程式碼：  
        import keras
        from keras.datasets import cifar10  # Keras 內建資料集
        
        from keras.layers import Input, Embedding, LSTM, Dense  
        from keras.utils import np_utils    # OneHot Encoding 
        from keras.models import Model 
        
        import matplotlib.pyplot as plt
        from keras.utils import plot_model
        from IPython.display import Image
        
        
        Main_Input = Input( shape = ( num1, ), dtype = 'int32', name = 'Main_Input' )  
        
        # Embedding Layers ( This layer can only be used as the first layer in a model. )
        Embedding = Embedding( output_dim = num2 , input_dim = num3, input_length = num4 )( Main_Input )
        
        # LSTM Layer
        LSTM_output = LSTM( 32 )( Embedding )  
        
        Input_2 = Input( shape = ( num5, ), name = 'Input_2' )
        Merge_Layer = keras.layers.concatenate( [ LSTM_output, Input_2 ] )  
        
        # 堆疊多個全連接網路層
        Hidden1 = Dense( 64, activation = 'relu', name = 'Hidden1' )( Merge_Layer )
        Hidden2 = Dense( 64, activation = 'relu', name = 'Hidden2' )( Hidden1 )  
        
        Main_Output = Dense( 1, activation = 'sigmoid', name = 'Main_Output' )( Hidden2 )  
        
        # 宣告 MODEL API, 分別採用自行定義的 Input/Output Layer
        model = Model( inputs = [ Main_Input, Input_2 ], outputs = Main_Output )
        
        # 模型結構總攬
        model.summary( )
        
        # 模型網絡的流程圖  
        plot_model( model, to_file = 'Model_Name.png' )
        Image( 'Model_Name.png' )
        
    
Reference :   
(1) [Getting started with the Keras functional API](https://keras.io/getting-started/functional-api-guide/)  
(2) [Embedding](https://keras.io/layers/embeddings/)   
(3) [Merge Layers](https://keras.io/layers/merge/)  
(4) [如何使用 Keras 函數式 API 進行深度學習](https://zhuanlan.zhihu.com/p/53933876)

    
#### Day_070: 多層感知器(Multi-layer Perceptron)  
為深度神經網絡的一種特例，係一種向前傳播遞迴的類神經網絡，且利用向後傳播的技術達到學習的目標。
        
        範例程式碼：  
        import keras
        from keras.datasets import mnist    # Keras 手寫辨識資料集  
        
        from keras.utils import np_utils    # OneHot Encoding  
        from keras.models import Sequential
        from keras.layers import Dense
        import matplotlib.pyplot as plt  
        
        # 載入資料
        ( x_train_image, y_train_label ), ( x_test_image, y_test_label ) = mnist.load_data( )
        
        x_Train = x_train_image.reshape( 60000, 784 ).astype( 'float32' )  # 784 = 28 * 28
        x_Test = x_test_image.reshape( 10000, 784 ).astype( 'float32' )
        
        # 資料標準化
        x_Train_normalize = x_Train / 255
        x_Test_normalize = x_Test / 255
        
        # OneHot Encoding
        y_Train_OneHot = np_utils.to_categorical( y_train_label ) 
        y_Test_OneHot = np_utils.to_categorical( y_test_label )
        
        # 建立模型
        model = Sequential( )

        # 1.輸入層
        model.add( Dense( units = 256,                     # 神經元數量 
                          input_shape = ( 784, ), 
                          kernel_initializer = 'normal',   # 初始化權重的方法
                          activation = 'relu', 
                          name = 'Input_layer'
                           ) )

        # 2.隱藏層
        model.add( Dense( 128, kernel_initializer = 'normal', activation = 'relu', name = 'Hidden_1' ) )    # 建立有 128 個神經元的隱藏層
        model.add( Dense( 64, kernel_initializer = 'normal', activation = 'tanh', name = 'Hidden_2' ) )     # 建立有 64 個神經元的隱藏層

        # 3.輸出層
        model.add( Dense( units = 10,                           # 神經元數量
                          kernel_initializer = 'normal',    # 初始化權重的方法
                          activation = 'softmax', 
                          name = 'Output_Layer'
                           ) )
                           
         # 模型摘要
         print( model.summary( ) )
         
         # 訓練模型
         model.compile( loss = 'categorical_crossentropy',   # 損失函數(Loss Function)
                        optimizer = 'adam',                  # 最佳化的演算法
                        metrics = [ 'accuracy' ]             # 模型成效的評量指標
                        )
         train_history = model.fit( x = x_Train_normalize,
                                    y = y_Train_OneHot, 
                                    validation_split = 0.2,   # 驗證集佔訓練集的比例
                                    epochs = 10,              # 模擬次數
                                    batch_size = 20,          # 每批次的資料筆數
                                    verbose = 1               # 顯示模型訓練進度 
                                    )
                                    
          # 評估模型
          scores = model.evaluate( x_Test_normalize, y_Test_OneHot )
          print( 'accuracy = ', scores[1] )  
          
          # 測試集資料預測
          prediction = model.predict_classes( x_Test_normalize )
          # 混淆矩陣
          pd.crosstab( y_test_label, prediction, rownames = [ 'label' ], colnames = [ 'predict' ] )
          

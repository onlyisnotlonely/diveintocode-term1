import numpy as np
import random

def train_test_split(X, y, train_size = 0.8, random_state = 1):
    """
    データセットを学習用データと検証用データにランダムに分割する。
    
    Parameters
    ----------
    X：次の形のndarray, shape(n_samples, n_features)
    　学習データ（特徴量）
    
    y：次の形のndarray, shape(n_samples, )
    　正解値
    
    train_size：float （0 < train_size < 1）
     学習データの割合
    
    random_state : int
    　乱数ジェネレーターのシード値
     
    
    Returns
    ----------
    X_train：次の形のndarray, shape(n_samples, n_features)
     学習データ
     
    X_test：次の形のndarray, shape(n_samples, n_features)
     検証データ
     
    y_train：次の形のndarray, shape(n_samples, )
     学習データの正解値
    
    y_test：次の形のndarray, shape(n_samples, )
     検証データの正解値
    
    """ 
    
    
    """
    1. データ、パラメータの妥当性検証
    """
    
    # 0. Numpy配列に変換(データ型がpandas形式だと、split時にエラーが発生するため)
    X = np.array(X)
    y = np.array(y)   
    
    # 1. サンプル数が１以上のデータが入力されていることを確認
    if (len(X) == 0) and (len(y) == 0):
        raise ValueError("At least one array required as input.")
        
    # 2. X, yのindex総数が異なる場合、アラートを出す
    if len(X) != len(y):
        n_array = [len(X), len(y)] # サンプル数を格納した配列
        raise ValueError("Found input variables with inconsistent numbers of samples: {}".format(n_array))
    
    # 3. サンプル数より多いtrain_sizeは設定不可
    if type(train_size) == int:
        if train_size >= min(len(X), len(y)):
            print("train_size={0} should be smaller than the number of samples {1}".format(train_size, min(len(X), len(y))))
    
    
    """
    2. 与えた配列、パラメータが条件を満たしている時、以下を実行する
    """
    
    # 分割数
    if type(train_size) == int:
        n_train = train_size
        n_test = len(X) - train_size
    
    if type(train_size) == float:
        n_train = int(np.floor(len(X) * train_size)) # 小数点は切り捨て、整数型へ変換
        n_test = len(X) - train_size
        
    # ランダムに分割
    random.seed(random_state) # シード値を設定
    index = random.sample([i for i in range(len(X))], len(X)) # ランダムに並び替えたインデックスの配列を生成
    
    index_train = index[0 : n_train] # indexのうち、最初のn_train個を学習用に用いる
    index_test = index[n_train:]      # 残りを検証用に用いる
    
    # 分割後の説明変数
    X_train = X[index_train] # 学習データ
    X_test = X[index_test]   # 検証データ
    
    # 分割後の目的変数
    y_train = y[index_train]  # 学習データ
    y_test = y[index_test]   # 検証データ
    
    return X_train, X_test, y_train, y_test
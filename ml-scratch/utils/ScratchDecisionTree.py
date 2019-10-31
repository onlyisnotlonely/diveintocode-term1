import numpy as np

class ScratchDecisionTreeClassifier():
    """
    決定木のスクラッチ実装
    
    Parameters
    ----------
    max_depth : int
        決定木の深さ
    
    min_impurity_decrease : float
        ジニ不純度の閾値（決定木の分割実施条件）
    
    Attributes
    ----------
    self.node : class
        ノード
    
    """
    
    def __init__(self, max_depth = None, ):
        
        self.depth = max_depth # 決定木の深さ
        #self.node = 1# ノード
        
        self.node = None
        
        self.node.X = None
        self.node.y = None
    
    
    def fit(self, X, y):
        """
        学習(学習用データに基づく決定木の作成) 
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習用データ（特徴量）
        
        y : 次の形のndarray, shape(m_samples, n_features)
            学習用データ（正解値）
        
        
        """
        # インスタンスの生成
        self.node = Node()
        
        # 最初のノードに学習用データ（特徴量、正解値を与える）
        self.node.X = X
        self.node.y = y
        
        # 決定機の深さの最大値を与える
        self.node.max_depth = self.depth
        
        # 分岐を生成する
        self.node.split()
    
    
    def predict(self, X):
        """
        テストデータの分類の予測値を返す
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            検証用データ（特徴量）
        
        Returns
        ----------
        y_pred : 次の形のndarray, shape(m_samples, n_features)
            予測値
        
        """
        
        # 予測値を格納するリスト
        y_pred_list = []
        
        # 繰り返し処理
        for i in range(len(X)):
            y_pred = self.node.predict(X[i])
            #print("y_pred : {}".format(y_pred))
            y_pred_list.append(y_pred)

        return y_pred_list
        
        
class Node():
    """
    決定木
    
    Parameters
    ----------
    None
    
    
    Attributes
    ----------
    self.depth : int
        ノードの深さ

    self.impurity : list
        ジニ不純度
    
    self.feature : list
        分類に用いた特徴量
    
    self.threshold : list
        分類に用いた閾値
    
    self.label : 
        予測値のラベル
        
    self.left : class
        子ノード（左側：特徴量が閾値以上）
        
    self.right : class
        子ノード（右側：特徴量が閾値未満）
    
    """

    def __init__(self, ):
        
        self.max_depth = max_depth # 深さの最大値
        self.depth = 0 # ノードの深さ（何層目のノードか？）
        
        self.X = 1 # ノードに含まれる説明変数（学習時に学習用データを保存） 
        self.y = 1 # ノードに含まれる目的変数（学習時に学習用データを保存）
        
        self.feature = -1 # 分割に用いる特徴量（column）
        self.threshold = -1 # 分割に用いる特徴量の閾値
        
        self.label = None # 正解ラベル
        self.impurity = 0 #ノードのジニ不純度
        
        self.left_index = 1 # 左側の子ノードに含まれるサンプル（X）のインデックス
        self.right_index = 1 # 右側の子ノードに含まれるサンプル（X）のインデックス
        
        self.left_node = 1 # 左側の子ノード（split関数で生成）
        self.right_node = 1 # 右側の子ノード（split関数で生成）
    
    
    def calc_impurity(self, y):
        """
        ジニ不純度を求める

        Parameters
        ----------
        y : 次の形のndarray, shape(m_samples)
            正解値

        Returns
        ----------
        impurity : float
            ジニ不純度

        """
        # 目的変数の値、及び要素数を取得する
        class_value, class_count = np.unique(y, return_counts = True)

        # ジニ不純度を算出する
        if len(y) > 0:
            impurity = 1 - np.sum(np.square(class_count / len(y)))
        elif len(y) == 0:
            impurity = 0
        
        return impurity
    
    
    def information_gain(self, y, left_index, right_index):
        """
        情報利得を求める

        Parameters
        ----------
        y : 次の形のndarray, shape(m_samples,)
            学習用データの正解値

        left_index : list
            特徴量が閾値以上のサンプルのインデックス

        rifht_index : list
            特徴量が閾値未満のサンプルのインデックス

        Returns
        ----------
        info_gain : float
            情報利得

        """

        # ジニ不純度を算出する
        self.impurity = self.calc_impurity(y) # 親ノード
        impurity_left = self.calc_impurity(y[left_index]) # 閾値以上で分割されたクラス
        impurity_right = self.calc_impurity(y[right_index]) # 閾値未満で分割されたクラス
        
        # 要素数
        count_parent = len(y) # 親ノード
        count_left = len(left_index) # 閾値以上で分割されたクラス
        count_right = len(right_index) # 閾値未満で分割されたクラス
        
        # 情報利得を求める
        info_gain = self.impurity - (count_left / count_parent) * impurity_left - (count_right / count_parent) * impurity_right
        
        return info_gain
        
    
    
    
    
        
    def split(self,):
        """
        情報利得を最大化する特徴量、閾値を求めた上で、子ノードを生成する

        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習用データ（特徴量）

        y : 次の形のndarray, shape(m_samples,)
            学習用データ（正解値）

        Returns
        ----------
        upper_index : 次の形のndarray, shape(m_samples, n_features)
            学習用データのindex（特徴量が閾値以上）

        lower_index : 次の形のndarray, shape(m_samples, n_features)
            学習用データのindex（特徴量が閾値未満）

        info_gain : float 
            情報利得
        
        """
        
        """
        1. ノードに正解ラベルを割り当てる
        """
        
        value, count = np.unique(self.y, return_counts = True)
        self.label = value[np.argmax(count)]
        self.impurity = self.calc_impurity(self.y)
        
        #print("self.y : {}".format(self.y))
        #print("len.self.y : {}".format(len(self.y)))
        #print("self.label : {}".format(self.label))
        #print("self.depth : {}".format(self.depth))
        #print("self.impurity : {}".format(self.impurity))
        #print("")
        
        
        """
        2. 情報利得が最大となる特徴量と閾値の組み合わせを探索し、その組み合わせを用いてサンプルを分割する
        """
        if self.max_depth is not None:
            if (self.depth < self.max_depth) and (self.impurity > 0):
                
                # 情報利得の最大値を初期化
                max_info_gain = 0

                # 情報利得の最大値を探索
                for row in range(self.X.shape[0]):
                    for column in range(self.X.shape[1]):

                        left_index = np.where(self.X[:, column] >= self.X[row][column])[0] # 特徴量が閾値以上のサンプルのインデックス         
                        right_index = np.where(self.X[:, column] < self.X[row][column])[0] # 特徴量が閾値未満のサンプルのインデックス

                        ig  = self.information_gain(self.y, left_index, right_index) # 情報利得

                        # 情報利得が最大となる分割を求める
                        if ig > max_info_gain:
                            max_info_gain = ig # 情報利得の最大値を更新する

                            # 親ノードのインスタンス変数を更新する                        
                            self.feature = column # ノードの分割に用いる特徴量
                            self.threshold = self.X[row][column] # ノードの分割に用いる特徴量の閾値
                            self.left_index = left_index # 左側の子ノードのインデックス
                            self.right_index = right_index # 右側の子ノードのインデックス
                            
                            # 子ノードの生成（左側）
                            left_node = Node()
                            #print("left_node : {}".format(left_node)
                            left_node.depth = self.depth + 1
                            left_node.max_depth = self.max_depth
                            left_node.X = self.X[self.left_index]
                            left_node.y = self.y[self.left_index]
                            #print("left_node.y : {}".format(left_node.y))
                            #print("len_left_node.y : {}".format(len(left_node.y)))
                            #print("")
                            left_node.impurity = self.calc_impurity(left_node.y)
                            value, count = np.unique(left_node.y, return_counts = True)
                            left_node.label = value[np.argmax(count)]
                            #print("left_node : {}".format(left_node))
                            
                            # 親ノードのインスタンス変数に格納
                            self.left_node = left_node
                            
                            # 子ノードを分割する
                            self.left_node.split()
                            
                            # 子ノードの生成（右側）
                            right_node = Node()
                            #print("right_node : {}".format(right_node))

                            # インスタンス変数を追加
                            right_node.depth = self.depth + 1 # 子ノードの深さ(depth)を親ノードの1階層したとする
                            right_node.max_depth = self.max_depth # 子ノードに深さの最大値を設定
                            right_node.X = self.X[self.right_index] # 説明変数（特徴量）を追加
                            right_node.y = self.y[self.right_index] # 目的変数を追加
                            #print("right_node.y : {}".format(right_node.y))
                            #print("len_right_node.y : {}".format(len(right_node.y)))
                            #print("")
                            right_node.impurity = self.calc_impurity(left_node.y) # ジニ不純度を追加
                            value, count = np.unique(right_node.y, return_counts = True) # ラベルを追加
                            right_node.label = value[np.argmax(count)]
                            #print("right_node : {}".format(right_node))
                            
                            # インスタンス変数に子ノードを格納する
                            self.right_node = right_node
                            
                            # 子ノードを分割する
                            self.right_node.split()
        
        elif self.max_depth is None:
            if self.impurity > 0:
                
                # 情報利得の最大値を初期化
                max_info_gain = 0

                # 情報利得の最大値を探索
                for row in range(self.X.shape[0]):
                    for column in range(self.X.shape[1]):

                        left_index = np.where(self.X[:, column] >= self.X[row][column])[0] # 特徴量が閾値以上のサンプルのインデックス         
                        right_index = np.where(self.X[:, column] < self.X[row][column])[0] # 特徴量が閾値未満のサンプルのインデックス

                        ig  = self.information_gain(self.y, left_index, right_index) # 情報利得

                        # 情報利得が最大となる分割を求める
                        if ig > max_info_gain:
                            max_info_gain = ig # 情報利得の最大値を更新する

                            # 親ノードのインスタンス変数を更新する                        
                            self.feature = column # ノードの分割に用いる特徴量
                            self.threshold = self.X[row][column] # ノードの分割に用いる特徴量の閾値
                            self.left_index = left_index # 左側の子ノードのインデックス
                            self.right_index = right_index # 右側の子ノードのインデックス
                            
                            # 子ノードの生成（左側）
                            left_node = Node()
                            #print("left_node : {}".format(left_node)
                            left_node.depth = self.depth + 1
                            left_node.max_depth = self.max_depth
                            left_node.X = self.X[self.left_index]
                            left_node.y = self.y[self.left_index]
                            #print("left_node.y : {}".format(left_node.y))
                            #print("len_left_node.y : {}".format(len(left_node.y)))
                            #print("")
                            left_node.impurity = self.calc_impurity(left_node.y)
                            value, count = np.unique(left_node.y, return_counts = True)
                            left_node.label = value[np.argmax(count)]
                            #print("left_node : {}".format(left_node))
                            
                            # 親ノードのインスタンス変数に格納
                            self.left_node = left_node
                            
                            # 子ノードを分割する
                            self.left_node.split()
                            
                            # 子ノードの生成（右側）
                            right_node = Node()
                            #print("right_node : {}".format(right_node))

                            # インスタンス変数を追加
                            right_node.depth = self.depth + 1 # 子ノードの深さ(depth)を親ノードの1階層したとする
                            right_node.max_depth = self.max_depth # 子ノードに深さの最大値を設定
                            right_node.X = self.X[self.right_index] # 説明変数（特徴量）を追加
                            right_node.y = self.y[self.right_index] # 目的変数を追加
                            #print("right_node.y : {}".format(right_node.y))
                            #print("len_right_node.y : {}".format(len(right_node.y)))
                            #print("")
                            right_node.impurity = self.calc_impurity(left_node.y) # ジニ不純度を追加
                            value, count = np.unique(right_node.y, return_counts = True) # ラベルを追加
                            right_node.label = value[np.argmax(count)]
                            #print("right_node : {}".format(right_node))
                            
                            # インスタンス変数に子ノードを格納する
                            self.right_node = right_node
                            
                            # 子ノードを分割する
                            self.right_node.split()
        
        
    def predict(self, X):
        """
        ノードのインスタンス変数に格納された特徴量（X）に基づき予測値を返す
        
        Parameters
        ----------
        X : 次の形のndarray, shape(n_features)
            特徴量        
        
        """
        
        if self.depth == self.max_depth:
            return self.label
        elif (type(self.left_node) == int) or (type(self.right_node) == int):
            return self.label
        elif (self.left_node.label == None) or (self.right_node.label == None): # 葉ノードであれば、ラベルを返す
            return self.label
        else:
            if X[self.feature] >= self.threshold:
                return self.left_node.predict(X)
            elif X[self.feature] < self.threshold:
                return self.right_node.predict(X)
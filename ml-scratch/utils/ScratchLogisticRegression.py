import numpy as np
import math
import matplotlib.pyplot as plt

class ScratchLogisticRegression():
    """
    ロジスティック回帰のスクラッチ実装
    
    Parameters
    ----------
    num_iter : int
        イテレーション回数
        
    lr : float
        学習率
        
    lmd : float
        正則化パラメータ
    
    no_bias : bool
        バイアス項を入れない場合はTrue
    
    verbose : bool
        学習過程を出力する場合はTrue
    
    
    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape(n_features,)
        パラメータ

    self.train_loss : 次の形のndarray, shape(self.iter,)
        学習用データに対する損失の記録
        
    self.val_loss : 次の形のndarray, shape(self.iter,)
        検証用データに対する損失の記録
        
    """
    
    # コンストラクタ
    def __init__(self, num_iter = 500, lr = 1e-2, lmd = 1, no_bias = True, verbose = False):
        
        # ハイパーパラメータを属性として記録
        self.iter = num_iter#イテレーション回数
        self.lr = lr#学習率
        self.lmd = lmd#正則化パラメータ
        self.no_bias = no_bias#バイアス項  True :　あり、False：なし
        self.verbose = verbose     # 学習過程の表示    True : あり、 False：なし 
        
        # 損失を記録する配列を用意
        self.train_loss = np.zeros(self.iter) # 学習用データに基づき計算した損失を記録
        self.val_loss = np.zeros(self.iter)    # 検証用データに基づき計算した損失を記録
        
        # パラメータベクトル
        self.coef = 1
        
        # 正解ラベル
        self.y = 1

    def fit(self, X, y, X_val = None, y_val = None):
        """
        シグモイド回帰を学習する
        
        Parameters
        ----------
        X : 次の形のndarray, shape (m_samples, n_features)
            学習用データの特徴量
            
        y : 次の形のndarray, shape (m_samples, )
            学習用データの正解値
        
        X_val : 次の形のndarray, shape (m_samples, n_features)
            検証用データの特徴量
            
        y_val : 次の形のndarray, shape (m_samples, )
        
        
        """
        # Numpy配列に変換(pandasデータにも対応するため)
        
        # 学習用データ
        X = np.array(X) # 説明変数
        y = np.array(y) # 目的変数
        
        # 検証用データ
        X_val = np.array(X_val) # 説明変数
        y_val = np.array(y_val) # 目的変数
        
        
        # 1次元配列の場合、軸を追加する
        if X.ndim == 1:
            X[:, np.newaxis] # 説明変数
        if y.ndim == 1:
            y[:, np.newaxis] # 目的変数
        if X_val.ndim == 1:
            X_val[:, np.newaxis] # 説明変数
        if y_val.ndim == 1:
            y_val[:, np.newaxis] # 目的変数
        

        # vervoseをTrueにした場合は学習過程を出力
        if self.verbose:
            print(self.train_loss)
            
        # バイアス項を入れる場合（no_bias = True）の場合、バイアス項を水平方向に連結する
        if self.no_bias == False:
            bias_term = np.ones(len(X)).reshape(-1, 1)  #  (m, 1)行列
            X = np.concatenate([bias_term, X], axis = 1) #  (m+1, n)行列
            
            if (X_val is not None) and (y_val is not None):
                bias_term = np.ones(len(X_val)).reshape(-1, 1)  #  (m, 1)行列
                X_val = np.concatenate([bias_term, X_val], axis = 1) #  (m+1, n)行列

        # パラメータベクトルをランダム関数で初期化
        np.random.seed(seed=0)
        self.coef = np.random.rand(X.shape[1]).reshape(1,-1)
        
       
        # 正解値のラベルをインスタンス変数に保存しておく（予測値の正解ラベルに使うため）
        self.y = y

                
        # 正解ラベルの要素を(0,1)に変換する
        self.y_train = y.copy()
      
        for i in range(len(self.y_train)):

            if self.y_train[i] == min(y):
                self.y_train[i] = 0
            elif self.y_train[i] == max(y):
                self.y_train[i] = 1
                
        # 検証用データも同様に        
        if (X_val is not None) and (y_val is not None):
            self.y_val = y_val.copy()
            
            for i in range(len(self.y_val)):

                if self.y_val[i] == min(y_val):
                    self.y_val[i] = 0
                elif self.y_val[i] == max(y_val):
                    self.y_val[i] = 1
        

        # 所定の試行回数だけ学習を繰り返す
        for i in range(self.iter):
            
            self.gradient_descent(X, self.y_train) # 最急降下法（パラメータ更新）
            
            train_loss = self.cross_entropy_loss(X, self.y_train) # 損失を計算
            
            self.train_loss[i] = train_loss # 配列に格納
            
            # vervoseをTrueにした場合は学習過程を出力
            if self.verbose:
                print("Train Loss in {0}th iteration : {1}".format(i, round(train_loss)))
                
            
            if (X_val is not None) and (y_val is not None):
                
                val_loss = self.cross_entropy_loss(X_val, y_val)              
                self.val_loss[i] = val_loss
                
                if self.verbose:
                    print("Valid Loss in {0}th iteration : {1}".format(i, round(val_loss)))
                    print("")
    
    
    def predict_prob(self, X):
        """
        予測値に対する確率を算出する
        
        Parameter
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            特徴量
        
        
        Return
        ----------
        y_pred_prob : 次の形のndarray, shape (m_samples,)
            予測値の正解率
        
        """
        # バイアス項を入れる場合（no_bias = True）の場合、バイアス項を水平方向に連結する
        if self.no_bias == False:
            bias_term = np.ones(len(X)).reshape(-1, 1)  #  (m, 1)行列
            X = np.concatenate([bias_term, X], axis = 1) #  (m+1, n)行列
        
        y_pred_prob = self._sigmoid_hypothesis(X)
        
        return y_pred_prob
        
        
        
    def predict(self, X, threshold = 0.5):
        """
        分類ラベルの予測(0 or 1)を返す
        
        Parameter
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            特徴量
            
        threshold : float
            閾値
        
        
        Return
        ----------
        y_pred : 次の形のndarray, shape (m_samples, )
            予測値
            
        """
            
        # 予測確率
        y_pred_prob = self.predict_prob(X) # no_bias = Falseの場合、Xのindexはm+1行
        
        # 正解ラベルの予測値をゼロで初期化
        y_pred = np.zeros(len(y_pred_prob)).reshape(-1,1) # (m,1)行列
        
        # y_predの各要素について、パラメータを更新する。
        for i in range(len(y_pred)):
            if y_pred_prob[i] < threshold:
                y_pred[i] = min(self.y) # 閾値を下回る場合、negative（正解ラベルのうち大きい値）
            else:
                y_pred[i] = max(self.y) # 閾値を上回る場合、positive（正解ラベルのうち小さい値）
                
        return y_pred.astype("int64")
        
        
    def sigmoid(self, z):
        """
        シグモイド関数
        
        Parameters
        ----------
        z : 次の形のndarray, shape(m_samples, 1_features)
            仮定関数
        
        
        Returns
        ----------
        prob : 
            シグモイド関数で算出した確率
            
        """
        
        prob = 1/(1 + np.exp(-z)) # 演算
        
        # sigmoid.reshape(-1, 1) # 出力されたベクトルの次元が不定の場合、reshapeする
        
        
        return prob
    
    

    def _linear_hypothesis(self, X):
        """
        線形の仮定関数

        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習データ


        Returns
        ----------
            次の形のndarray, shape(m_samples, 1)
            線形の仮定関数による推定結果

        """
        # 仮定関数
        line_hypo = np.dot(X, self.coef.T)

        return line_hypo
    
   

    def _sigmoid_hypothesis(self, X):
        """
        シグモイド仮定関数

        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習データ


        Returns
        ----------
            次の形のndarray, shape(m_samples, 1)
            シグモイド形の仮定関数による推定結果

        """

        z = self._linear_hypothesis(X) # 線形和

        sig_hypo = self.sigmoid(z) # 予測確率

        return sig_hypo
    
    

    def regularization_term(self, X):
        """
        正則化項

        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習用データ

        coef : 次の形のndarray, shape(1_sample, n_features)
            パラメータベクトル

        lmd : int
            正則化パラメータ

        Returns
        ----------
        reg_term : float64
            正則化項

        """
        
        reg_term = self.lmd / len(X) * np.sum(self.coef ** 2)  # 正則化項

        return reg_term    
    
    
    def cross_entropy_loss(self, X, y):
        """
        クロスエントロピー損失を求める
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習データ

        y : 次の形のndarray, shape(m_samples, 1_feature)
            正解値

        Returns
        ----------
        loss : float64
            損失

        """

        # 正則化項
        reg_term = self.regularization_term(X)
        #print("regularization_term.shape:{}".format(reg_term.shape))

        # シグモイド仮定関数
        sig_hypo = self._sigmoid_hypothesis(X) # 確率

        # 目的関数の第１項
        first_term = - y * np.log(sig_hypo)
        
        # 目的関数の第２項
        second_term = - (1 - y) * np.log(1- sig_hypo)

        # 目的関数の計算結果
        loss = np.sum(first_term + second_term + reg_term)

        return loss
    
    
    
    def gradient_descent(self, X, y):
        """
        最急降下法（パラメータの更新） 
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_sample, n_features)
            学習データ

        y ： 次の形のndarray, shape(m_sample, 1_feature)
            正解値


        Returns
        ----------
        coef : 次の形のndarray, shape(1_sample, n_features)
            パラメータベクトル

        """

        # シグモイド仮定関数
        sig_hypo = self._sigmoid_hypothesis(X)

        # 第1項
        grad_first_term =  np.dot(sig_hypo.T,  X) / len(X)
        
        # 第２項
        temp_coef = self.coef #演算用に用いるパラメータを作成
        
        if self.no_bias == False:
            temp_coef[0][0] = 0 # バイアスありの場合、バイアス項に対するパラメータ（θ）をゼロにする
        
        #print("temp_coef:{}".format(temp_coef))
        grad_second_term = self.lmd / len(X) * temp_coef
        grad_second_term = grad_second_term.reshape(1,-1)
        
        # パラメータ更新
        self.coef -= self.lr * (grad_first_term + grad_second_term)        
        
        
    def show_learning_curve(self,):
        """
        学習過程をグラフに描画
        
        """
        
        if self.val_loss.all() == 0:
            None
        else:
            plt.plot(self.val_loss, label="val_loss") # 検証用データによる交差エントロピー損失
        
        plt.plot(self.train_loss, label = "train_loss") # 学習用データによる交差エントロピー損失        
        plt.xlabel("iteration") 
        plt.ylabel("cross_entropy_loss")
        plt.title("Learning_Curve")
        plt.legend()
        plt.show()
        
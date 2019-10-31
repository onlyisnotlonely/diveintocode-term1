# 数値演算
import numpy as np

# クラス
class LinearRegression():
    
    """
    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録
      
    """
    
    # 初期値を設定
    def __init__(self, num_iter = 1000, lr = 1e-2, no_bias = True, verbose = False):
        
        # ハイパーパラメータを属性として記録
        self._iter = num_iter
        self.lr = lr
        self.no_bias = no_bias #バイアス項を入れる場合はFalse
        self.verbose = verbose
        
        # パラメータ
        self.coef_ = 1
                
        # 損失を記録する配列を用意
        self.loss = np.zeros(self._iter)
        self.val_loss = np.zeros(self._iter)
        
        # 検証用データに対する予測精度（決定係数）を記録する配列を用意
        self.coef_determination = np.zeros(self._iter)
        
            
    def fit(self, X, y, X_val = None, y_val = None):
        
        """
        線形回帰を学習する。
        検証用データが入力された場合はそれに対する損失と予測精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        
        if (X_val is not None ) and (y_val is not None):
            print("Validation data was completely set as arguments.")
        
        
        if self.verbose:
            #verboseをTrueにした際は学習過程（損失、予測精度）を出力
            print("Mean_Squared_Error_for_train_data:{}".format(self.loss))
            if (X_val is not None ) and (y_val is not None):
                print("Mean_Squared_Error_for_val_data:{}".format(self.val_loss)) # 損失
                #print("Prediction_accuracy.{}".format(self.coef_determination)) # 予測精度（決定係数）
                
        y = y.reshape(-1,1)
        
        # バイアス項を入れない場合はTrue, バイアス項を入れる場合はFalse
        if self.no_bias == False:
            bias_array = np.ones(len(X)).reshape(-1, 1)
            X = np.concatenate([bias_array, X], axis = 1)
        
        #print("X.shape:{}".format(X.shape))
        
        # 特徴量を持つ回帰係数（１で初期化）
        self.coef_ = np.ones(X.shape[1])[np.newaxis, :]
        
        # 最急降下法を所定の回数（＝self._iter回）繰り返す
        for i in range(self._iter):
            
            # 仮定関数
            y_pred = self._linear_hypothesis(X)
            #print("y_pred:{}".format(y_pred)) # テスト用に出力
            
            # 平均二乗誤差、予測値と正解の差異を求める
            mse, error = self.MSE(y_pred, y)
            
            # 所与の正解ラベル（y）に対してMSEを求めて、格納する
            self.loss[i] = mse
            
            # 新しいthetaを求める
            self._gradient_descent(X, error)
            
            # 決定係数を格納
            self.coef_determination[i] = self.calc_coef_determination(X, y)
            
            # 検証データを引数に入れた時、以下を実行する
            if (X_val is not None ) and (y_val is not None):
                
                y_val = y_val.reshape(-1,1)
                
                # 検証データに対する予測値を求める 
                #print("X_val.shape:{}".format(X_val.shape))
                y_val_pred = self._linear_hypothesis(X_val)
                #print("y_val_pred:{}".format(y_val_pred))
                
                # 平均二乗誤差、誤差を求める
                #print("y_val_pred.shape : {}".format(y_val_pred.shape))　　# テスト用に出力
                #print("y_val.shape : {}".format(y_val.shape))　# テスト用に出力
                
                mse_val, error_val = self.MSE(y_val_pred, y_val)
                #print("mse_val:{}".format(mse_val))
                
                # 平均二乗誤差を格納する
                self.val_loss[i] = mse_val
            
    def predict(self, X):
        """
        
        線形回帰を使い推定する。
        
        
        Parameters
        ----------
        X : 次の形のndarray, shape(n_samples, n_features)
            サンプル
            
        Returns
        ----------
            次の形のndarray, shape(n_samples, 1)
            線形回帰による推定結果
        
        """
        
       # バイアス項を入れない場合はTrue, バイアス項を入れる場合はFalse
        if self.no_bias == False:
            bias_array = np.ones(len(X)).reshape(-1, 1)
            X = np.concatenate([bias_array, X], axis = 1)
        
        y_pred = np.dot(X, self.coef_.T)
        
        return y_pred
    
     # 線形の仮定関数を定義する
    def _linear_hypothesis(self, X):
        
        """
        線形の仮定関数を出力する
        
        Parameters
        ----------
        X ： 次の形のndarray, shape(n_samples, n_features) (2 × 4)
            学習データ
            
        
        Returns
        ----------
        self.y_pred
            次の形のndarray, shape(n_sample, 1)
            線形の仮定関数による推計結果(インスタンス変数) 
        
        """
        
        #print("X.shape:{}".format(X.shape))　# テスト用に出力
        #print("self.coef_.shape".format(self.coef_.shape))　# テスト用に出力
        
        # 仮定関数による推計結果
        y_pred = np.dot(X, self.coef_.T)
        
        return y_pred
    
    def MSE(self, y_pred, y):
                
        """
        平均二乗誤差（MSE）の計算
        
        Parameters
        ----------
        y : 次の形のndarray, shape (n_samples, 1_feature)
            正解値
        
        Returns
        ----------
        mse : numpy.float
            平均二乗誤差
        
        """
        
        # 正解と予測の差異
        #print("y_pred.shape:{}".format(y_pred.shape))　# テスト用に出力
        #print("y.shape:{}".format(y.shape))　# テスト用に出力
        
        error = y_pred - y
        
        # 平均二乗誤差
        mse = np.sum(error ** 2) * (1/2) / len(y)
        
        return mse, error
    
    def _gradient_descent(self, X, error):
        
        """
        最急降下法
        
        Parameters
        ----------
        X ： 次の形のndarray, shape(n_samples, n_features)
            学習データ
        
        error: 
            誤差（＝仮定関数 - 正解）

        """
        #print("X:{}".format(X.shape))　# テスト用に出力
        #print("error:{}".format(error.shape))　# テスト用に出力
        
        self.coef_ -= self.lr / len(X) * np.dot(error.T, X)
        #print("self.coef_ : {}".format(self.coef_))　# テスト用に出力
        
    def calc_coef_determination(self, X, y):
        """
        決定係数を求める
        
        Parameters
        ----------
        X : 次の形のndarray, shape(n_samples, n_features)
            学習データ
            
        y_val : 次の形のndarray, shape(n_sameple, 1_feature)
            正解値
        
        Returns
        ----------
        coef_determination : int
            決定係数
        
        """
        
        # 実測値の平均
        y_mean = sum(y) / len(y)
        #print("y_val_mean:{}".format(y_val_mean)) # テスト用に出力
        
        # 検証データ（X_val）に対する予測値
        y_pred = self._linear_hypothesis(X)
        #print("y_val_pred : {}".format(y_val_pred))　# テスト用に出力
        
        # 回帰変動(実測値の平均値と推計値の誤差二乗和)
        SR = sum((y_pred - y_mean) ** 2)
        #print("SR: {}".format(SR))#テスト用に出力
        
        # 全体変動（実測値の平均値と実測値の誤差二乗和）
        ST = sum((y - y_mean) ** 2)
        #print("ST:{}".format(ST))#テスト用に出力
        
        coef_determination = SR / ST
        
        return coef_determination
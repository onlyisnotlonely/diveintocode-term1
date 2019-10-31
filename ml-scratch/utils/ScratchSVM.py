import numpy as np # 演算
import matplotlib.pyplot as plt # 可視化

class ScratchSVC():
    """
    SVMのスクラッチ実装
    
    Parameters
    ----------
    num_iter : int
        イテレーション回数
        
    lr : float
        学習率
        
    threshold : float
        閾値
        
    verbose : bool
        学習過程を出力する場合はTrue
        
    
    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape(n_features,)
        パラメータ
    

    """
    
    def __init__(self, num_iter=300, lr=1e-3, threshold=1e-5, verbose=False, kernel="linear", gamma = 1, coef0=0, degree = 1):
        
        # ハイパーパラメータとして属性を記録
        self.num_iter = num_iter # イテレーション回数
        self.lr = lr #学習率
        self.threshold = threshold # 閾値
        self.verbose = verbose     # 学習過程の表示    True : あり、 False：なし 
        
        # パラメータベクトル
        self.coef = 1
        self.cons = 1 # 分類境界線の定数項（theta_0）

        # ラグランジュ
        self.lmd = 1 # ラグランジュ乗数
        self.lag = []  # ラグランジュアン
        
        # サポートベクター
        self.sv_index = 1 # インデックス
        self.sv_count = 1 # 個数
        
        # カーネル
        self.kernel = kernel # "linear" or "polynomial"
        self.kernel_calc = 1
        
        # 多項式カーネルのパラメーター
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
    
    
    def fit(self, X, y, X_val = None, y_val=None):
        """    
        学習

        Parameter
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習用データの特徴量

        y : 次の形のndarray, shape(m_samples,)
            学習用データの正解値

        X_val : 次の形のndarray, shape(m_samples, n_features)
            検証用データの特徴量

        y_val : 次の形のndarray, shape(m_samples,)
            検証用データの正解値

        """
        
        # 最急降下法
        lmd = self.gradient_descent(X, y)
        
        # サポートベクターを抽出する
        sv = self._find_support_vector(X)
        
        # 重みの計算に用いるサンプルを抽出する
        X = X[self.sv_index] # 特徴量：(m, n) 行列
        y = y[self.sv_index].reshape(-1, 1) # 正解値：(m, 1)行列
        lmd = self.lmd[self.sv_index].reshape(-1, 1) # ラグランジュ乗数：(m, 1行列)
        
        # 特徴量と同じ個数の要素を持つ零配列を生成
        self.coef = np.zeros(X.shape[1]).reshape(1, -1)
        
        # 重み（n_features）を更新
        for i in range(self.sv_count):
            self.coef += lmd[i] * y[i] * X[i] 
        
        # θ0を計算
        self.cons = np.sum(y - np.dot(X, self.coef.T))
    
    
    def predict(self, X):
        """
        分類を予測する

        Parameter
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            テストデータ

        coef : 次の形のndarray, shape(n_features)
            重み

        Returns
        ----------
        y_pred : 次の形のndarray, shape(m_samples)
            分類の予測値

        """
        # 形状を確認
        #print("X.shape : {}".format(X.shape))
        #print("coef.shape : {}".format(coef.shape))

        # 予測値を計算
        y_pred = np.dot(X, self.coef.T) + self.cons
        #print("y_pred : {}".format(y_pred))

        # ラベルを更新
        y_pred[y_pred < 0] = -1
        y_pred[y_pred > 0] = 1

        return y_pred.astype(np.int64)
    
    

    def linear_kernel(self, xi, xj):
        """
        線形カーネル

        Parameters
        ----------
        xi : 次の形のndarray, shape(n_features,)
            特徴量のサンプル

        xj : 次の形のndarray, shape(n_features,)
            特徴量のサンプル 

        Returns
        ----------
        kernel : float
            カーネル

        """
        
        kernel = float(np.dot(xj.reshape(1, -1), xi.reshape(1, -1).T))

        return kernel
    
    
    def polynomial_kernel(self, xi, xj):
        """
        多項式カーネル

        Parameters
        ----------
        xi : 次の形のndarray, shape(n_features,)
            特徴量のサンプル

        xj : 次の形のndarray, shape(n_features,)
            特徴量のサンプル 

        Returns
        ----------
        kernel : float
            カーネル

        """
        
        kernel = float(self.gamma * (np.dot(xj.reshape(1, -1), xi.reshape(1, -1).T) + self.coef0)^self.degree)
        
        return kernel

    
    
    def gradient_descent(self, X, y):
        """
        最急降下法（ラグランジュ乗数の更新）

        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習データ

        y : 次の形のndarray, shape(m_samples,)
            正解値


        Returns
        ----------
        lmd : 次の形のndarray, shape(m_samples,)
            ラグランジュ乗数

        """
        
        # カーネルを選択する
        self.kernel_calc = self.linear_kernel
        
        if self.kernel == "polynomial":
            self.kernel_calc = self.polynomial_kernel
        
        # ラグランジュ乗数を更新する
        self.lmd = np.ones(len(X)).reshape(-1, 1) # 初期化
        
        temp_lmd = np.ones(len(X)).reshape(-1, 1) # 更新後のラグランジュ乗数（一時的に格納）

        for iter_num in range(self.num_iter):
            for i in range(len(X)): 
                result = 0 # ラグランジュ乗数の変化を累積する変数
                
                for j in range(len(X)):
                    result += self.lmd[j] * y[i] * y[j] * self.kernel_calc(X[i], X[j])
                    #result += self.lmd[j] * y[i] * y[j] * self.linear_kernel(X[i], X[j])
   
                # 一時的に結果を保存しておく
                temp_lmd[i] = self.lmd[i] + self.lr * (1- float(result))
            
            # ラムダを更新
            for i in range(len(self.lmd)):
                self.lmd[i] = temp_lmd[i]

            # 制約条件（更新毎にλ>0を満たす必要がある）
            self.lmd[self.lmd < 0] = 0
            
            # ラグランジュアンを計算し、リストに格納する
            lag = self._lagrangian(X, y)
            self.lag.append(lag)    
            
            # 学習過程を表示
            if self.verbose == True:
                print(lag)
            
    def _find_support_vector(self, X):
        """
        データセットからサポートベクターを抽出する

        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習用データ

        lmd : 次の形のndarray, shape(m_samples)
            ラグランジュ乗数（最急降下法による出力）


        Returns
        ----------
        sv_index : 次の形のndarray, shape(m_samples,)
            サポートベクターのインデックス

        sv_count : int
            サポートベクターの個数

        sv : 次の形のndarray, shape(m_samples, n_features)
            サポートベクター

        """
        # サポートベクターのインデックス
        self.sv_index = np.where(self.lmd > 0)[0]

        # サポートベクターの個数
        self.sv_count = len(self.sv_index)

        # サポートベクター
        sv = X[self.sv_index]

        return sv
    
    
    def scatter_plot_sv(self, X, y):
        """
        学習用データの散布図の中でサポートベクターをハイライトする

        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習用データ

        y : 次の形のndarray, shape(m_samples,)
            正解値

        sv : 次の形のndarray, shape(m_samples, n_features)
            サポートベクター

        """

        plt.scatter(X[:, 0], X[:,1], c = y) # 学習用データの散布図
        plt.scatter(X[self.sv_index][:, 0], X[self.sv_index][:,1], c = "r", label = "support vector") # サポートベクターの散布図
        plt.xlabel("factor_1")
        plt.ylabel("factor_2")
        plt.title("Where are the support vectors??", color = "r")
        plt.legend()
        plt.show()
        
     
    def _lagrangian(self, X, y):
        """
        ラグランジュアン

        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習データ

        y : 次の形のndarray, shape(m_samples,)
            正解値

        lmd : 次の形のndarray, shape(m_samples,)
            ラグランジュ乗数

        """
       
       # カーネルを選択する
        self.kernel_calc = self.linear_kernel
        
        if self.kernel == "polynomial":
            self.kernel_calc = self.polynomial_kernel
        

        # 第一項
        lag_first_term = float(np.sum(self.lmd))

        # 第二項
        lag_second_term = 0 # 初期化（for文で足していく）
        for i in range(len(X)):
            for j in range(len(X)):
                lag_second_term += float(self.lmd[i] * self.lmd[j] * y[i] * y[j] * self.kernel_calc(X[i], X[j]))
                #lag_second_term += float(self.lmd[i] * self.lmd[j] * y[i] * y[j] * self.linear_kernel(X[i], X[j]))

        # ラグランジュアン
        lag = lag_first_term - lag_second_term / 2
        
        return lag
    
    def show_learning_curve(self,):
        """
        学習曲線を描画する
        
        """
        
        plt.plot(self.lag)
        plt.xlabel("iter")
        plt.ylabel("lagrangian")
        plt.title("Learning Curve")
        plt.show()
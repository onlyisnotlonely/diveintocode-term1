import numpy as np
import random

class ScratchKMeans():
    """
    KMeansのスクラッチ実装
    
    Parameters
    ----------
    n_clusters : int
        クラスタ数
    
    max_iter : int
        イテレーション
        
    tol : float
        中心点と重心間の距離の誤差（学習の停止条件）
        
    random_state : int
        中心点の初期値をランダムに生成する際のシード値
    
    Attributes
    ----------
    self.loss : float
        クラスタ内誤差平方和
    
    """
    
    def __init__(self, n_clusters = 2, max_iter = 20, tol = 1e-8, random_state = 1):
        
        self.n_clusters = n_clusters # クラスタ数
        self.max_iter = max_iter # 学習回数の最大値
        self.tol = tol # 誤差の許容度
        self.seed = random_state # シード値
        
        self.centers = [] # 中心点の集合(n_clusters_samples, m_features)
        self.g = [] # 重心の集合
        self.dist = [] # 中心点と重心間の距離
        
        self.label = [] # 各サンプルが属するクラスタのラベルの配列
        self.label_matrix = None # ラベルのマトリクス（データ点Xnがクラスタkに所属していたら1、そうでなければ0）
        self.cluster_matrix = None # 各サンプルのクラスタ情報
        self.cluster_labels = []
        self.silhouette_vals = 1 # シルエット係数
        
        self.loss = None # 誤差平方和
        
        self.X = None # データセット（特徴量）
    
    
    def fit(self, X):
        """
        学習：データセットの重心との誤差が許容範囲に納まる中心点を求める
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習用データ（特徴量）
            
        Returns
        ----------
        center_point : list
            中心点
        
        """
        # インスタンス変数にデータセットを格納
        self.X = X
        self.cluster_labels = [i for i in range(self.n_clusters)]
        
        sse_list = [] # 各イテレーションの結果、SSEの最小値を格納する
        centers_list = [] # 中心点のリスト
        label_list = [] # クラスタのラベル
        
        random.seed(self.seed) # 乱数生成のシード値を固定
        sample_index = [i for i in range(len(self.X))] # サンプルのインデックス
        
        # クラスタマトリクスを初期化
        cluster = []
        cluster_matrix = np.zeros((X.shape[0], self.n_clusters))
        cluster.append(cluster_matrix)
        
        # 重心を初期化
        self.g = [0 for i in range(self.n_clusters)]
        
        # 中心点の初期値を生成
        random.seed(self.seed)
        init_index = random.sample(sample_index, self.n_clusters)
        self.centers = X[init_index]
        
        # SSEが最小となる中心点を求める
        for i in range(self.max_iter):
              
            # 距離行列を計算する
            dist_matrix = self.create_dist_matrix(X, self.centers)
            
            # 各データに対して、中心点からの距離が最も小さいクラスタの番号を与える
            label = self._create_label(X, dist_matrix)
            label_list.append(label)

            # 各クラスタの重心を求める
            self.g = self._calc_gravity(label)
            
            # クラスタマトリクスを計算する
            cluster_matrix = self.create_cluster_matrix(self.X, label)

            # クラスタ内誤差平方和を求める
            sse = np.sum(dist_matrix * cluster_matrix)

            # 中心点と重心の間の距離を求める
            dist = self._calc_dist(self.centers, self.g)

            # 中心点と重心間の距離の最大値が許容範囲内に収まった場合、学習を終える
            if max(dist) < self.tol:
                break
            else:
                 # 算出した重心を次の中心点とする
                self.centers = self.g
                centers_list.append(self.centers)
                # SSEをリストに格納する
                sse_list.append(sse)
            
            # データ点 Xnのクラスタへの割り当てが変化しなくなる場合、学習を止める
            if (np.array(cluster[-1] == cluster_matrix)).all():
                break
            else:
                cluster.append(cluster_matrix)

        # iterationが完了したら、sseが最小となる中心点、ラベル、sseの最小値をインスタンス変数に格納する
        
        min_sse_index = np.argmin(np.array(sse_list)) # SSEの最小値のインデックスを取得する        
        self.centers = np.array(centers_list[min_sse_index]) # SSEが最小となるケースの中心点をインスタンスに格納
        self.label = np.array(label_list[min_sse_index]) # SSEが最小となるケースで、各サンプルが所属するクラスタを格納
        self.loss = sse_list[min_sse_index] # SSEの最小値を格納
        
        value, count = np.unique(self.label, return_counts = True)
        
        # 凝集度を計算する
        aggregation_list = self._calc_aggregation()
        
        # 乖離度を計算する
        divergence_list = self.calc_divergence()
        
        # 各サンプルのシルエット係数を格納する
        silhouette_vals = [] 
        
        for i in range(len(self.X)):
            silhouette_val = (divergence_list[i] - aggregation_list[i]) / max(divergence_list[i], aggregation_list[i])
            silhouette_vals.append(silhouette_val)
        
        self.silhouette_vals = np.array(silhouette_vals)
        
    
    def predict(self, X):
        """
        与えたデータセットが属するクラスタを求める
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            テストデータ（特徴量）
        
        Returns
        ----------
        y_pred : 次の形のndarray, shape(m_samples, n_features)
            予測値（データセットが属するクラスタ）
        
        """
        
        # 距離行列を作成する
        dist_matrix = self.create_dist_matrix(X, self.centers)
        
        X_pred = self._create_label(X, dist_matrix)
 
        return X_pred
       
    
    def create_dist_matrix(self, X, centers):
        """
        各サンプルに対して、各中心点からの距離を格納した配列を生成する
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習用データ（特徴量）
        
        centers : 次の形のndarray, shape(m_sampls, n_features)
            中心点の集合
        
        Returns
        ----------
        dist_matrix : 次の形のndarray, shape(m_samples, n_features)
            距離行列
            
        """
        
        dist_matrix = np.zeros((X.shape[0], 1))        
        
        # 各中心点とサンプル間の距離を算出し、距離を格納した行列（距離行列）を生成する
        for center in centers:
            dist = np.linalg.norm(X - center, axis = 1).reshape(-1,1) # 各サンプルと重心の距離の配列（m_samples,）
            dist_matrix = np.hstack([dist_matrix, dist])
        
         # 1列目（零配列）を削除
        dist_matrix = np.delete(dist_matrix, 0, axis=1)
        
        return dist_matrix
    
    
    def _create_label(self, X, dist_matrix):
        """
        各サンプルに対して、中心点からの距離が最も小さいクラスタのラベルを与える
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            学習用データ（特徴量）
        
        dist_matrix : 次の形のndarray, shape(m_samples, n_features)
            中心点とサンプルの距離を格納した行列
        
        """
        
        # 各サンプルについて、距離が最小となる列番号（＝クラスタ番号）を返し、ラベルとする
        label = np.argmin(dist_matrix, axis = 1)
        
        return label
    
    
    def create_cluster_matrix(self, X, label):
        """
        各サンプルのクラスタ所属情報を格納した行列を生成する
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            データセット
            
        
        Returns
        ----------
        cluster_matrix : 次の形のndarray, shape(m_samples, n_clusters)
            クラスタマトリクス（cluster_iに属している場合:1, 属していない場合：0)
        
        """        
        
        # 所属マトリクス（各サンプルの所属するクラスタを0,1で表現）
        cluster_matrix = np.zeros(X.shape[0] * self.n_clusters).reshape(X.shape[0], self.n_clusters)
        
        for i, column in enumerate(label):
            cluster_matrix[i][column] = 1

        return cluster_matrix

        
    
    def _calc_gravity(self, label):
        """
        各クラスタの重心を求める
        
        Parameters
        ----------
        label : 次の形のndarray, shape(m_samples,)
            各サンプルが属するクラスタ
        
        Return
        ----------
        g : list
            重心の集合
        
        """
        # 各クラスタの重心を格納するリスト
        g = []
        
        for i in range(self.n_clusters):
            index = np.where(label==i)[0] # i番目のクラスタに属するサンプルのインデックスを取得する
            gravity = np.mean(self.X[index], axis = 0) # i番目のクラスタの重心を求める
            g.append(gravity)
        
        return g
    
    
    
    def _calc_dist(self, centers, g):
        """
        中心点と重心の間の距離を求める
        
        Parameters
        ----------
        centers : 次の形のndarray, shape(n_clusters, m_features,)
            中心点の集合
        
        g : list
            重心
        
        Returns
        ----------
        dist : 
            中心点と重心の間の距離
        """
        # 中心点と重心の間の距離を求める
        dist = np.linalg.norm(np.array(centers) - np.array(g), axis = 1)
        
        return dist
    
    
    
    def calc_loss(self, dist_matrix, cluster_matrix):
        """
        クラスタ内誤差平方和（SSE, Sum of Squared Errors）を求める
        
        Parameters
        ---------
        X : 次の形のndarray, shape(m_samples, n_features)
            特徴量
        
        dist_matrix : 次の形のndarray, shape(m_samples, n_features)
            距離行列
        
        Returns
        ----------
        loss  : float
            クラスタ内誤差平方和
        
        """                
        # クラスタ内誤差平方和を求める
        loss = np.sum(dist_matrix * cluster_matrix)
        
        return loss
    
    
    def _calc_aggregation(self, ):
        """
        凝集度（同じクラスタ内の他のデータ点との距離の平均値）を計算する
        
        Returns
        ----------
        aggregation_list : list
            凝集度
        
        """
        
        # クラスタ数が2以上の時に凝集度を計算する
        if self.n_clusters == 1:
            aggregation_list = [1 for i in range(len(self.X))] # 
        elif self.n_clusters > 1:
            aggregation_list = []

            for i in range(len(self.X)):
                label = self.label[i] # i番目のサンプルが属するクラスタ
                
                # 同じクラスタに所属するサンプルのインデックス(i番目のサンプル自身を含むので後で取り除く)
                same_cluster_sample_index = np.where(self.label == label)[0]
                
                # サンプルi以外に同じクラスタに属するサンプルがある場合のみ凝集度を計算する
                if len(same_cluster_sample_index) == 1:
                    aggregation_list.append(0)
                elif len(same_cluster_sample_index) > 1:    
                    same_cluster_sample_index = same_cluster_sample_index[same_cluster_sample_index != i]
                    same_cluster_sample = self.X[same_cluster_sample_index]
                    
                    # サンプルi、及び同じクラスタに所属する他サンプル間の距離の平均値
                    aggregation = np.sum(np.linalg.norm(same_cluster_sample - self.X[i], axis = 1)) / len(same_cluster_sample) 
                    aggregation_list.append(aggregation)
                    
        return aggregation_list
    
    
    
    def calc_divergence(self, ):
        """
        乖離度（最も近い他のクラスタ内の全てのデータ点との距離の平均値）を計算する
        
        Parameters
        ----------
        X : 次の形のndarray, shape(m_samples, n_features)
            データセット
        
        Returns
        ----------
        divergence : list
            凝集度
        
        """
        # クラスター数が1以上の時に乖離度を計算する
        if self.n_clusters == 1:
            divergence_list = [1 for i in range(len(self.X))]
        elif self.n_clusters > 1:
            # 各サンプルの乖離度を格納するリスト
            divergence_list = []
            
            # クラスタ
            cluster = np.array([i for i in range(self.n_clusters)])

            for i in range(len(self.X)):
                label = self.label[i] # サンプルが属するクラスタ
                other_cluster = cluster[cluster != label] # サンプルが属していないクラスタ                
                centers = self.centers[other_cluster] # サンプルが属していないクラスタの中心点
                
                # 最も近い他のクラスタを選択する
                nearest_cluster = other_cluster[np.argmin(np.linalg.norm(centers - self.X[i], axis = 1))]

                # 選択したクラスタ内の全てのデータ点との距離の平均値を求める
                nearest_cluster_samples = self.X[np.where(self.label == nearest_cluster)[0]]
                
                if len(nearest_cluster_samples) == 0:
                    divergence_list.append(0)
                elif len(nearest_cluster_samples) > 0:
                    divergence = np.mean(np.linalg.norm(nearest_cluster_samples - self.X[i], axis = 1))
                    divergence_list.append(divergence)
        
        return divergence_list
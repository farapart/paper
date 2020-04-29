import pandas as pd
import numpy as np
import random as rd
#定义常量信息
TOTAL_USER_NUMBER = 943
TOTAL_MOVIE_NUMBER = 1682
TOTAL_RATING_NUMBER = 100000
#辅助矩阵的大小定义
TEST_USER_NUMBER = 250
TEST_MOVIE_NUMBER = 250
#矩阵分解常量定义
K = 15
L = 15
#聚类算法迭代数据定义
MIN = 175

#矩阵相乘函数
def Matrix_Matmul(matrixs=[]):
    result = matrixs[0]
    for i in range(1, len(matrixs)):
        result = np.matmul(result, matrixs[i])
    return result


#读用户数据
# users_name = ['user_id','age','gen
# der','occupation','zip']
# users_data = pd.read_csv("ml-100k\ml-100k\\u.user",sep='|', header=None, names=users_name, engine='python')
# print(users_data.head(5))

#读评分数据
ratings_name = ['user_id','movie_id','rating','timestamp']
ratings_data = pd.read_csv("ml-100k\ml-100k\\u.data",sep='\t', header=None, names=ratings_name, engine='python')

#读电影数据
# movie id | movie title | release date | video release date |IMDb URL | unknown | Action | Adventure | Animation |
# Children's | Comedy | Crime | Documentary | Drama | Fantasy |Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |Thriller |
# War | Western |
# movies_name = ['movie_id', 'title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure'
#                , 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror'
#                , 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
# # movies_name = ['movie_id','title','genres']
# movies_data = pd.read_csv("ml-100k\ml-100k\\u.item",sep='|', header = None, names=movies_name, engine='python')

#创建用户评分矩阵,用户id对应行数+1，电影id对应列数+1，评分值放在矩阵中。
ratings_matrix = np.zeros((TOTAL_USER_NUMBER, TOTAL_MOVIE_NUMBER), dtype=int)
for row in ratings_data.itertuples():
    ratings_matrix[getattr(row,'user_id')-1][getattr(row, 'movie_id')-1] = getattr(row, 'rating')

#随机抽取TEST_USER_NUMBER行和TEST_MOVIE_NUMBER列
rows = rd.sample(range(0, TOTAL_USER_NUMBER), TEST_USER_NUMBER)
columns = rd.sample(range(0, TOTAL_MOVIE_NUMBER), TEST_MOVIE_NUMBER)
rows.sort()
columns.sort()

test_rating_matrix = ratings_matrix[rows]
test_rating_matrix = test_rating_matrix[:, columns]
#创建矩阵F，代表用户特征矩阵，大小为TEST_USER_NUMBER * k，矩阵内数据在[0,1]之间初始化
F = np.random.rand(TEST_USER_NUMBER, K)
F_T = F.T
#创建矩阵S，代表中和矩阵，大小为K*L，矩阵内数据在[0,1]之间初始化
S = np.random.rand(K, L)
S_T = S.T
#创建矩阵G，代表物品特征矩阵，大小为TEST_MOVIE_NUMBER * L，矩阵内数据在[0,1]之间初始化
G = np.random.rand(TEST_MOVIE_NUMBER, L)
G_T = G.T
#开始矩阵分解迭代
while(np.linalg.norm(test_rating_matrix - Matrix_Matmul([F, S, G.T])) > MIN):
    #print(np.linalg.norm(test_rating_matrix - Matrix_Matmul([F, S, G.T])))
    #先迭代更新G
    temp_G_up = Matrix_Matmul([test_rating_matrix.T, F, S])
    temp_G_down = Matrix_Matmul([G, G.T, test_rating_matrix.T, F, S])
    for i in range(0, temp_G_up.shape[0]):
        for j in range(0, temp_G_up.shape[1]):
            if temp_G_down[i][j] != 0:
                G[i][j] = G[i][j] * (temp_G_up[i][j] / temp_G_down[i][j])**0.5
    #再迭代更新F
    temp_F_up = Matrix_Matmul([test_rating_matrix, G, S.T])
    temp_F_down = Matrix_Matmul([F, F.T, test_rating_matrix, G, S.T])
    for i in range(0, temp_F_up.shape[0]):
        for j in range(0, temp_F_up.shape[1]):
            if temp_F_down[i][j] != 0:
                F[i][j] = F[i][j] * (temp_F_up[i][j] / temp_F_down[i][j])**0.5
    #再迭代更新S
    temp_S_up = Matrix_Matmul([F.T, test_rating_matrix, G])
    temp_S_down = Matrix_Matmul([F.T, F, S, G.T, G])
    for i in range(0, temp_S_up.shape[0]):
        for j in range(0, temp_S_up.shape[1]):
            if temp_S_down[i][j] != 0:
                S[i][j] = S[i][j] * (temp_S_up[i][j] / temp_S_down[i][j])**0.5
print(np.linalg.norm(test_rating_matrix - Matrix_Matmul([F, S, G.T])))
#从矩阵F和G中找最大值
max_F = 0
max_G = 0
for i in range(0, F.shape[0]):
    for j in range(0, F.shape[1]):
        if max_F < F[i][j]: max_F = F[i][j]
for i in range(0, G.shape[0]):
    for j in range(0, G.shape[1]):
        if max_G < G[i][j]: max_G = G[i][j]
#对矩阵F和G进行归一化
for i in range(0, F.shape[0]):
    for j in range(0, F.shape[1]):
        if(F[i][j] > max_F/2): F[i][j] = 1
        else: F[i][j] = 0
for i in range(0, G.shape[0]):
    for j in range(0, G.shape[1]):
        if(G[i][j] > max_G/2): G[i][j] = 1
        else: G[i][j] = 0
print("ok")





# #创建列表代表全为0的行和全为0的列：
# zero_row = []
# zero_column = []
# #遍历行
# for i in range(0, ratings_matrix.shape[0]):
#     flag = True  # 如果改行或列全为0 flag=True
#     for j in range(0,ratings_matrix.shape[1]):
#         if ratings_matrix[i][j] != 0:
#             flag = False
#     if flag:
#         zero_row.append(i)
# #遍历列
# for i in range(0, ratings_matrix.shape[1]):
#     flag = True  # 如果改行或列全为0 flag=True
#     for j in range(0,ratings_matrix.shape[0]):
#         if ratings_matrix[j][i] != 0:
#             flag = False
#     if flag:
#         zero_column.append(i)
#
# ratings_matrix = np.delete(ratings_matrix,zero_row,axis=0)
# ratings_matrix = np.delete(ratings_matrix,zero_column,axis=1)

#创建将要进行正交矩阵分解的三个辅助矩阵
#先设定中间特征矩阵的参数
k=15
l=15
#创建用户特征矩阵

```python
import pandas as pd

df = pd.read_csv(r'C:\Users\Julio\Google Drive\On the uniqueness of critical state in gothenburg clay\TRIAX ANALYSIS\spexdf_.txt')

df["Sample disturbance factor (1)"]
```




    0     0.0388
    1     0.0723
    2     0.0340
    3     0.0749
    4     0.1652
           ...  
    89    0.0195
    90    0.0353
    91    0.0804
    92    0.0023
    93    0.0107
    Name: Sample disturbance factor (1), Length: 94, dtype: float64




```python
df["Quality of sample"]
```




    0      excellent/good
    1                 bad
    2      excellent/good
    3                 bad
    4                 bad
               ...       
    89     excellent/good
    90     excellent/good
    91                bad
    92     excellent/good
    93     excellent/good
    Name: Quality of sample, Length: 94, dtype: object




```python
df['Quality of sample'].value_counts()
```




     excellent/good            52
     fair/good                 17
    Could not be calculated    14
     bad                       11
    Name: Quality of sample, dtype: int64




```python
df['x pos'] = np.random.randint(0,120, df.shape[0])
df['y pos'] = np.random.randint(0,120, df.shape[0])
```


```python
df1 = df[['Test name','x pos','y pos','Undrained shear strength','Depth (m)']]
```


```python
import matplotlib.pyplot as plt 

x = df1['x pos']
y = df1['y pos']
z = df1['Depth (m)']
f = df1['Undrained shear strength']

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')

ax.axis('tight')
plt.show()
```


![png](output_5_0.png)



```python
import numpy as np
from itertools import combinations
import scipy.linalg


G = np.c_[x, y, z]

F=[]

# quadratic eq.
dim = G.shape[1]
A = np.concatenate((G**2, np.array([np.prod(G[:, k], axis=1) for k in combinations(range(dim), dim-1)]).transpose(), G, np.ones((G.shape[0], 1))), axis=1)
C,C1, C2, C3 = scipy.linalg.lstsq(A, f)
# C will have now the coefficients for:
# f(x, y, z) = ax**2 + by**2 + cz**2 + dxy+ exz + fyz + gx + hy + iz + j

# This can be used then:
def quadratic(a):
    dim = a.shape[0]
    A = np.concatenate((a**2, np.array([np.prod(a[k,]) for k in combinations(range(dim), dim-1)]), a, [1]))
    return np.sum(np.dot(A, C))

for i in range(G.shape[0]):
    F.append(quadratic(G[i,:]))
    print(quadratic(G[i,:]), f[i])
    
```

    117.76123603100676 108.55
    31.87527949963105 -10.77
    39.16218799221401 62.34
    53.92870019237352 61.46
    38.17717165403006 -31.82
    89.09453624496942 110.11
    39.31227152900119 19.28
    30.44990933825777 83.85
    33.61264071768879 30.19
    57.60018501187001 55.71
    40.26452377000553 53.14
    32.996973888302435 60.26
    45.859222094149445 39.42
    29.44429854497887 36.96
    39.860629461231326 53.73
    35.17929113818221 31.83
    38.124521373844374 19.55
    40.556015971833546 28.45
    27.11280859745275 28.45
    24.286799938231557 25.13
    27.11630192176026 42.45
    32.63674462302476 48.17
    39.381473836915234 0.67
    33.185977405144826 23.57
    29.35092330937946 30.82
    31.57396372221489 42.58
    41.292950949105546 52.01
    33.041766596953046 28.64
    28.933108413793704 34.72
    31.15651316439652 31.03
    36.041235005193315 22.39
    32.250960541830594 29.23
    36.57484943922111 20.75
    30.699398157321014 31.32
    33.23755869522667 32.32
    28.245251573754896 27.5
    50.0353950677798 30.16
    40.541521119213165 -2.67
    36.68671318561963 25.22
    35.97053824842025 21.67
    24.062662009256744 23.64
    26.619599104523374 25.35
    38.67971274173476 46.0
    30.835276628965886 25.02
    35.85701129899547 40.69
    33.67414167699852 44.26
    33.122425912054524 -5.64
    43.76875931406877 27.19
    40.26907837938039 49.2
    30.136057451419248 29.72
    44.81130625623945 42.18
    34.04842055916886 21.19
    30.170420815388034 26.72
    27.81817270183469 27.87
    30.456406017793487 39.19
    34.478025308398614 38.28
    31.513964608068257 0.37
    36.9885809447919 141.44
    39.78752697205958 55.38
    73.34195514371174 76.99
    38.43145036178315 50.32
    66.18394940974734 65.61
    51.29427373420352 32.07
    32.990491564110066 29.95
    42.91017243987847 55.38
    72.59109459340672 76.99
    52.26500856507666 65.61
    28.081263226114974 35.86
    25.073358689864516 11.17
    28.72984803014982 72.98
    32.61453550042558 30.34
    38.709479434381315 75.46
    41.12225954345293 119.94
    29.321219865213394 31.78
    46.772648021569765 32.64
    33.31122027133479 45.66
    39.46669385739673 35.53
    31.7144619538592 24.75
    35.74542087003216 23.67
    34.635986445035606 0.17
    30.47671953947406 31.34
    54.10856745141308 49.13
    40.339187595312815 28.26
    26.42143681011095 34.55
    16.82096439112714 25.54
    30.071858141477605 24.15
    24.382998430367966 27.56
    26.19441828361914 32.31
    41.35720166760078 37.43
    22.247664813908987 23.84
    25.566336399893395 37.19
    31.86291954751058 -16.95
    43.88715500909813 68.05
    26.46588373167377 3.53
    


```python
import scipy.stats
scipy.stats.pearsonr(F, f)[0]

```




    0.5186670554980501




```python
scipy.stats.spearmanr(F, f)[0]
```




    0.4187323445489259




```python
scipy.stats.kendalltau(F, f)[0]
```




    0.27901124461505317




```python
df2 = df1[(df1[['Undrained shear strength']] > 0).all(axis=1)]
```


```python
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test name</th>
      <th>x pos</th>
      <th>y pos</th>
      <th>Undrained shear strength</th>
      <th>Depth (m)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRIAX100mm_Marieholmstunneln_bh1306_51m_tLBF96...</td>
      <td>57</td>
      <td>3</td>
      <td>108.55</td>
      <td>51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRIAX_Marieholmstunneln_bh1306_21m_t297_CTH_20...</td>
      <td>0</td>
      <td>20</td>
      <td>62.34</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRIAX_Marieholmstunneln_bh1306_27m_t272_CTH_20...</td>
      <td>82</td>
      <td>33</td>
      <td>61.46</td>
      <td>27</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TRIAX_Marieholmstunneln_bh1306_55m_t89_CTH_201...</td>
      <td>12</td>
      <td>12</td>
      <td>110.11</td>
      <td>55</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TRIAX_Marieholmstunneln_bh1306_6m_t36_CTH_2010...</td>
      <td>35</td>
      <td>101</td>
      <td>19.28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>TRIAX_Västlänken_OC5036_12m_t271_Sweco_130617....</td>
      <td>31</td>
      <td>14</td>
      <td>37.43</td>
      <td>12</td>
    </tr>
    <tr>
      <th>89</th>
      <td>TRIAX_Västlänken_OC5036_4m_t2780_Sweco_130617....</td>
      <td>118</td>
      <td>92</td>
      <td>23.84</td>
      <td>4</td>
    </tr>
    <tr>
      <th>90</th>
      <td>TRIAX_Västlänken_OC5098_12m_t143_Sweco_130617....</td>
      <td>118</td>
      <td>59</td>
      <td>37.19</td>
      <td>12</td>
    </tr>
    <tr>
      <th>92</th>
      <td>TXC_Vasabron_18m_170116.xlsx</td>
      <td>97</td>
      <td>27</td>
      <td>68.05</td>
      <td>18</td>
    </tr>
    <tr>
      <th>93</th>
      <td>TXE_Vasabron_8m_170116.xlsx</td>
      <td>102</td>
      <td>88</td>
      <td>3.53</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>89 rows × 5 columns</p>
</div>




```python
df2.index = range(len(df2.index))
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Test name</th>
      <th>x pos</th>
      <th>y pos</th>
      <th>Undrained shear strength</th>
      <th>Depth (m)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRIAX100mm_Marieholmstunneln_bh1306_51m_tLBF96...</td>
      <td>57</td>
      <td>3</td>
      <td>108.55</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRIAX_Marieholmstunneln_bh1306_21m_t297_CTH_20...</td>
      <td>0</td>
      <td>20</td>
      <td>62.34</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRIAX_Marieholmstunneln_bh1306_27m_t272_CTH_20...</td>
      <td>82</td>
      <td>33</td>
      <td>61.46</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRIAX_Marieholmstunneln_bh1306_55m_t89_CTH_201...</td>
      <td>12</td>
      <td>12</td>
      <td>110.11</td>
      <td>55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRIAX_Marieholmstunneln_bh1306_6m_t36_CTH_2010...</td>
      <td>35</td>
      <td>101</td>
      <td>19.28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>84</th>
      <td>TRIAX_Västlänken_OC5036_12m_t271_Sweco_130617....</td>
      <td>31</td>
      <td>14</td>
      <td>37.43</td>
      <td>12</td>
    </tr>
    <tr>
      <th>85</th>
      <td>TRIAX_Västlänken_OC5036_4m_t2780_Sweco_130617....</td>
      <td>118</td>
      <td>92</td>
      <td>23.84</td>
      <td>4</td>
    </tr>
    <tr>
      <th>86</th>
      <td>TRIAX_Västlänken_OC5098_12m_t143_Sweco_130617....</td>
      <td>118</td>
      <td>59</td>
      <td>37.19</td>
      <td>12</td>
    </tr>
    <tr>
      <th>87</th>
      <td>TXC_Vasabron_18m_170116.xlsx</td>
      <td>97</td>
      <td>27</td>
      <td>68.05</td>
      <td>18</td>
    </tr>
    <tr>
      <th>88</th>
      <td>TXE_Vasabron_8m_170116.xlsx</td>
      <td>102</td>
      <td>88</td>
      <td>3.53</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>89 rows × 5 columns</p>
</div>




```python
import matplotlib.pyplot as plt 

x = df2['x pos']
y = df2['y pos']
z = df2['Depth (m)']
f = df2['Undrained shear strength']

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')

ax.axis('tight')
plt.show()
```


![png](output_13_0.png)



```python
import numpy as np
from itertools import combinations
import scipy.linalg


G = np.c_[x, y, z]

F=[]

# quadratic eq.
dim = G.shape[1]
A2 = np.concatenate((G**2, np.array([np.prod(G[:, k], axis=1) for k in combinations(range(dim), dim-1)]).transpose(), G, np.ones((G.shape[0], 1))), axis=1)
C,C1, C2, C3 = scipy.linalg.lstsq(A2, f)
# C will have now the coefficients for:
# f(x, y, z) = ax**2 + by**2 + cz**2 + dxy+ exz + fyz + gx + hy + iz + j

# This can be used then:
def quadratic(a):
    dim = a.shape[0]
    A2 = np.concatenate((a**2, np.array([np.prod(a[k,]) for k in combinations(range(dim), dim-1)]), a, [1]))
    return np.sum(np.dot(A2, C))

for i in range(G.shape[0]):
    F.append(quadratic(G[i,:]))
    print(quadratic(G[i,:]), f[i])

    
```

    112.98896529269345 108.55
    47.85923739224854 62.34
    57.61011464660213 61.46
    105.638904158951 110.11
    38.04868438624759 19.28
    53.35641363730593 83.85
    30.585486289026797 30.19
    54.87797383086305 55.71
    46.222768077000374 53.14
    41.67409862761481 60.26
    43.691118624100525 39.42
    35.266998961766944 36.96
    42.99396077288128 53.73
    34.47945426502754 31.83
    32.62143389128594 19.55
    34.91053993915429 28.45
    32.34912862145136 28.45
    28.385577096163413 25.13
    32.15378745916738 42.45
    38.27195307071429 48.17
    41.761501099531834 0.67
    35.24498892414146 23.57
    34.396148040738495 30.82
    43.79359993122438 42.58
    41.18759620827216 52.01
    39.84547294566867 28.64
    33.319004870826376 34.72
    32.59226121708531 31.03
    37.629983944328785 22.39
    38.42914036899871 29.23
    34.83831339401363 20.75
    35.209541403465636 31.32
    38.24831416365399 32.32
    30.061854481641095 27.5
    46.78455676902697 30.16
    36.9528944450437 25.22
    32.92352807827395 21.67
    27.17069963726783 23.64
    31.498165240979908 25.35
    42.81151220133961 46.0
    31.64551344927503 25.02
    39.72813976878858 40.69
    38.410218611037756 44.26
    40.064539809185504 27.19
    47.481759467399364 49.2
    30.06580998703459 29.72
    46.97271134706665 42.18
    32.62467447481343 21.19
    33.52461423783192 26.72
    42.15660765320541 27.87
    35.777582662443805 39.19
    33.50684197934399 38.28
    35.33559746595396 0.37
    36.77165342913535 141.44
    44.77171034032635 55.38
    80.94633811366911 76.99
    43.59000341022035 50.32
    70.25398246591413 65.61
    54.110404823492715 32.07
    50.56097260834804 29.95
    45.83702417512099 55.38
    80.34140811114028 76.99
    49.048004718067205 65.61
    32.48790502219936 35.86
    29.166661319614285 11.17
    32.48502314650848 72.98
    34.85133828921201 30.34
    37.82889518381757 75.46
    36.53662636887083 119.94
    32.83793881752982 31.78
    43.019617770142844 32.64
    37.991417558688774 45.66
    35.272541743931534 35.53
    36.021256784282855 24.75
    33.86492278053933 23.67
    40.876812630459206 0.17
    27.04305228046949 31.34
    52.727509732698735 49.13
    37.69514807145617 28.26
    28.427358134274918 34.55
    22.242618776572442 25.54
    27.319412069295357 24.15
    28.70451476328441 27.56
    28.914254086012214 32.31
    41.02928415032609 37.43
    20.564526004230252 23.84
    30.266051557774595 37.19
    46.38741989172371 68.05
    26.296103551482304 3.53
    


```python
scipy.stats.pearsonr(F, f)[0]
```




    0.6006352660030101




```python
scipy.stats.spearmanr(F, f)[0]
```




    0.5902126356428273




```python
scipy.stats.kendalltau(F, f)[0]
```




    0.4276955080599178




```python
import numpy as np
from itertools import combinations
import scipy.linalg


G = np.c_[x, y, z]
F=[]
comb = (np.array([np.prod(G[:, k], axis=1) for k in combinations(range(dim), dim-1)]).transpose())

comcub = np.array([G[:,0]*comb[:,0],
                  G[:,0]*comb[:,1],
                  G[:,1]*comb[:,0],
                  G[:,1]*comb[:,2],
                  G[:,2]*comb[:,1],
                  G[:,2]*comb[:,2],
                  G[:,0]*comb[:,2]+
                  G[:,1]*comb[:,1]+
                  G[:,2]*comb[:,0]]).transpose()

dim = G.shape[1]


# quadratic eq.
dim = G.shape[1]
A3 = np.concatenate((G**3,comcub,G**2,comb,G,np.ones((G.shape[0], 1))), axis=1)
C,C1, C2, C3 = scipy.linalg.lstsq(A3, f)
# C will have now the coefficients for:
# f(x, y, z) = ax**3 + by**3 + cz**3 + dx**2y+ ex**2z + fy**2z + gz**2y + hy**2x + iz**2x + jxyz + kx**2 + ly**2 + mz**2 + nxz + oxy + pzy + rx + sy + tz + u



for i in range(G.shape[0]):
    F.append(np.sum(np.dot(A3[i,:], C)))
    print(np.sum(np.dot(A3[i,:], C)),f[i],z[i])

```

    112.09154870692912 108.55 51
    62.95232582487589 62.34 21
    63.31887320882023 61.46 27
    110.47677199667224 110.11 55
    45.77565959178392 19.28 6
    66.35817130736619 83.85 30
    27.107126112026318 30.19 10
    41.84513827587708 55.71 20
    44.46669804532488 53.14 20
    43.12930514847829 60.26 20
    50.29279355966692 39.42 10
    34.30443301305631 36.96 10
    37.2115401665576 53.73 21
    35.70496558493811 31.83 6
    32.184643301745226 19.55 10
    39.50029080131135 28.45 10
    33.221677347604796 28.45 10
    27.633278677750187 25.13 8
    31.17736613940128 42.45 12
    37.23688020682984 48.17 18
    40.39639049940975 0.67 24
    39.13800507344557 23.57 6
    32.54111453857301 30.82 8
    31.547089224987147 42.58 15
    29.857425141302826 52.01 21
    20.482595983239225 28.64 8
    35.2498191684073 34.72 10
    32.064540316182146 31.03 5
    36.198743937318994 22.39 10
    39.87010977813567 29.23 15
    36.06051932262392 20.75 6
    37.26125943858789 31.32 12
    37.39536641592696 32.32 18
    35.501509971773245 27.5 6
    47.180005077584845 30.16 15
    33.7628071759533 25.22 8
    30.282879178421943 21.67 3
    26.988201567751076 23.64 7
    30.378539009583886 25.35 10
    48.30145182336011 46.0 21
    31.280491638452276 25.02 4
    37.938571287653616 40.69 18
    39.84649538588795 44.26 15
    50.08986568354019 27.19 5
    46.05077799131949 49.2 21
    35.383845428437596 29.72 7
    46.37354420324162 42.18 18
    32.65158782385889 21.19 4
    31.891587798891724 26.72 8
    41.20497677492594 27.87 18
    34.55144594728877 39.19 15
    42.75731082480526 38.28 7
    27.797447116564292 0.37 7
    44.516755089505864 141.44 7
    46.35659261795677 55.38 20
    70.41863264387145 76.99 40
    40.377009153047865 50.32 25
    61.3094707802685 65.61 33
    60.91758920807017 32.07 23
    54.141010367535344 29.95 30
    57.271797513825796 55.38 20
    75.85731780556996 76.99 40
    52.124964483133596 65.61 33
    33.55719340598996 35.86 10
    26.101836453363045 11.17 10
    31.77366876338378 72.98 7
    32.80245505203309 30.34 7
    36.11918164018973 75.46 7
    48.66492314896879 119.94 7
    33.856343147537515 31.78 11
    37.60315502555246 32.64 11
    38.3156865787098 45.66 15
    41.13743817869009 35.53 10
    26.524710509692095 24.75 6
    35.58345804158267 23.67 5
    39.742173044566286 0.17 21
    24.399777684652108 31.34 7
    58.718466934049616 49.13 18
    35.829138585770266 28.26 6
    28.404476471751213 34.55 8
    23.202783270486307 25.54 3
    26.021856800564606 24.15 7
    29.38947438892782 27.56 6
    29.58240490838644 32.31 8
    43.75293020026782 37.43 12
    10.381261371642111 23.84 4
    27.215926019782408 37.19 12
    46.520687956488786 68.05 18
    20.311645189738815 3.53 8
    


```python
scipy.stats.pearsonr(F, f)[0]
```




    0.6442150384236122




```python
scipy.stats.spearmanr(F, f)[0]
```




    0.5857352015694159




```python
scipy.stats.kendalltau(F, f)[0]
```




    0.42360761789924956




```python
C
```




    array([ 1.79250984e-05, -2.41019669e-06,  8.17952656e-04, -1.37514391e-04,
            2.87540589e-04, -7.68634270e-05, -4.28939782e-04,  9.13149096e-04,
            5.39642097e-04, -4.99031672e-05, -4.93286503e-04,  1.16814614e-02,
           -7.34910003e-02,  2.48992554e-02, -5.04811954e-02,  3.02518462e-02,
           -2.83271918e-01, -1.49511521e+00,  2.26279477e+00,  5.16547839e+01])




```python
from mayavi import mlab
import numpy as np

x2, y2, z2 = np.ogrid[0:120:120j, 0:120:120j, 0:60:60j]
s = 1.79e-5*x2**3 - 2.41e-6*y2**3 + 8.18e-4*z2**3 - 1.37e-4*x2**2*y2 + 2.88e-4*x2**2*z2 - 7.69e-5*x2*y2**2 - 4.29e-4*y2**2*z2 + 9.13e-4*x2*z2**2 + 5.39e-4*y2*z2**2 - 5e-4*x2*y2*z2 - 4.93e-4*x2**2 + 1.17e-2*y2**2 - 7.35e-2*z2**2 + 2.49e-2*x2*y2 - 5.05e-2*x2*z2 - 3.02e-2*y2*z2 - 2.83e-1*x2 - 1.5*y2 + 2.26*z2 + 51.7

mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                            plane_orientation='x_axes',
                            slice_index=20,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                            plane_orientation='y_axes',
                            slice_index=20,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                            plane_orientation='z_axes',
                            slice_index=20,
                        )
mlab.axes(x_axis_visibility=True,y_axis_visibility=True,z_axis_visibility=True,
          xlabel='x (m)',ylabel='y (m)',zlabel='Depth (m)')

mlab.colorbar(orientation='vertical',nb_colors=10)

mlab.points3d(x,y,z,f)

mlab.title('My graph')

mlab.outline()
mlab.show()

```


```python
x,y,z,f
```




    (0      57
     1       0
     2      82
     3      12
     4      35
          ... 
     84     31
     85    118
     86    118
     87     97
     88    102
     Name: x pos, Length: 89, dtype: int32,
     0       3
     1      20
     2      33
     3      12
     4     101
          ... 
     84     14
     85     92
     86     59
     87     27
     88     88
     Name: y pos, Length: 89, dtype: int32,
     0     51
     1     21
     2     27
     3     55
     4      6
           ..
     84    12
     85     4
     86    12
     87    18
     88     8
     Name: Depth (m), Length: 89, dtype: int64,
     0     108.55
     1      62.34
     2      61.46
     3     110.11
     4      19.28
            ...  
     84     37.43
     85     23.84
     86     37.19
     87     68.05
     88      3.53
     Name: Undrained shear strength, Length: 89, dtype: float64)




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

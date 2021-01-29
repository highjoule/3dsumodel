```python

```


```python

```


```python

```

# 3dsumodel

This code will take the information of an existing database set of 94 triaxial tests analysed in an area of 120 km. The parameter taken is the undrained shear stregth and the depth of the sample where was extracted, this stress parameter measures the strength under critical state of soft soil.

I have already an existant database of 94 compression triaxial tests. I will change the name of all the tests and will upload a version in this repository of the set analysed here.




```python
import pandas as pd

df = pd.read_csv(r'C:\Users\Julio\Documents\Github_files\df1.txt')
df
```




<div>


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Undrained shear strength</th>
      <th>x pos</th>
      <th>y pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>108.55</td>
      <td>107</td>
      <td>68</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-10.77</td>
      <td>10</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62.34</td>
      <td>15</td>
      <td>44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61.46</td>
      <td>2</td>
      <td>97</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-31.82</td>
      <td>49</td>
      <td>82</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>89</th>
      <td>23.84</td>
      <td>11</td>
      <td>84</td>
    </tr>
    <tr>
      <th>90</th>
      <td>37.19</td>
      <td>99</td>
      <td>82</td>
    </tr>
    <tr>
      <th>91</th>
      <td>-16.95</td>
      <td>58</td>
      <td>31</td>
    </tr>
    <tr>
      <th>92</th>
      <td>68.05</td>
      <td>104</td>
      <td>65</td>
    </tr>
    <tr>
      <th>93</th>
      <td>3.53</td>
      <td>64</td>
      <td>55</td>
    </tr>
  </tbody>
</table>
<p>94 rows × 3 columns</p>
</div>



Firstly, the points of the tests are visualised in the lines below using matplotlib


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
ax.invert_zaxis()

ax.axis('tight')
plt.show()
```


![alt text](https://github.com/highjoule/3dsumodel/blob/main/images/3dpoints.png)

# Quadratic fit

In this part scipy is used to model the shear undrained stregth based on the position of the sample taken:

![](https://latex.codecogs.com/gif.latex?f(x,y,z)=ax^2&space;&plus;&space;by^2&space;&plus;&space;cz^2&space;&plus;&space;dxz&space;&plus;&space;eyz&space;&plus;&space;gxy&space;&plus;&space;hx&space;&plus;&space;iy&space;&plus;&space;jz&space;&plus;&space;k)

In order to accomplish this, scipy.linalg.lstsq is used. This solves the equation Ax = b

On the left, coefficients arrays are listed and on the rigth the results is in a different array. 

After computing the solution of the quadratic equation, coeffcients are listed in the array C. The points in the position matrix are evaluated in the model to check accuracy in the model.


```python
import numpy as np
from itertools import combinations
import scipy.linalg


G = np.c_[x, y, z]#position matrix

F=[]#list that stores the function of the model

# quadratic eq
dim = G.shape[1]
A = np.concatenate((G**2, np.array([np.prod(G[:, k], axis=1) for k in combinations(range(dim), dim-1)]).transpose(), G, np.ones((G.shape[0], 1))), axis=1)
C,C1, C2, C3 = scipy.linalg.lstsq(A, f)
# C will have now the coefficients for:
# f(x, y, z) = ax**2 + by**2 + cz**2 + dxy+ exz + fyz + gx + hy + iz + j

# Accuracy checking
def quadratic(a):
    dim = a.shape[0]
    A = np.concatenate((a**2, np.array([np.prod(a[k,]) for k in combinations(range(dim), dim-1)]), a, [1]))
    return np.sum(np.dot(A, C))

for i in range(G.shape[0]):
    F.append(quadratic(G[i,:]))
    print(quadratic(G[i,:]), f[i])
```
## Usage of the function retrieved in this first attempt and calculation of the points measured 
   
```

    111.42267363866827 108.55
    49.03974059905342 -10.77
    51.2605322999674 62.34
    56.95545464881671 61.46
    56.605059265971995 -31.82
    90.55239499581474 110.11
    22.08794443311629 19.28
    52.9956855668552 83.85
    26.685274666678797 30.19
    24.961940607454302 55.71
    42.01251313250428 53.14
    41.04012552003918 60.26
    48.220978846115116 39.42
    39.9460325979908 36.96
    40.88677537526196 53.73
    35.774901262381974 31.83
    30.22017917090454 19.55
    52.59594302062171 28.45
    32.13908194318395 28.45
    30.702743420612826 25.13
    28.97845573334395 42.45
    22.237339957880213 48.17
    29.944896449682357 0.67
    32.82381046093412 23.57
    27.83906325116989 30.82
    30.248734156832658 42.58
    33.9794759362448 52.01
    27.83906325116989 28.64
    51.80777736094366 34.72
    26.185349186354514 31.03
    33.6979353015718 22.39
    30.27466384303216 29.23
    24.296584551183745 20.75
    24.319099933906816 31.32
    38.93157721091792 32.32
    31.657535883628864 27.5
    33.9830292660994 30.16
    35.48464518499006 -2.67
    33.615082559432544 25.22
    32.48485245135987 21.67
    29.67669643256934 23.64
    48.55080110815838 25.35
    33.80232186612456 46.0
    21.06137763843309 25.02
    41.00416113293602 40.69
    33.195857492781485 44.26
    31.499950412734187 -5.64
    27.548943671791307 27.19
    38.69867029479163 49.2
    23.390499381776227 29.72
    34.01572971253907 42.18
    24.310636705176677 21.19
    37.36943266939819 26.72
    40.3974563510846 27.87
    25.025689300333248 39.19
    44.03527246131432 38.28
    32.27987936505413 0.37
    53.42219164721811 141.44
    45.42757524455682 55.38
    67.46083097732418 76.99
    61.6254247684405 50.32
    59.832579818146016 65.61
    41.83667683237618 32.07
    40.08558752680074 29.95
    38.17174064720976 55.38
    61.057833197613235 76.99
    53.8933563038358 65.61
    22.385514892259884 35.86
    42.596616738777946 11.17
    48.94665463287415 72.98
    28.407821129412454 30.34
    57.00588812401655 75.46
    36.92399269345037 119.94
    32.26802822232469 31.78
    26.86430170249513 32.64
    32.49868625646546 45.66
    26.174416486610056 35.53
    23.142397099493184 24.75
    30.53537008208706 23.67
    33.935892815454324 0.17
    24.336690066915562 31.34
    46.74501294829655 49.13
    22.729408389633736 28.26
    30.072079967646815 34.55
    27.136866324645432 25.54
    27.61294348180397 24.15
    24.436905778316383 27.56
    23.597534977018825 32.31
    28.91349831816339 37.43
    48.76796660813821 23.84
    32.37334239275398 37.19
    32.94308060249996 -16.95
    42.91760935842865 68.05
    25.5393600068028 3.53
```    

## Correlation check

Using scipy.stats an correlation analysis is done to measure the accuracy of the model

### Pearson coefficient


```python
import scipy.stats
scipy.stats.pearsonr(F, f)[0]

```




    0.5374491129145494



### Spearman coefficient


```python
scipy.stats.spearmanr(F, f)[0]
```




    0.44762575637054636



### Kendall Tau coefficient


```python
scipy.stats.kendalltau(F, f)[0]
```




    0.30948840098458213



# Negative results cut

The array analised includes negative shear strength values, since this migth have been brougth as a result from errors on the test phase. In the next line these negative results are cut in order to increase the correlation coefficient.


```python
df2 = df1[(df1[['Undrained shear strength']] > 0).all(axis=1)]
```


```python
df2
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x pos</th>
      <th>y pos</th>
      <th>Undrained shear strength</th>
      <th>Depth (m)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>107</td>
      <td>68</td>
      <td>108.55</td>
      <td>51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>44</td>
      <td>62.34</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>97</td>
      <td>61.46</td>
      <td>27</td>
    </tr>
    <tr>
      <th>5</th>
      <td>67</td>
      <td>88</td>
      <td>110.11</td>
      <td>55</td>
    </tr>
    <tr>
      <th>6</th>
      <td>74</td>
      <td>32</td>
      <td>19.28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>81</td>
      <td>56</td>
      <td>37.43</td>
      <td>12</td>
    </tr>
    <tr>
      <th>89</th>
      <td>11</td>
      <td>84</td>
      <td>23.84</td>
      <td>4</td>
    </tr>
    <tr>
      <th>90</th>
      <td>99</td>
      <td>82</td>
      <td>37.19</td>
      <td>12</td>
    </tr>
    <tr>
      <th>92</th>
      <td>104</td>
      <td>65</td>
      <td>68.05</td>
      <td>18</td>
    </tr>
    <tr>
      <th>93</th>
      <td>64</td>
      <td>55</td>
      <td>3.53</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>89 rows × 4 columns</p>
</div>



## Reset index


```python
df2.index = range(len(df2.index))
df2
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x pos</th>
      <th>y pos</th>
      <th>Undrained shear strength</th>
      <th>Depth (m)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>107</td>
      <td>68</td>
      <td>108.55</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>44</td>
      <td>62.34</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>97</td>
      <td>61.46</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>67</td>
      <td>88</td>
      <td>110.11</td>
      <td>55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74</td>
      <td>32</td>
      <td>19.28</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>84</th>
      <td>81</td>
      <td>56</td>
      <td>37.43</td>
      <td>12</td>
    </tr>
    <tr>
      <th>85</th>
      <td>11</td>
      <td>84</td>
      <td>23.84</td>
      <td>4</td>
    </tr>
    <tr>
      <th>86</th>
      <td>99</td>
      <td>82</td>
      <td>37.19</td>
      <td>12</td>
    </tr>
    <tr>
      <th>87</th>
      <td>104</td>
      <td>65</td>
      <td>68.05</td>
      <td>18</td>
    </tr>
    <tr>
      <th>88</th>
      <td>64</td>
      <td>55</td>
      <td>3.53</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>89 rows × 4 columns</p>
</div>



# Second quadratic model iteration


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
## Usage of the function retrieved in this second attempt and calculation of the points measured 

```

    121.89347007846484 108.55
    57.90991899379328 62.34
    61.48146165175865 61.46
    102.00647518395829 110.11
    23.660115201425196 19.28
    64.78027208060794 83.85
    31.740634016998925 30.19
    31.309475450495665 55.71
    45.943390451828584 53.14
    44.418809749560424 60.26
    52.349208604151315 39.42
    39.70264879845395 36.96
    49.80108263874633 53.73
    39.035249267547066 31.83
    30.927886759822556 19.55
    50.905151925225205 28.45
    35.610597016886935 28.45
    28.720939341337715 25.13
    35.592937988961225 42.45
    27.14703288686668 48.17
    37.18798740031039 0.67
    27.935903233914193 23.57
    28.993331596276228 30.82
    32.8942039064081 42.58
    36.78196887565257 52.01
    28.993331596276228 28.64
    50.97965555933153 34.72
    26.41996047878967 31.03
    31.87925159822739 22.39
    35.25754118211478 29.23
    27.53239184381991 20.75
    25.5822767743152 31.32
    45.0273124555153 32.32
    27.141276640474043 27.5
    41.59226387005141 30.16
    30.136743371820938 25.22
    34.64719125351209 21.67
    30.473412054645586 23.64
    52.44732630545185 25.35
    36.00753385304023 46.0
    20.64185381886448 25.02
    44.277318394367974 40.69
    39.08085876992386 44.26
    21.83883862552498 27.19
    43.57662334335002 49.2
    26.204768529591988 29.72
    35.52047574610533 42.18
    20.688236170101103 21.19
    41.85324528483652 26.72
    44.69749944693916 27.87
    24.93503214429375 39.19
    40.35455690334175 38.28
    35.100021321554244 0.37
    55.97337263122743 141.44
    52.87386304255301 55.38
    72.69017972700264 76.99
    56.139124797902426 50.32
    68.03519734570419 65.61
    50.98419872963869 32.07
    50.837312988512394 29.95
    45.552298198463106 55.38
    70.83710099080187 76.99
    59.989658871448356 65.61
    23.07725252646187 35.86
    47.29650563709851 11.17
    49.65194432780574 72.98
    28.54851841258315 30.34
    59.15522298567079 75.46
    41.01226951196041 119.94
    38.01463395197206 31.78
    29.393995087606815 32.64
    36.09345502155321 45.66
    31.43483864718064 35.53
    25.771988891889226 24.75
    34.03073878491208 23.67
    36.53244790587249 0.17
    27.260968772409868 31.34
    53.75357449861561 49.13
    24.327536387970785 28.26
    28.686575541032187 34.55
    21.638794079455057 25.54
    32.05996246075499 24.15
    27.92320293793304 27.56
    23.3989471696829 32.31
    34.413770603975344 37.43
    48.010969726210604 23.84
    36.56705453103463 37.19
    49.36685409804783 68.05
    30.118719741420584 3.53
    
```
## Pearson coefficient 2nd iteration


```python
scipy.stats.pearsonr(F, f)[0]
```




    0.6728982346382903



## Spearman coefficient 2nd iteration



```python
scipy.stats.spearmanr(F, f)[0]
```




    0.5894235066769764



## Kendall Tau coefficient 2nd iteration



```python
scipy.stats.kendalltau(F, f)[0]
```




    0.4198288285899355



# Third iteration


## Cubic model

Below the model is computed to retrieve te model in a cubic function in the form:

![](https://latex.codecogs.com/gif.latex?f(x,y,z)=ax^3&space;&plus;&space;by^3&space;&plus;&space;cz^3&space;&plus;&space;dx^2z&space;&plus;&space;ey^2z&space;&plus;&space;gx^2y&space;&plus;&space;hxz^2&space;&plus;&space;iyz^2&space;&plus;&space;jxy^2&space;&plus;&space;kxz&space;&plus;&space;lyz&space;&plus;&space;mxy&space;&plus;&space;nx^2&space;&plus;&space;oy^2&space;&plus;&space;pxyz&space;&plus;&space;qz^2&space;&plus;&space;rx&space;&plus;&space;sy&space;&plus;&space;tz&space;&plus;&space;u)




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


# cubic eq.
dim = G.shape[1]
A3 = np.concatenate((G**3,comcub,G**2,comb,G,np.ones((G.shape[0], 1))), axis=1)
C,C1, C2, C3 = scipy.linalg.lstsq(A3, f)
# C will have now the coefficients for:
# f(x, y, z) = ax**3 + by**3 + cz**3 + dx**2y+ ex**2z + fy**2z + gz**2y + hy**2x + iz**2x + jxyz + kx**2 + ly**2 + mz**2 + nxz + oxy + pzy + rx + sy + tz + u



for i in range(G.shape[0]):
    F.append(np.sum(np.dot(A3[i,:], C)))
    print(np.sum(np.dot(A3[i,:], C)),f[i],z[i])
```
### Usage of the function retrieved in this third attempt and calculation of the points measured, ext to them depth of sample is shown

```

    108.84764263778888 108.55 51
    60.39571590183691 62.34 21
    57.58855918998997 61.46 27
    111.4191213892793 110.11 55
    28.091305248217974 19.28 6
    67.4837520220058 83.85 30
    31.9918561790307 30.19 10
    35.33066783523718 55.71 20
    48.03910492004692 53.14 20
    40.23973926960109 60.26 20
    54.97672180683166 39.42 10
    32.82527387219349 36.96 10
    44.762473726535376 53.73 21
    36.04495819655877 31.83 6
    31.605137718139776 19.55 10
    54.61298759843544 28.45 10
    31.71729973522027 28.45 10
    24.819119948657207 25.13 8
    33.62149849138927 42.45 12
    36.1664722998847 48.17 18
    30.287832299598602 0.67 24
    26.125382061798465 23.57 6
    24.936013873386546 30.82 8
    48.88488329971966 42.58 15
    39.82427922475574 52.01 21
    24.936013873386546 28.64 8
    52.19896157024545 34.72 10
    22.31362862338185 31.03 5
    30.189064061371585 22.39 10
    34.138204086001046 29.23 15
    26.954930216698433 20.75 6
    34.39135943542131 31.32 12
    49.071664978208105 32.32 18
    22.896559068613445 27.5 6
    37.40347682321457 30.16 15
    21.744596027181263 25.22 8
    32.123052975477485 21.67 3
    28.57354394688115 23.64 7
    55.297777684423 25.35 10
    36.602467698219385 46.0 21
    23.91355194981718 25.02 4
    44.87733805057821 40.69 18
    41.32420645458869 44.26 15
    18.372545013102613 27.19 5
    47.76290412596633 49.2 21
    28.75702711326748 29.72 7
    20.856297402110556 42.18 18
    25.145205063792115 21.19 4
    38.963226194175874 26.72 8
    44.67390748031277 27.87 18
    33.79944083522942 39.19 15
    40.32185376793049 38.28 7
    31.19531541714402 0.37 7
    64.62638278078389 141.44 7
    50.75333168512469 55.38 20
    74.59971617847273 76.99 40
    54.660797420263926 50.32 25
    74.74075439431873 65.61 33
    47.129244194750406 32.07 23
    44.609136472454146 29.95 30
    54.10785035606689 55.38 20
    70.47172495118167 76.99 40
    61.93206867735226 65.61 33
    26.46848455366251 35.86 10
    45.98946868116338 11.17 10
    54.02236040701517 72.98 7
    26.211580588906397 30.34 7
    71.70101609077093 75.46 7
    38.41344018356126 119.94 7
    34.01130130245542 31.78 11
    27.815211549068046 32.64 11
    32.05103929864502 45.66 15
    31.751982604699183 35.53 10
    27.41136737831496 24.75 6
    31.171969022821994 23.67 5
    37.672360348185336 0.17 21
    27.772068404200148 31.34 7
    52.59998457978595 49.13 18
    26.36720800971243 28.26 6
    25.850222840546937 34.55 8
    24.113608360061285 25.54 3
    29.683174661976285 24.15 7
    27.126966121542697 27.56 6
    21.58557075400335 32.31 8
    35.02734194535619 37.43 12
    54.13811908591864 23.84 4
    34.047864965893524 37.19 12
    43.44444403575337 68.05 18
    29.554918432332602 3.53 8
```    

### Pearson coefficient 3rd iteration (cubic model)


```python
scipy.stats.pearsonr(F, f)[0]
```




    0.7026160157381666



### Spearman coefficient 3rd iteration (cubic model)


```python
scipy.stats.spearmanr(F, f)[0]
```




    0.6310740355050403



### Kendall Tau coefficient 3rd iteration (cubic model)


```python
scipy.stats.kendalltau(F, f)[0]
```




    0.4510029716745199



## Coefficients of cubic model


```python
C
```




    array([-1.36081139e-04,  5.90281803e-06,  2.86610758e-04, -2.30807436e-05,
            7.22567957e-05,  1.10179590e-04, -8.77205197e-05, -5.27537191e-04,
           -1.35315129e-04,  9.62747693e-05,  2.90612186e-02, -8.06473265e-03,
            3.98547137e-02, -1.33564507e-02,  5.32786524e-03, -3.97291263e-03,
           -1.71000488e+00,  9.67380654e-01, -1.42353993e-01,  4.84452354e+01])



# Comparison of Pearson's coefficient

## First iteration (quadratic)

0.5374491129145494

## Second iteration (quadratic)

0.6728982346382903

## Third iteration (cubic)

0.7026160157381666

# Visualisation of cubic model in 3d

Using the tool Mayavi, the model in 3d is plotted.


```python
from mayavi import mlab
import numpy as np

x2, y2, z2 = np.ogrid[0:120:120j, 0:120:120j, 0:60:60j]
s = -1.36e-4*x2**3 + 5.9e-6*y2**3 + 2.87e-4*z2**3 - 2.31e-5*x2**2*y2 + 7.22e-5*x2**2*z2 + 1.1e-4*x2*y2**2 - 8.77e-5*y2**2*z2 - 5.28e-4*x2*z2**2 - 1.35e-4*y2*z2**2 + 9.63e-5*x2*y2*z2 + 2.91e-2*x2**2 - 8.06e-3*y2**2 + 3.99e-2*z2**2 - 1.33e-2*x2*y2 + 5.33e-3*x2*z2 - 3.97e-3*y2*z2 - 1.71*x2 + 9.67e-1*y2 - 1.42e-1*z2 + 48.44

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
          xlabel='x (km)',ylabel='y (km)',zlabel='Depth (m)')

mlab.colorbar(orientation='vertical',nb_colors=10,title='Undrained shear strength')

mlab.points3d(x,y,z,f)

mlab.title('3dsumodel')

mlab.outline()
mlab.show()

```
![Heat map of 3d cubic model 1](https://github.com/highjoule/3dsumodel/blob/main/images/mayavi1.png)

```python

```
![Heat map of 3d cubic model 2](https://github.com/highjoule/3dsumodel/blob/main/images/mayavi2.png)

```python

```
![Heat map of 3d cubic model 3](https://github.com/highjoule/3dsumodel/blob/main/images/mayavi3.png)

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

#This is a python program that creates a linear regression model that predicts home prices using the area of the house.
import numpy
import numpy as np
import pandas as pd

from sklearn import linear_model

#create a dataframe
d = [[2600,550000],[3000,565000],[3200,610000],[3600,680000], [4000,725000] ]
df = pd.DataFrame(d,columns=['area','price'])
print('      Data')
print(df)
print('')



x=np.array(df.area).reshape(-1,1)
y=np.array(df.price).reshape(-1,1)
lr = linear_model.LinearRegression()
lr.fit(x,y)

m = lr.coef_
b = lr.intercept_

m = numpy.array_str(m).split('[')[2]
m = m.split(']')[0]
b = numpy.array_str(b).split('[')[1]
print('function for line of best fit')


print('y='+m+'x'+'+'+b)
print('')


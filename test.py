
'''
import matplotlib
matplotlib.use('webAgg')
import matplotlib.pyplot as plt


import seaborn as sb

from scipy.stats import norm
# generate random numbers from N(0,1)
data_normal = norm.rvs(size=10000,loc=0,scale=1)

ax = sb.distplot(data_normal,
                  bins=5,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Normal Distribution', ylabel='Frequency')
plt.show()


import matplotlib; 
import matplotlib.pyplot as plt; 
print('Done')
plt.plot([1,2,3]);
plt.show()
'''

import pandas as pd
import ipdb as ipdb


xTest = pd.read_excel('Data/ageMatched/ADNI_test_data.xlsx')
xTrain = pd.read_excel('Data/ageMatched/ADNI_train_data.xlsx')


xTest = list(xTest["Paths"])
xTrain = list(xTrain["Paths"])

xTest = xTest[1:]
xTrain = xTrain[1:]

count = 0
stringy = "  "
for i in range(len(xTest)):
    temp = False
    a = xTest[i].replace("'", '')
    a = list(a.split(','))
    a1 = int(a[0])
    a2 = int(a[1])
    for j in range(len(xTrain)):
        b = xTrain[j].replace("'", '')
        b = list(b.split(','))
        b1 = int(b[0])
        b2 = int(b[1])
        
        if a1 == b1 and a2 == b2:
            temp = True
            #print(a1, " ",a2, " ",b1, " ",b2)

        elif a1 == b2 and a2 == b1:
            temp = True
            #print(a1, " ",a2, " ",b1, " ",b2)
    if temp == False:
        count += 1    
        print(count,"    ",a, "  doesn't exist      ",a1, " ",a2, " ",b1, " ",b2)
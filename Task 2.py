import pandas as pd
import datetime
import numpy as np
from datetime import timedelta
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns
import datetime as dt
from scipy.stats import ks_2samp,ttest_ind,t

# =============================================================================
# # Importing datasets
# =============================================================================
data = pd.read_csv(r"D:\Downloads\quantium python\Task 2\QVI_data.csv")
check_stores = pd.DataFrame({'FREQ':data.groupby('STORE_NBR').size().sort_values(ascending=False)}).reset_index()
# To extract specific month 
# Convert date col from object to datetime to get month_ID
data['DATE'] = pd.to_datetime(data['DATE'])
data['month_ID'] = data['DATE'].dt.strftime('%Y%m')
## data.dtypes

store_sales  = data.groupby(['STORE_NBR', 'month_ID']).agg(
    {'TOT_SALES':'sum' ,
     'LYLTY_CARD_NBR': 'nunique',
     'PROD_QTY':'sum',
     'TXN_ID':'nunique'
     }).reset_index()
store_sales.columns = store_sales.columns.str.replace("LYLTY_CARD_NBR", "nCustomers")
store_sales['Trans_per_Cus'] =store_sales['TXN_ID']/store_sales['nCustomers']
store_sales['chips_per_Cus'] =store_sales['PROD_QTY']/store_sales['nCustomers']
store_sales['Avg_Price_P_unit'] =store_sales['TOT_SALES']/store_sales['PROD_QTY']
store_sales.drop(['TXN_ID'],axis=1,inplace=True)
store_sales.dtypes
# Convert the 'object' dtype column to 'int' dtype
store_sales['month_ID'] = store_sales['month_ID'].astype(int)
# Setting 'STORE_NBR' and 'month_ID' as indexs to be able to  
# apply upcoming comparison (in line 40)
store_sales = store_sales.set_index(['STORE_NBR', 'month_ID'])

# =============================================================================
# Splitting the data to pre_trial and trial period
# =============================================================================
# Copy data before splitting
full=store_sales.copy()
trial=[]
for i in store_sales.index:
    if(i[1]>=201902):
        if(i[1]<=201904):
            trial.append(store_sales.loc[i])
        store_sales.drop(i,inplace=True)
trial=pd.DataFrame(trial)
# Malhash lazma
# # convert index (i,j) to 2 seperate indexs
# trial.index.name=('Index')
# k=0
# trial['STORE_NBR']=0
# trial['MONTHYEAR']=0
# for (i,j) in trial.index:
#     trial['STORE_NBR'][k]=i
#     trial['MONTHYEAR'][k]=j 
#     k=k+1
# trial=trial.set_index(['STORE_NBR','MONTHYEAR'])

# =============================================================================
# Functions to find correlation and magnitude of any store wih another store
# =============================================================================
def calcCorr(store):
    '''
    input=store number which is to be compared
    output=dataframe with corelation coefficient values
    '''
    a=[]
    metrix=store_sales[['TOT_SALES','nCustomers']]#add metrics as required e.g. ,'TXN_PER_CUST'
    for i in metrix.index:
        a.append(metrix.loc[store].corrwith(metrix.loc[i[0]]))
    df= pd.DataFrame(a)
    df.index=metrix.index
    df=df.drop_duplicates()
    # Take the first col idx only
    df.index=[s[0] for s in df.index]
    df.index.name="STORE_NBR"
    return df
def standardizer(df):
    '''
    input=dataframe with metrics
    output=dataframe with mean of the metrics in a new column
    '''
    df=df.abs()
    df['MAGNITUDE']=df.mean(axis=1)
    return df

# =============================================================================
# Finding stores corelated to store 77 to find control store
# =============================================================================
corr77=calcCorr(77)
corr77=standardizer(corr77)
corr77=corr77.sort_values(['MAGNITUDE'],ascending=False).dropna()
# =============================================================================
# We found that stores 233,119,71 are the most correlated to store 77
# Selecting 233 as control store as it has max correlation
# 
# Visualizing pre-trail period
# =============================================================================
# Taking 0.7 as threshold corelation
''' Magnitude > 0.7 '''
corr77[(corr77.MAGNITUDE.abs()>0.7)].plot(kind='bar',figsize=(15,8))
plt.show()
''' TOT_SALES > 0.7 '''
sns.heatmap(corr77[corr77.TOT_SALES.abs()>0.7])
# In pre-trail period 
sns.distplot(store_sales.loc[77]['TOT_SALES'])
sns.distplot(store_sales.loc[233]['TOT_SALES'])
sns.distplot(store_sales.loc[19]['TOT_SALES'])
plt.legend(labels=['77','233','19'])
plt.show()
''' nCustomers > 0.7 '''
# good plot
corr77[(corr77.nCustomers>0.7)].plot(kind='bar',figsize=(15,8))
plt.show()
# In pre-trail period 
sns.distplot(store_sales.loc[77]['nCustomers'])
sns.distplot(store_sales.loc[233]['nCustomers'])
sns.distplot(store_sales.loc[19]['nCustomers'])
plt.legend(labels=['77','233','19'])
plt.show()

# =============================================================================
# Since distributions of store 233 are similar to that of store 77, selecting store 233 as control store with max similarities to store 77
# 
# Calculating difference between scaled control sales and trial sales
# Let null hypothesis be that both stores 77 ans 233 have no difference
# =============================================================================
# difference between control and trial sales
a=[]
for x in store_sales.columns:
    a.append(ks_2samp(store_sales.loc[77][x], store_sales.loc[233][x]))
a=pd.DataFrame(a,index=store_sales.columns)
# =============================================================================
# For pre trial period, since p-values for TOT_SALES, CUSTOMERS and PROD_QTY are high (say more than 0.95), we can't reject the null hypothesis
# 
# Assessment of trial
# The trial period goes from the start of February 2019 to April 2019. We now want to see if there has been an uplift in overall chip sales.
# =============================================================================
b=[]
for x in trial.columns:
    b.append(ttest_ind(trial.loc[77][x], trial.loc[233][x]))
b=pd.DataFrame(b,index=store_sales.columns)
# critical value
t.ppf(0.95,df=6)
## 1.9431802803927816
# =============================================================================
# Since all of the p-values are high (say more than 0.05), we reject the null hypothesis i.e. there means are significantly different.
# We can observe that the t-value is much larger than the 95th percentile value of the t-distribution for March and April - i.e. the increase in sales in the trial store in March and April is statistically greater than in the control store.
# 
# Vizualizing
# =============================================================================
sns.distplot(trial.loc[77]['TOT_SALES'])
sns.distplot(trial.loc[233]['TOT_SALES'])
plt.legend(labels=['77','233'])
plt.show()
sns.distplot(trial.loc[77]['nCustomers'])
sns.distplot(trial.loc[233]['nCustomers'])
plt.legend(labels=['77','233'])
plt.show()
## Note: It can be visualized that the is a significant difference in the means, so trial store behavior(77) is different from control store (233).
## The results show that the trial in store 77 is significantly different to its control store in the trial period as the trial store performance lies outside the 5% to 95% confidence interval of the control store in two of the three trial months.
# =============================================================================
# Finding stores corelated to store 86 to find control store
# =============================================================================
corr86=calcCorr(86)
corr86=standardizer(corr86)
corr86=corr86.sort_values(['MAGNITUDE'],ascending=False).dropna()
# =============================================================================
# We found that stores 155,23,120 are the most correlated to store 77
# Selecting 155 as control store as it has max correlation
# 
# Visualizing pre-rail period
# =============================================================================
# Taking 0.7 as threshold corelation
''' Magnitude > 0.7 '''
corr86[(corr86.MAGNITUDE.abs()>0.7)].plot(kind='bar',figsize=(15,8))
''' TOT_SALES > 0.7 '''
sns.heatmap(corr86[corr86.TOT_SALES.abs()>0.7])
# In pre-trail period 
sns.distplot(store_sales.loc[86]['TOT_SALES'])
sns.distplot(store_sales.loc[155]['TOT_SALES'])
sns.distplot(store_sales.loc[19]['TOT_SALES'])
plt.legend(labels=['86','155','19'])
plt.show()
''' nCustomers > 0.7 '''
corr86[(corr86.nCustomers>0.7)].plot(kind='bar',figsize=(15,8))
# In pre-trail period 
sns.distplot(store_sales.loc[86]['nCustomers'])
sns.distplot(store_sales.loc[155]['nCustomers'])
sns.distplot(store_sales.loc[19]['nCustomers'])
plt.legend(labels=['86','155','19'])
plt.show()

# =============================================================================
# Since distributions of store 155 are similar to that of store 86, selecting store 233 as control store with max similarities to store 86
# 
# Calculating difference between scaled control sales and trial sales
# Let null hypothesis be that both stores 86 and 155 have no difference
# =============================================================================
# difference between control and trial sales
a2=[]
for x in store_sales.columns:
    a2.append(ks_2samp(store_sales.loc[86][x], store_sales.loc[155][x]))
a2=pd.DataFrame(a2,index=store_sales.columns)
# =============================================================================
# For pre trial period, since p-values for TOT_SALES, CUSTOMERS and PROD_QTY are high (say more than 0.95), we can't reject the null hypothesis
# 
# Assessment of trial
# The trial period goes from the start of February 2019 to April 2019. We now want to see if there has been an uplift in overall chip sales.
# =============================================================================
b2=[]
for x in trial.columns:
    b2.append(ttest_ind(trial.loc[86][x], trial.loc[155][x]))
b2=pd.DataFrame(b2,index=store_sales.columns)
# critical value
t.ppf(0.95,df=7)
## 1.894578605061305
# =============================================================================
# Since all of the p-values are high (say more than 0.05), we reject the null hypothesis i.e. there means are significantly different.
# We can observe that the t-value is much larger than the 95th percentile value of the t-distribution for March and April - i.e. the increase in sales in the trial store in March and April is statistically greater than in the control store.
# 
# Vizualizing Trail period
# =============================================================================
sns.distplot(trial.loc[86]['TOT_SALES'])
sns.distplot(trial.loc[155]['TOT_SALES'])
plt.legend(labels=['86','155'])
plt.show()
sns.distplot(trial.loc[86]['nCustomers'])
sns.distplot(trial.loc[155]['nCustomers'])
plt.legend(labels=['86','155'])
plt.show()
## Note: It can be visualized that the is a significant difference in the means, so trial store behavior(86) is different from control store (155).
## It looks like the number of customers is significantly higher in all of the three months. This seems to suggest that the trial had a significant impact on increasingthe number of customers in trial store 86 but as we saw, sales were not significantly higher. We should check with the Category Manager if there were special deals in the trial store that were may have resulted in lower prices, impacting the results.
# =============================================================================
# Finding stores corelated to store 88 to find control store
# =============================================================================
corr88=calcCorr(88)
corr88=standardizer(corr88)
corr88=corr88.sort_values(['MAGNITUDE'],ascending=False).dropna()
# =============================================================================
# We found that stores 178,14,133 are the most correlated to store 77
# Selecting 178 as control store as it has max correlation
# 
# Visualizing ...
# =============================================================================
# Taking 0.7 as threshold corelation
''' Magnitude > 0.7 '''
corr88[(corr88.MAGNITUDE.abs()>0.7)].plot(kind='bar',figsize=(15,8))
''' TOT_SALES > 0.7 '''
sns.heatmap(corr88[corr88.TOT_SALES.abs()>0.7])
# In pre-trail period 
sns.distplot(store_sales.loc[88]['TOT_SALES'])
sns.distplot(store_sales.loc[178]['TOT_SALES'])
sns.distplot(store_sales.loc[19]['TOT_SALES'])
plt.legend(labels=['88','178','19'])
plt.show()
''' nCustomers > 0.7 '''
corr88[(corr88.nCustomers>0.7)].plot(kind='bar',figsize=(15,8))
# In pre-trail period 
sns.distplot(store_sales.loc[88]['nCustomers'])
sns.distplot(store_sales.loc[178]['nCustomers'])
sns.distplot(store_sales.loc[19]['nCustomers'])
plt.legend(labels=['88','178','19'])
plt.show()

# =============================================================================
# Since distributions of store 178 are similar to that of store 88, selecting store 178 as control store with max similarities to store 88
# 
# Calculating difference between scaled control sales and trial sales
# Let null hypothesis be that both stores 88 and 178 have no difference
# =============================================================================
# difference between control and trial sales
a3=[]
for x in store_sales.columns:
    a3.append(ks_2samp(store_sales.loc[88][x], store_sales.loc[178][x]))
a3=pd.DataFrame(a3,index=store_sales.columns)
# =============================================================================
# For pre trial period, since all of the p-values are high (more than 0.05), we can't reject the null hypothesis
# 
# Assessment of trial
# The trial period goes from the start of February 2019 to April 2019. We now want to see if there has been an uplift in overall chip sales.
# =============================================================================
b3=[]
for x in trial.columns:
    b3.append(ttest_ind(trial.loc[88][x], trial.loc[178][x]))
b3=pd.DataFrame(b3,index=store_sales.columns)
# critical value
t.ppf(0.95,df=7)
## 1.894578605061305
# =============================================================================
# Since all of the p-values are high (say more than 0.05), we reject the null hypothesis i.e. there means are significantly different.
# We can observe that the t-value is much larger than the 95th percentile value of the t-distribution for March and April - i.e. the increase in sales in the trial store in March and April is statistically greater than in the control store.
# 
# Vizualizing Trail period
# =============================================================================
sns.distplot(trial.loc[88]['TOT_SALES'])
sns.distplot(trial.loc[178]['TOT_SALES'])
plt.legend(labels=['88','178'])
plt.show()
sns.distplot(trial.loc[88]['nCustomers'])
sns.distplot(trial.loc[178]['nCustomers'])
plt.legend(labels=['88','178'])
plt.show()
## Note: It can be visualized that the is a significant difference in the means, so trial store behavior(88) is different from control store (237).
## Total number of customers in the trial period for the trial store is significantly higher than the control store for two out of three months, which indicates a positive trial effect.


x =['77','86','88','233','155','178']
a =[]
for i in range(3):
# get the data from TOT_SALES column by knowing for loop technique
# and the location of raws that are needed .
    a.append(store_sales.loc[int(x[i])]['TOT_SALES']/store_sales.loc[int(x[i+3])]['TOT_SALES'])
scalingFactorForControlSales= pd.DataFrame(a)
# Take the control store as index .
scalingFactorForControlSales['STORE_NBR']=[233,155,237]
scalingFactorForControlSales = scalingFactorForControlSales.set_index('STORE_NBR')
# get the data in TOT_SALES column and raws 233,155,237
# # convrt the data form single to multi indexed df.
# out2 = scalingFactorForControlSales.stack(value_name='TOT_SALES')
# # Multi-indexed o/p
# scaledControlSales = store_sales.loc[[233,155,237],'TOT_SALES']*out2
# Single-indexed o/p
zz=store_sales.loc[[233,155,237],'TOT_SALES'].unstack()
scaledControlSales2 = zz*scalingFactorForControlSales

fig, ax = plt.subplots(figsize=(15, 8))
for i in  x:
    sns.lineplot(data=full.loc[int(i)],y='TOT_SALES',x=full.index.get_level_values(1).unique(),label=i)


#ax.set_xlim(201807,201812)
ax.set_xlim(201901,201906)

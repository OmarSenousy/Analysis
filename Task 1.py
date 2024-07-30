import pandas as pd
import datetime
import numpy as np
from datetime import timedelta
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns


# Importing datasets
customerData = pd.read_csv(r'D:\Downloads\quantium python\Task 1\QVI_purchase_behaviour.csv') 
transactionData = pd.read_excel(r"D:\Downloads\quantium python\Task 1\QVI_transaction_data.xlsx")

# Convert Excel time to date format
origin = pd.Timestamp("30/12/1899")
transactionData['DATE'] = transactionData['DATE'].apply(lambda x: origin + pd.Timedelta(days=x))
print(transactionData['DATE'])

# Filter the rows that doesn't contain the substring
# substring = 'Chip'
# filter = transactionData['PROD_NAME'].str.contains(substring)
# filtered_transactionData1 = transactionData[filter]
# filter2 = transactionData['PROD_NAME'].str.contains('Chp')
# filtered_transactionData2 = transactionData[filter2]
# filtered_transactionData = pd.merge(filtered_transactionData1, filtered_transactionData2, how = 'outer')
# Filter the DataFrame based on 'Chip' or 'Chp' in 'PROD_NAME'
# filtered_transactionData = transactionData[transactionData['PROD_NAME'].str.contains('Chip|Chp')]
# filtered_transactionData_packsize = filtered_transactionData['PROD_NAME'].str.extract('(\d+)', expand=False)
filtered_transactionData = transactionData[~transactionData['PROD_NAME'].str.contains('SALSA|Salsa|salsa')]
# Display the filtered data frame
print("Data Frame after removing rows that contain '{substring}' in 'Name' column:")
print(filtered_transactionData)

# Remove special char, digits and the last letter(g) 
filtered_transactionData['PROD_NAME'] = filtered_transactionData['PROD_NAME'].str.replace(r'\W+', '',regex=True)
# Taking a copy of the data w/o removing digits
filtered_transactionData_digits = filtered_transactionData.copy()
filtered_transactionData['PROD_NAME'] = filtered_transactionData['PROD_NAME'].str.replace(r'\d+','',regex=True)
#df['column_name'] = df['column_name'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
# filtered_transactionData['PROD_NAME'] = filtered_transactionData['PROD_NAME'].str.replace(r'[\W\d]', '', regex=True)
filtered_transactionData['PROD_NAME'] = filtered_transactionData['PROD_NAME'].str.rstrip('g')
# Add space befor each capital letter
filtered_transactionData['PROD_NAME'] = filtered_transactionData['PROD_NAME'].str.replace( r"([A-Z])", r" \1",regex=True).str.strip()
# .str.strip() is another string method that removes leading and trailing whitespace from each string.
# Count the frequency of each prod name #1 #2
# 1 .. as a dict not sorted
# def countWord(input_string):
#     d = {}
#     for word in input_string:
#         try:
#             d[word] += 1
#         except:
#             d[word] = 1

#     for k in d.keys():
#         print ("%s: %d", (k, d[k]))
#     return d    
# freq_transactionData1 = countWord(filtered_transactionData['PROD_NAME'])
# 2 .. As a series and sorted  
freq_transactionData2 = filtered_transactionData.groupby('PROD_NAME').size().sort_values(ascending=False)
# remove Salsa product
# filtered_transactionData = transactionData[transactionData['PROD_NAME'].str.contains('salsa|SALSA')]

# Describing the filtered_transactionData keys by using statistics
statistics = filtered_transactionData.describe()

# Transaction over 200 at once
res = filtered_transactionData[filtered_transactionData.PROD_QTY>=200]
# Get the customer card number 
Card_no = res['LYLTY_CARD_NBR'] # found that both are for the same customer
# Find customer sales
find_customer = filtered_transactionData[filtered_transactionData.LYLTY_CARD_NBR == 226000]
# Remove this customer by index
# filtered_transactionData = filtered_transactionData.drop(filtered_transactionData.LYLTY_CARD_NBR == 226000,axis=1)
filtered_transactionData = filtered_transactionData[filtered_transactionData['LYLTY_CARD_NBR'] != 226000]
# Count no of pruchase by days
filtered_transactionData_freqByDate = pd.DataFrame({'FREQ':filtered_transactionData.groupby('DATE').size().sort_values(ascending=False)}).reset_index()
sort_date = filtered_transactionData_freqByDate.sort_values(by='DATE')
# Construct a data frame from 1 Jul to 30 June
dates = pd.DataFrame({'DATE':(pd.date_range(start='7/1/2018', end='6/30/2019' ))})
# merge 2 DataFrames TO Add the missing day
filtered_transactionData_freqByDate = pd.merge(dates,filtered_transactionData_freqByDate , how='left')
# Plot The DataFrame
filtered_transactionData_freqByDate.plot(x='DATE', y='FREQ')
# Zoom on December Period
december = filtered_transactionData_freqByDate[(filtered_transactionData_freqByDate.DATE > '11/30/2018')&( filtered_transactionData_freqByDate.DATE< '1/1/2019')] 
# Plot December period
december.plot(x='DATE', y='FREQ')
plt.xticks(rotation=90)
plt.show()

# Extract the digits from (PROD NAME) Column
filtered_transactionData_digits['PACK_SIZE'] = filtered_transactionData_digits['PROD_NAME'].str.extract('(\d+)', expand=False)
## Add g that refers to grams in (PACK_SIZE) column, freq. of packsize,
##filtered_transactionData_digits['PACK_SIZE'] =  filtered_transactionData_digits['PACK_SIZE'].astype(str) + 'g' 
# Also drawing the histogram of pack size
filtered_transactionData_digitfreqByPS = pd.DataFrame({'FREQ':filtered_transactionData_digits.groupby('PACK_SIZE').size().sort_values(ascending=False)}).reset_index()
# Convert Pack size column to integer to apply histogram
filtered_transactionData_digits['PACK_SIZE'] = filtered_transactionData_digits['PACK_SIZE'].astype(int)
filtered_transactionData_digits.PACK_SIZE.plot( kind = 'hist')
plt.xlabel('Pack Size')
plt.show()

# Add space befor each capital letter and extract First Word
filtered_transactionData_digits['PROD_NAME'] = filtered_transactionData_digits['PROD_NAME'].str.replace( r"([A-Z])", r" \1",regex=True).str.strip()
filtered_transactionData_digits['BRAND'] = filtered_transactionData_digits['PROD_NAME'].str.split(' ').str.get(0)
# Getting all of the brand names and frequency in a seperate dataframe 
filtered_transactionData_digits_freqbybrand = pd.DataFrame({'FREQ':filtered_transactionData_digits.groupby('BRAND').size().sort_values(ascending=False)}).reset_index()
# Found that there're only 2 close BRAND names 	Doritos & Dorito
# So we will change both of their names to Dorito in PROD & BRAND NAME 
# filtered_transactionData_digits['BRAND'] = filtered_transactionData_digits['BRAND'].str.replace(r"Doritos", r"Dorito")
replace = {'Doritos':'Dorito','R':'Red','N':'Natural'}
filtered_transactionData_digits['BRAND'] = filtered_transactionData_digits['BRAND'].replace(replace,regex=True)

# Then extracting the brand names again to check the result
filtered_transactionData_digits_freqbybrand_final = pd.DataFrame({'FREQ':filtered_transactionData_digits.groupby('BRAND').size().sort_values(ascending=False)}).reset_index()

### Examining customer data ###
statistics2 = customerData.describe()
length = customerData.apply(lambda x :len(x))
Mode = customerData.mode().iloc[0]
clas = customerData.dtypes
customerData_freq = pd.DataFrame({'FREQ':customerData.groupby('LIFESTAGE').size().sort_values(ascending=False)}).reset_index()
customerData_freq2 = pd.DataFrame({'FREQ':customerData.groupby('PREMIUM_CUSTOMER').size().sort_values(ascending=False)}).reset_index()

# Merge both customer & transaction data
merged_data = pd.merge(filtered_transactionData_digits, customerData, how = 'inner')

# Check for any NANs in the merged data 
null_df = merged_data.isnull()
null_check = null_df.any()
## NOTE: There're no NANs in the data in any column in the merged data 

# saving the dataframe as a csv file
# merged_data = merged_data.drop([77827,77828],axis=0)
merged_data.to_csv('merged_data.csv')

#calculating total sales by LIFESTAGE and PREMIUM_CUSTOMER
merged_LP_TotSales = merged_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).agg({'TOT_SALES': 'sum'}).reset_index()

#plot total sales by LIFESTAGE and PREMIUM_CUSTOMER
p = sns.catplot(data=merged_LP_TotSales, kind='bar', x='LIFESTAGE', y='TOT_SALES', hue='PREMIUM_CUSTOMER')
p.set(xlabel='Lifestage', ylabel='Premium customer flag', title='Proportion of sales')
p.set_xticklabels(rotation=90)
plt.show()
## NOTE: Higher """Total Sales"""  are coming mainly from Budget - older families, Mainstream - young
## singles/couples, and Mainstream - retirees

# Lets see if the higher sales are due to there being more customers who buy chips
# OR higher total sales per each transaction
customers = merged_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).agg({'LYLTY_CARD_NBR': 'nunique'}).reset_index().sort_values(by='LYLTY_CARD_NBR', ascending=False)
customers.describe()

# Plot total sales by LIFESTAGE and PREMIUM_CUSTOMER Freq
p = sns.catplot(data=customers, kind='bar', x='LIFESTAGE', y='LYLTY_CARD_NBR', hue='PREMIUM_CUSTOMER')
p.set(xlabel='Lifestage', ylabel='Premium customer flag', title='Proportion of customers')
p.set_xticklabels(rotation=90)
plt.show()
## NOTE: Most """Customers""" are coming from Mainstream - young
## singles/couples and Riterees

# Calculating """Average unit per customers""" by LIFESTAGE and PREMIUM_CUSTOMER
avg_unit_customers = merged_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).agg({'PROD_QTY': 'sum'}).reset_index().sort_values(by='PROD_QTY', ascending=False)
avg_unit_customers['Average_Qty_Customers'] = avg_unit_customers.PROD_QTY/customers.LYLTY_CARD_NBR

# Plot """Average unit per customers""" by LIFESTAGE and PREMIUM_CUSTOMER Freq
p = sns.catplot(data=avg_unit_customers, kind='bar', x='LIFESTAGE', y='Average_Qty_Customers', hue='PREMIUM_CUSTOMER')
p.set(xlabel='Lifestage', ylabel='Premium customer flag', title='Average unit per customers')
p.set_xticklabels(rotation=90)
plt.show()

# Calculating """Average price per unit chip""" by LIFESTAGE and PREMIUM_CUSTOMER
avg_unit_customers['Avg_Unit_Price'] = merged_LP_TotSales.TOT_SALES / avg_unit_customers.PROD_QTY
# Plot """Average unit per customers""" by LIFESTAGE and PREMIUM_CUSTOMER Freq
p = sns.catplot(data=avg_unit_customers, kind='bar', x='LIFESTAGE', y='Avg_Unit_Price', hue='PREMIUM_CUSTOMER')
p.set(xlabel='Lifestage', ylabel='Premium customer flag', title='Average unit Price')
p.set_xticklabels(rotation=90)
plt.show()

# Conduct Welch's t-Test and print the result
# Perform independent two sample t-test
group1 = avg_unit_customers[(avg_unit_customers['PREMIUM_CUSTOMER']=='Budget') & ((avg_unit_customers['LIFESTAGE']=='YOUNG SINGLES/COUPLES')|(avg_unit_customers['LIFESTAGE']=='MIDAGE SINGLES/COUPLES')) ]
group2 = avg_unit_customers[(avg_unit_customers['PREMIUM_CUSTOMER']=='Mainstream')& ((avg_unit_customers['LIFESTAGE']=='YOUNG SINGLES/COUPLES')|(avg_unit_customers['LIFESTAGE']=='MIDAGE SINGLES/COUPLES'))]
group3 = avg_unit_customers[(avg_unit_customers['PREMIUM_CUSTOMER']=='Premium')& ((avg_unit_customers['LIFESTAGE']=='YOUNG SINGLES/COUPLES')|(avg_unit_customers['LIFESTAGE']=='MIDAGE SINGLES/COUPLES'))]


ttest_ind(group1['Avg_Unit_Price'], group2['Avg_Unit_Price'])
# TtestResult(statistic=-5.9898134296923295, pvalue=0.02675865611789059, df=2.0)(=-4.850113922918966, pvalue=0.03997856099397995)
ttest_ind(group2['Avg_Unit_Price'], group3['Avg_Unit_Price'])
# TtestResult(statistic=5.014728140466416, pvalue=0.03754044934620897, df=2.0)(statistic=4.287205193593253, pvalue=0.05033416863924055)
ttest_ind(group1['Avg_Unit_Price'], group3['Avg_Unit_Price'])
# TtestResult(statistic=-0.3097031480980674, pvalue=0.7860764094558135, df=2.0)(statistic=-0.2650602979546056, pvalue=0.815781791309708)

## Notes : group 2 > 1
## Mainstream > Budget (for young and mid-age singles and couples)
## The unit price for mainstream, young and mid-age singles and couples 
## """ ARE """ significantly higher than that of 
## budget - young and midage singles and couples.
## While p-value is higher than 0.05 betw. Mainstream and Premium
## meaning they both have the equal means.

# Most Buying lifestage
merged_Mainstream = merged_data[(merged_data['PREMIUM_CUSTOMER']=='Mainstream')]
merged_premiumsum= (merged_Mainstream.groupby('LIFESTAGE')['TOT_SALES'].apply(lambda x:  x.sum()).sort_values(ascending=False)).reset_index()
  
# most buying brand
merged_Mainstream_young = merged_Mainstream[(merged_Mainstream['LIFESTAGE']=='YOUNG SINGLES/COUPLES')]
merged_Mainstream_young_freq = pd.DataFrame({'FREQ':merged_Mainstream_young.groupby('BRAND').size().sort_values(ascending=False)}).reset_index()

# most buying pack size
merged_pack_size = pd.DataFrame({'FREQ':merged_Mainstream_young.groupby('PACK_SIZE').size().sort_values(ascending=False)}).reset_index()                  





import pandas as pd 
import numpy as np 
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_rows", None, "display.max_columns", None)

# 1. Loading and exploring the data

data = pd.read_csv('online_retail.csv')
print(data.info())
print(data.head())
# print(data.columns)
# print(data.Country.unique())



# 2. Cleaning the data

#Stripping extra spaces in the description
data['Description'] = data['Description'].str.strip()

#Dropping the rows without any invoice number
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
  
# Dropping all transactions which were done on credit
data = data[~data['InvoiceNo'].str.contains('C')]



# 3. Splitting the data according to the region of transaction

# Transactions done in France
basket_France = (data[data['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
  
# Transactions done in the United Kingdom
basket_UK = (data[data['Country'] =="United Kingdom"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
  
# Transactions done in Portugal
basket_Por = (data[data['Country'] =="Portugal"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Transactions done in Sweden
basket_Sweden = (data[data['Country'] =="Sweden"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))



# 4. Hot encoding the Data

# Defining the hot encodign function to make the data suitable
# for the concerned libraries
def hot_encode(x):
	if(x <= 0):
		return 0
	elif(x >= 1):
		return 1

# Encoding datasets
basket_ecoded = basket_France.applymap(hot_encode)
basket_France = basket_ecoded

basket_encoded = basket_UK.applymap(hot_encode)
basket_UK = basket_encoded
  
basket_encoded = basket_Por.applymap(hot_encode)
basket_Por = basket_encoded
  
basket_encoded = basket_Sweden.applymap(hot_encode)
basket_Sweden = basket_encoded




# 5. Building the models and analyzing the results


# a) France------------

# Building the model
frq_items = apriori(basket_France, min_support = 0.05,use_colnames = True)
  
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric = "lift",min_threshold = 1)
rules = rules.sort_values(['confidence','lift'],ascending = [False,False])
print(rules.head(10))


# b) United Kingdom----------

# Building the model
frq_items = apriori(basket_UK, min_support = 0.05, use_colnames = True)
  
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())


# c) Portugal-----------

# Building the model
frq_items = apriori(basket_Por, min_support = 0.05, use_colnames = True)
  
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())


# d) Sweden------------

# Building the model
frq_items = apriori(basket_Sweden, min_support = 0.05, use_colnames = True)
  
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())
#!/usr/bin/env python
# coding: utf-8

# In[666]:


from pyspark.sql import SparkSession #imports SparkSession


# In[802]:


spark = SparkSession.builder.appName('House_Build_Prediction').getOrCreate() #Creates the appname 


# In[671]:


df = spark.read.csv('/users/f/desktop/House_Build_Prediction5.csv',inferSchema = True, header = True) #Reads the data 


# In[672]:


df.head(5)


# In[ ]:





# In[673]:


df.groupBy('Company_Name').count().show(5)#Shows the groups and counts. 


# In[674]:


from pyspark.ml.feature import StringIndexer #imports string indexer - We should import this because we need to give index numbers to the string values to make it understandable. 


# In[675]:


indexer = StringIndexer(inputCol = 'Company_Name', outputCol = 'Company_Name_Index')#indexing the first string feature "Job_title"


# In[711]:


indexed3 = indexer.fit(df).transform(df)


# In[712]:


indexed2.head(5)


# In[713]:


indexed3.head(5)


# In[679]:


from pyspark.ml.linalg import Vectors 


# In[680]:


from pyspark.ml.feature import VectorAssembler 


# In[714]:


indexed3.columns


# In[719]:


assembler2  = VectorAssembler(inputCols = [
 'Houses_Built',
 'Size_Each_House_m2',
 'Company_Name_Index',
 ], outputCol = 'features')


# In[720]:


assembler2


# In[721]:


output = assembler2.transform(indexed3)


# In[722]:


output.show()


# In[723]:


output.select('features','Duration_Months').show()


# In[724]:


final_data = output.select(['features','Duration_Months'])


# In[725]:


final_data.show()


# In[726]:


final_data.head()


# In[727]:


train_data,test_data = final_data.randomSplit([0.7,0.3])


# In[728]:


train_data.describe().show()


# In[785]:


test_data.describe().show()


# In[786]:


from pyspark.ml.regression import LinearRegression #imports the lineerRegression


# In[787]:


duration_linear = LinearRegression(labelCol = 'Duration_Months')


# In[788]:


trained_duration_model = duration_linear.fit(train_data)


# In[789]:


duration_results = trained_duration_model.evaluate(test_data)


# In[790]:


duration_results.rootMeanSquaredError


# In[791]:


duration_results.r2


# In[792]:


data = ([[2,140,0]])


# In[793]:


new_df = spark.createDataFrame(data)


# In[794]:


new_df.show()


# In[795]:


new_assembler = VectorAssembler(inputCols = [
 '_1',
 '_2',
 '_3',
 ], outputCol = 'features')


# In[796]:


new_data_output = new_assembler.transform(new_df)


# In[797]:


new_data_output.show()


# In[798]:


final_new_data_output = new_data_output.select('features')


# In[799]:


final_new_data_output.show()


# In[800]:


prediction_new_data = trained_duration_model.transform(final_new_data_output)


# In[801]:


prediction_new_data.show()


# In[ ]:





# In[ ]:





# In[ ]:





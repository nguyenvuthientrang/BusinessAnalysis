#!/usr/bin/env python
# coding: utf-8

# # 0. Preparing

# ## Format the cell output

# In[1]:


from IPython.core.display import HTML

HTML(
    """<style>div.output_area pre{white-space: pre;}</style>
    <script>
code_show_err=false; 
function code_toggle_err() {
 if (code_show_err){
 $('div.output_stderr').hide();
 } else {
 $('div.output_stderr').show();
 }
 code_show_err = !code_show_err
} 
$( document ).ready(code_toggle_err);
</script>
To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>."""
)


# ## Create Spark Session on top of YARN

# In[2]:


import pyspark

spark = (
    pyspark.sql.SparkSession.builder.appName("RegressionAnalysis")
    .master("yarn")
    .config("spark.executor.memory", "512m")
    .config("spark.executor.instances", "2")
    .getOrCreate()
)
spark


# ## Load data from hdfs

# In[3]:


df = spark.read.option("header", "true").csv(
    "hdfs://node-master:9000/user/hadoop/KAG_conversion_data.csv", inferSchema=True
)


# ## Quick look at the data

# In[4]:


df.printSchema()
df.show()
df.count()


# ## Create a temp view for this session

# In[5]:


df.select(
    "ad_id",
    "xyz_campaign_id",
    "fb_campaign_id",
    "age",
    "gender",
    "interest",
    "Impressions",
    "Clicks",
    "Spent",
    "Total_Conversion",
    "Approved_Conversion",
).createOrReplaceTempView("df")


# # 1. Regression Analysis
# In this section, we do some regression analysis to see how dependent variables depends on indepent variables. So far, we have the following independent variables:
# - xyz_campaign_id
# - age
# - gender
# - interest
# 
# The dependent variables are:
# - Impressions
# - Clicks
# - Total_Conversion
# - Approved_Conversion

# In[6]:


from pyspark.ml.feature import StringIndexer

stringIndexer = StringIndexer(inputCols=['age', 'gender'], outputCols=['new_age', 'new_gender'])
# Applying stringindexer object on dataframe movie title column
model  = stringIndexer.fit(df)
# Creating new dataframe with transformed values
indexed_df = model.transform(df)
# Validate the numerical title values
indexed_df.show(5)


# In[7]:


from pyspark.ml.feature import VectorAssembler

numericCols = ['xyz_campaign_id', 'new_age', 'new_gender', 'interest', 'Spent']
assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
indexed_df = assembler.transform(indexed_df)
indexed_df.show()
     


# In[8]:


train, test = indexed_df.randomSplit([0.8, 0.2], seed = 0)


# In[9]:


from pyspark.ml.regression import RandomForestRegressor
import pyspark.sql.functions as func
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline


# ## Reputation Variables

# ### Impressions

# In[10]:


rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'Impressions', seed=42)
rfevaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Impressions",metricName="r2")


paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10]).build()
crossval = CrossValidator(estimator = rf,
                      estimatorParamMaps = paramGrid,
                      evaluator = rfevaluator,
                      numFolds = 5)
cvModel = crossval.fit(train)
predictions = cvModel.transform(test)


# In[11]:


rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'Impressions', numTrees=10, maxDepth=10, seed=42)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))


# In[12]:


predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))
r2 = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Impressions",metricName="r2")
print("R Squared (R2) on test data = %g" % r2.evaluate(predictions))


# In[13]:


rfModel.featureImportances


# In[14]:


list(zip(numericCols, rfModel.featureImportances))


# We can clearly see that Spent have the most effect on Impressions.

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
importance = pd.DataFrame(list(zip(numericCols, rfModel.featureImportances)))
importance = importance.sort_values([1], ascending=True).reset_index(drop=True)

height = importance[1]
bars = importance[0]
y_pos = np.arange(len(bars))
 
# Create horizontal bars
plt.barh(y_pos, height, color='#bb2649')
 
# Create names on the x-axis
plt.yticks(y_pos, bars)

# Reorder it based on the values
my_range=importance[0]
 
# The horizontal plot is made using the hline function
# plt.hlines(y=my_range, xmin=0, xmax=importance[1], color='#bb2649', linewidth='4')
plt.plot(importance[1], my_range, "o", color='black')

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


# ### Clicks

# In[16]:


rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'Clicks', numTrees=50, maxDepth=20, seed=42)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))


# In[17]:


predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))
r2 = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Clicks",metricName="r2")
print("R Squared (R2) on test data = %g" % r2.evaluate(predictions))


# In[18]:


list(zip(numericCols, rfModel.featureImportances))


# In[19]:


import numpy as np
importance = pd.DataFrame(list(zip(numericCols, rfModel.featureImportances)))
importance = importance.sort_values([1], ascending=True).reset_index(drop=True)

height = importance[1]
bars = importance[0]
y_pos = np.arange(len(bars))
 
# Create horizontal bars
plt.barh(y_pos, height, color='#bb2649')
 
# Create names on the x-axis
plt.yticks(y_pos, bars)

# Reorder it based on the values
my_range=importance[0]
 
# The horizontal plot is made using the hline function
# plt.hlines(y=my_range, xmin=0, xmax=importance[1], color='#bb2649', linewidth='4')
plt.plot(importance[1], my_range, "o", color='black')

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


# We can clearly see that Spent have the most effect on Clicks.

# ## Sales Conversion

# ### Total Conversion

# In[20]:


rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'Total_Conversion', numTrees=10, maxDepth=10, seed=42)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))


# In[21]:


predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))
r2 = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Total_Conversion",metricName="r2")
print("R Squared (R2) on test data = %g" % r2.evaluate(predictions))


# In[22]:


rfModel.featureImportances


# In[23]:


list(zip(numericCols, rfModel.featureImportances))


# In[24]:


import numpy as np
importance = pd.DataFrame(list(zip(numericCols, rfModel.featureImportances)))
importance = importance.sort_values([1], ascending=True).reset_index(drop=True)

height = importance[1]
bars = importance[0]
y_pos = np.arange(len(bars))
 
# Create horizontal bars
plt.barh(y_pos, height, color='#bb2649')
 
# Create names on the x-axis
plt.yticks(y_pos, bars)

# Reorder it based on the values
my_range=importance[0]
 
# The horizontal plot is made using the hline function
# plt.hlines(y=my_range, xmin=0, xmax=importance[1], color='#bb2649', linewidth='4')
plt.plot(importance[1], my_range, "o", color='black')

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


# ### Approved Conversion

# In[25]:


rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'Approved_Conversion', numTrees=20, maxDepth=20, seed=42)
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))


# In[26]:


predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))
r2 = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Approved_Conversion",metricName="r2")
print("R Squared (R2) on test data = %g" % r2.evaluate(predictions))


# In[27]:


rfModel.featureImportances


# In[28]:


list(zip(numericCols, rfModel.featureImportances))


# ## Algorithm 1: Personalized Spent for each group of customers
# As stated above that among independent variables, Spent dominates the effects on Clicks and Impressions. Thus, in this section, we model the usecase as follow:
# - Given a range of budget, find the point that can balance between maximize the Clicks and minimize the
# 
# ## Train a function to map the setting spaces to clicks

# In[29]:


from pyspark.ml.regression import RandomForestRegressor
import pyspark.sql.functions as func
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# In[30]:


from pyspark.ml.feature import StringIndexer

stringIndexer = StringIndexer(inputCols=['age', 'gender'], outputCols=['new_age', 'new_gender'])
# Applying stringindexer object on dataframe movie title column
model  = stringIndexer.fit(df)
# Creating new dataframe with transformed values
indexed_df = model.transform(df)
# Validate the numerical title values
indexed_df.show(5)


# In[31]:


from pyspark.ml.feature import VectorAssembler

numericCols = ['xyz_campaign_id', 'new_age', 'new_gender', 'interest', 'Spent']
assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
indexed_df = assembler.transform(indexed_df)
indexed_df.show()


# In[32]:


train, val = indexed_df.randomSplit([0.8, 0.2], seed = 0)


# In[33]:


rf = RandomForestRegressor(featuresCol = 'features', labelCol = 'Clicks',  numTrees=50, maxDepth=20, seed=42)
rfModel = rf.fit(train)
predictions = rfModel.transform(val)
predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))


# In[34]:


predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))
r2 = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Clicks",metricName="r2")
print("R Squared (R2) on test data = %g" % r2.evaluate(predictions))


# In[35]:


def map_click(setting, budget, predictor):
    test_pd = pd.DataFrame({
      'xyz_campaign_id': [setting['xyz_campaign_id']]*len(budget),
    
      'age': [setting['age']]*len(budget),
    
      'gender': [setting['gender']]*len(budget),
    
      'interest': [setting['interest']]*len(budget),
    
      'Spent': budget
      })
    
    test = spark.createDataFrame(test_pd)
    
    numericCols = ['xyz_campaign_id', 'age', 'gender', 'interest', 'Spent']
    assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
    test = assembler.transform(test)

    predictions = predictor.transform(test)
    predictions = predictions.withColumn("prediction", func.round(predictions["prediction"], 0))

    sns.set_style("white")
    sns.lineplot(data=predictions.toPandas(), x="Spent", y="prediction", color="#bb2649")

    test.createOrReplaceTempView("test")

    my_window = Window.partitionBy().orderBy("Spent")

    predictions = predictions.withColumn("prev_prediction", F.lag(predictions.prediction).over(my_window))
    predictions = predictions.withColumn("diff", F.when(F.isnull(predictions.prediction - predictions.prev_prediction), 0)
                                  .otherwise(predictions.prediction - predictions.prev_prediction))
    
    predictions.createOrReplaceTempView("predictions")
    res = spark.sql("SELECT Spent, prediction, diff FROM predictions WHERE diff = (SELECT MAX(diff) FROM predictions)")
    res.show()

    return res.collect()[0]['Spent']



# In[36]:


import seaborn as sns
setting = {'xyz_campaign_id': 916.0,
           'age': 0.0,
           'gender': 0.0,
           'interest': 15}
budget = [i for i in range(100,400,1)]
map_click(setting, budget, rfModel)


# ## Feature Engineering
# ### Adding conversions features
# From the sanity checks and the optimization problem, we define 4 additional feature as follow:
# - Spent per Click: Cost per Click (CPC)
# - Click per Impression: Click through rate (CTR)
# - Total/Approved Conversion per Impression ((A)CR)
# - Approved Conversion per Total Conversion (AR)
# 
# We also want to include the cost per conversion, however some spent = 0 yet having positive conversion, so we did not include this measure.

# In[37]:


df = df.withColumn('CPC', df['Spent'] / df['Clicks'])
df = df.withColumn('CTR', df['Clicks']*100 / df['Impressions'])
df = df.withColumn('TCR', df['Total_Conversion']*100 / df['Impressions'])
df = df.withColumn('ACR', df['Approved_Conversion']*100 / df['Impressions'])
df = df.withColumn('AR', df['Approved_Conversion']*100 / df['Total_Conversion'])


# In[38]:


df=df.na.fill(value=10,subset=["CPC"])


# In[39]:


df.createOrReplaceTempView("df")


# In[40]:


df.show(10)


# ### EDA for addiontal features

# In[41]:


labels = []
i = 0
fig, axs = plt.subplots(1, 5, figsize=(50, 10))
for feat in ['CPC', 'CTR', 'TCR', 'ACR', 'AR']:
  x = spark.sql("SELECT {0} \
            FROM df".format(feat)).toPandas()
  labels.append(feat)
  sns.distplot(a=x, hist=False, kde=True, norm_hist=True, ax=axs[i], color='#BB2649')
  i += 1
     


# In[42]:


labels = []
i = 0
fig, axs = plt.subplots(1, 5, figsize=(50, 10))
for feat in ['CPC', 'CTR', 'TCR', 'ACR', 'AR']:
  for cp in ['916', '936', '1178']:
    x = spark.sql("SELECT {0} \
              FROM df \
              WHERE xyz_campaign_id = '{1}'".format(feat, cp)).toPandas()
    sns.distplot(a=x, hist=False, kde=True, ax=axs[i])
  i += 1
     


# In[43]:


labels = []
i = 0
fig, axs = plt.subplots(1, 5, figsize=(50, 10))
for feat in ['CPC', 'CTR', 'TCR', 'ACR', 'AR']:
  for gender in ['M', 'F']:
    x = spark.sql("SELECT {0} \
              FROM df \
              WHERE GENDER = '{1}'".format(feat, gender)).toPandas()
    sns.distplot(a=x, hist=False, kde=True, ax=axs[i])
  i += 1


# In[44]:


labels = []
i = 0
fig, axs = plt.subplots(1, 5, figsize=(50, 10))
for feat in ['CPC', 'CTR', 'TCR', 'ACR', 'AR']:
  for age in ['30-34', '35-39', '40-44', '45-49']:
    x = spark.sql("SELECT {0} \
              FROM df \
              WHERE AGE = '{1}'".format(feat, age)).toPandas()
    sns.distplot(a=x, hist=False, kde=True, ax=axs[i])
  i += 1
     


# In[45]:


labels = []
i = 0
fig, axs = plt.subplots(1, 5, figsize=(50, 10))
for feat in ['CPC', 'CTR', 'TCR', 'ACR', 'AR']:
  labels.append(feat)
  sns.regplot(y=df.toPandas()[feat], x=df.toPandas()["interest"], ax=axs[i],fit_reg=False)
  i += 1
     


# In[46]:


g=sns.PairGrid(df.toPandas(), vars=['CPC', 'CTR', 'TCR', 'ACR', 'AR'], diag_sharey=False, corner=True)
g.map_lower(sns.kdeplot, color="#BB2649", fill=True)
g.map_diag(sns.histplot, color="#BB2649", kde=True)


# In[47]:


sns.heatmap(df.select(['CPC', 'CTR', 'TCR', 'ACR', 'AR']).toPandas().corr())


# ## Algorithm 2: Classifying potential customers

# In[48]:


df_avg = spark.sql("SELECT xyz_campaign_id, fb_campaign_id, age, gender, interest, \
 AVG(CPC) CPC, AVG(CTR) CTR, AVG(TCR) TCR, AVG(ACR) ACR, AVG(AR) AR FROM df GROUP BY xyz_campaign_id, fb_campaign_id, age, gender, interest")


# In[49]:


df_avg.show()


# ### Clustering based on CTR, CPC
# We can see that there are 2 modes in the CTR, hence we first bring this to an unsupervised task. We try to cluster the customers into 2 clusters, using (CPC, CTR) which are variables related to sales/revenue.

# In[50]:


from pyspark.ml.feature import VectorAssembler
assemble=VectorAssembler(inputCols=['CPC', 'CTR'], outputCol='features')
assembled_data=assemble.transform(df)


# In[51]:


assembled_data.show()


# In[52]:


from pyspark.ml.feature import StandardScaler
scale=StandardScaler(inputCol='features',outputCol='standardized')
data_scale=scale.fit(assembled_data)
data_scale_output=data_scale.transform(assembled_data)
data_scale_output.show()


# In[53]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')
    
KMeans_algo=KMeans(featuresCol='standardized', k=2)

KMeans_fit=KMeans_algo.fit(data_scale_output)

output=KMeans_fit.transform(data_scale_output)

score=evaluator.evaluate(output)


# In[54]:


output.show()


# In[55]:


sns.displot(data=output.select('CTR', 'prediction').toPandas(), x='CTR', kde=True, color ='#000000')


# In[56]:


sns.displot(data=output.select('CTR', 'prediction').toPandas(), x='CTR', kde=True, hue='prediction', palette =['#BB2649','#2B526B'])


# ### Clustering based on AR
# We can see that there are 3 modes in the AR, hence we first bring this to an unsupervised task. We try to cluster the customers into 3 clusters, using (AR) which are variables related to sales/revenue.

# In[57]:


df_ar_drop = df.na.drop(subset=["AR"])


# In[58]:


from pyspark.ml.feature import VectorAssembler
assemble=VectorAssembler(inputCols=['AR'], outputCol='features')
assembled_data=assemble.transform(df_ar_drop)


# In[59]:


assembled_data.show()


# In[60]:


from pyspark.ml.feature import StandardScaler
scale=StandardScaler(inputCol='features',outputCol='standardized')
data_scale=scale.fit(assembled_data)
data_scale_output=data_scale.transform(assembled_data)
data_scale_output.show()


# In[61]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')
    
KMeans_algo=KMeans(featuresCol='standardized', k=3)

KMeans_fit=KMeans_algo.fit(data_scale_output)

output=KMeans_fit.transform(data_scale_output)

score=evaluator.evaluate(output)


# In[62]:


output.show()


# In[63]:


sns.displot(data=output.select('AR').toPandas(), x='AR', kde=True, color ='#000000')


# In[64]:


sns.displot(data=output.select('AR', 'prediction').toPandas(), x='AR', kde=True, hue='prediction', palette =['#BB2649','#FFE4EA','#2B526B'])


# ## Hypothesis Testing On Conversion Rate

# In[65]:


df.createOrReplaceTempView("df")


# In[66]:


df.show()


# In[67]:


spark.sql("SELECT xyz_campaign_id, avg(TCR), AVG(ACR) FROM df group by xyz_campaign_id ").show()


# In[68]:


def uni_eda(feat, ftype, scale=None, prob=False, agg='count', palette=None, font_scale=2):
  spark.sql("SELECT COUNT(DISTINCT {}) FROM df".format(feat)).show()
  spark.sql("SELECT DISTINCT {} FROM df".format(feat)).show()

  spark.sql("SELECT {0},count(*) \
            FROM df \
            GROUP BY {0}".format(feat)).show()

  spark.sql("SELECT {0}, {1}(TCR) TCR, \
            {1}(ACR) ACR \
            FROM df \
            GROUP BY {0}".format(feat, agg)).show()
  sns.set(font_scale = font_scale)
  sns.set_style('whitegrid')
  sns.barplot(
        x=feat, 
        y='count', 
        data=spark.sql("SELECT {0},count(*) count \
            FROM df \
            GROUP BY {0}".format(feat)).toPandas(),
        palette=palette
    )
  plt.savefig("{}_uni.png".format(feat))
  
  # sns.set(font_scale = 2)
  fig, axes=plt.subplots(nrows=1, ncols=2, figsize=(50,10))
  i=0
  for col in ['TCR', 'ACR']:
    sns.set_style('whitegrid')
    sns.barplot(
        x=feat, 
        y=col, 
        data=spark.sql("SELECT {0}, {1}(TCR) TCR, \
                {1}(ACR) ACR \
                FROM df \
                GROUP BY {0}".format(feat, agg)).toPandas(),
                palette=palette,
        ax=axes[i]
    )    
    if scale:
      axes[i].set_yscale(scale)
    # axes[i].set_ylabel(col, fontsize=40)
    # axes[i].set_xlabel(feat, fontsize=40)
    # axes[i].set_yticklabels(fontsize=40)
    # axes[i].set_xticklabels(fontsize=40)
    i += 1    
  sns.despine()
  plt.savefig("{}_conversion_rate.png".format(feat))


# In[69]:


uni_eda("age", "cat", agg='AVG', palette=['#BB2649','#FFE4EA', '#91AEB9' ,'#2B526B'])


# In[70]:


from pyspark.sql.functions import mean, variance, count
from scipy.stats import t
from math import sqrt
from pyspark.sql.functions import mean, variance
from math import sqrt

def one_sided_test(df, feat, vals, test_col, sample_rate=0.2):
  samples_1 = spark.sql("SELECT {0}, * FROM df WHERE {0} = '{1}'".format(feat, vals[0]))
  samples_2 = spark.sql("SELECT {0}, * FROM df WHERE {0} = '{1}'".format(feat, vals[1]))

  mean_1, var_1, count_1 = tuple(samples_1.select(mean(test_col), variance(test_col), count(test_col)).collect()[0])
  mean_2, var_2, count_2 = tuple(samples_2.select(mean(test_col), variance(test_col), count(test_col)).collect()[0])

  numerator = mean_1 - mean_2
  denominator = sqrt((var_1 / count_1) + (var_2 / count_2))
  t_test_statistic = numerator / denominator

  degrees_of_freedom = count_1 + count_2 - 2
  p_value = 1- t.cdf(t_test_statistic, df = degrees_of_freedom)

  print("Probability that the two have the same distribution:", p_value > 0.05)
  return p_value


# In[71]:


one_sided_test(df, 'age', ['30-34', '35-39'], 'TCR')


# In[72]:


one_sided_test(df, 'age', ['30-34', '45-49'], 'TCR')


# In[73]:


one_sided_test(df, 'age', ['30-34', '40-44'], 'TCR')


# In[74]:


one_sided_test(df, 'age', ['30-34', '35-39'], 'ACR')
one_sided_test(df, 'age', ['30-34', '45-49'], 'ACR')
one_sided_test(df, 'age', ['30-34', '40-44'], 'TCR')


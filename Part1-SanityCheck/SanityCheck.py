#!/usr/bin/env python
# coding: utf-8

# # Sanity Check

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
    pyspark.sql.SparkSession.builder.appName("SanityCheck")
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


# # 1. Sanity Check

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


# ## Check Missing 

# In[6]:


from pyspark.sql.functions import isnan, when, count, col

df.select(
    [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
).show()


# From the query, we found that the data contains no null/nan values

# ## Figure out the features
# All the columns seems to be easy to understand from its definition, except for the column 'fb_campaign_id'. Let's figure out how they indexed this!

# ### fb_campaign_id

# In[7]:


spark.sql("SELECT COUNT(DISTINCT fb_campaign_id) FROM df").show()


# In[8]:


spark.sql(
    "SELECT xyz_campaign_id, fb_campaign_id, gender, age, interest, Spent FROM df WHERE fb_campaign_id IN \
            (SELECT fb_campaign_id FROM df GROUP BY fb_campaign_id HAVING COUNT(fb_campaign_id) > 1)"
).show()


# We can see that, with similar fb_campaign_id, the xyz_id, gender, age, interest are the same, only the Spent is different. Hence, fb_cp_id might be the same campaign ad with the same group of customers but with different budget.

# ## Check outliers

# ### Spent

# In[9]:


import seaborn as sns

sns.boxplot(x=df.toPandas()["Spent"])
df.sort(col("Spent").desc()).show()


# ### Impressions

# In[10]:


sns.boxplot(x=df.toPandas()["Impressions"])
df.sort(col("Impressions").desc()).show()


# ### Clicks

# In[11]:


sns.boxplot(x=df.toPandas()["Clicks"])
df.sort(col("Clicks").desc()).show()


# ### Total_Conversion

# In[12]:


sns.boxplot(x=df.toPandas()["Total_Conversion"])
df.sort(col("Total_Conversion").desc()).show()


# ### Approved_Conversion

# In[13]:


sns.boxplot(x=df.toPandas()["Approved_Conversion"])
df.sort(col("Approved_Conversion").desc()).show()


# ## Check for constraints violation
# We test some constraints that can be inferred from the definition of columns:
# - If Spent = 0 then Impressions, Clicks, Total_Conversion, Approved_Conversion = 0
# - Approved_Conversion < Total Conversions < Clicks < Impressions

# In[14]:


print("Spent=0:")
spark.sql(
    "SELECT count(*) \
            FROM df \
            WHERE Spent = 0 AND (Impressions > 0 OR Clicks > 0 OR Total_Conversion > 0 OR Approved_Conversion > 0)"
).show()


# There are 207 records does not satisfy the first constraints, let's see where it comes from

# In[15]:


print("Spent=0 & Impressions>0:")
spark.sql(
    "SELECT count(*) \
            FROM df \
            WHERE Spent = 0 AND (Impressions > 0)"
).show()


# In[16]:


print("Spent=0 & Clicks>0:")
spark.sql(
    "SELECT count(*) \
            FROM df \
            WHERE Spent = 0 AND (Clicks > 0)"
).show()


# In[17]:


print("Spent=0 & Total_Conversion>0:")
spark.sql(
    "SELECT count(*) \
            FROM df \
            WHERE Spent = 0 AND (Total_Conversion > 0)"
).show()


# In[18]:


print("Spent=0 & Approved_Conversion>0:")
spark.sql(
    "SELECT count(*) \
            FROM df \
            WHERE Spent = 0 AND (Approved_Conversion > 0)"
).show()


# The contraints violation mostly comes from the Impression. In practice, we thought of this as the cases that the auto-recommender system of facebook recommend the ads for users that have the same taste with the targeted users, or the friend lists of the targeted users.

# In[19]:


print("Clicks > Impressions:")
spark.sql(
    "SELECT COUNT(*) \
            FROM df \
            WHERE Clicks > Impressions"
).show()


# In[20]:


print("Total_Conversion > Clicks:")
spark.sql(
    "SELECT COUNT(*) \
            FROM df \
            WHERE Total_Conversion > Clicks"
).show()


# In[21]:


print("Approved_Conversion > Clicks:")
spark.sql(
    "SELECT COUNT(*) \
            FROM df \
            WHERE Approved_Conversion > Clicks"
).show()


# print("Total_Conversion > Impressions:")
# spark.sql("SELECT COUNT(*) \
#             FROM df \
#             WHERE Total_Conversion > Impressions").show()

# In[22]:


print("Approved_Conversion > Impressions")
spark.sql(
    "SELECT COUNT(*) \
            FROM df \
            WHERE Approved_Conversion > Impressions"
).show()


# In[23]:


print("Approved_Conversion > Total_Conversion:")
spark.sql(
    "SELECT * \
            FROM df \
            WHERE Approved_Conversion > Total_Conversion"
).show()


# Also, there are some none-clicked ads but still have conversions. We can see that not all enquiries/sales comes from the ads.

# # 2. Stop the Spark Session

# In[24]:


spark.stop()


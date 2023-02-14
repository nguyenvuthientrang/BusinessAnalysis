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


# ## Format the cell output

# In[2]:


from IPython.core.display import HTML

HTML("""<style>div.output_area pre{white-space: pre;}</style>""")


# ## Create Spark Session on top of YARN

# In[3]:


import pyspark

spark = (
    pyspark.sql.SparkSession.builder.appName("ExploratoryDataAnalysis")
    .master("yarn")
    .config("spark.executor.memory", "512m")
    .config("spark.executor.instances", "2")
    .getOrCreate()
)
spark


# ## Load data from hdfs

# In[4]:


df = spark.read.option("header", "true").csv(
    "hdfs://node-master:9000/user/hadoop/KAG_conversion_data.csv", inferSchema=True
)


# ## Quick look at the data

# In[5]:


df.printSchema()
df.show()
df.count()


# # 1. Visualization
# Only for this section, we query and transform pyspark Data Frame into Pandas Data Frame for a better visualization.
# ## Overview

# In[6]:


df.summary().show()


# In[7]:


import seaborn as sns

sns.heatmap(
    df.select(
        "Impressions", "Clicks", "Spent", "Total_Conversion", "Approved_Conversion"
    )
    .toPandas()
    .corr()
)


# It is clearly see that Spent, Clicks, Impressions are highly correlated with each other, the same goes for Total_Conversion and Approved Conversion

# ## Create a temp view for this session

# In[8]:


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


# ## Functions

# In[9]:


def uni_eda(
    feat, ftype, scale=None, prob=False, agg="count", palette=None, font_scale=1
):
    spark.sql("SELECT COUNT(DISTINCT {}) FROM df".format(feat)).show()
    spark.sql("SELECT DISTINCT {} FROM df".format(feat)).show()

    spark.sql(
        "SELECT {0},count(*) \
            FROM df \
            GROUP BY {0}".format(
            feat
        )
    ).show()

    spark.sql(
        "SELECT {0}, {1}(Impressions) Impressions, \
            {1}(Clicks) Clicks, \
            {1}(Spent) Spent, \
            {1}(Total_conversion) Total_conversion, \
            {1}(Approved_conversion) Approved_conversion \
            FROM df \
            GROUP BY {0}".format(
            feat, agg
        )
    ).show()
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")
    sns.barplot(
        x=feat,
        y="count",
        data=spark.sql(
            "SELECT {0},count(*) count \
            FROM df \
            GROUP BY {0}".format(
                feat
            )
        ).toPandas(),
        palette=palette,
    )
    plt.savefig("{}_uni.png".format(feat))

    # sns.set(font_scale = 2)
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(50, 10))
    i = 0
    for col in [
        "Impressions",
        "Clicks",
        "Spent",
        "Total_Conversion",
        "Approved_Conversion",
    ]:
        sns.set_style("whitegrid")
        sns.barplot(
            x=feat,
            y=col,
            data=spark.sql(
                "SELECT {0}, {1}(Impressions) Impressions, \
                {1}(Clicks) Clicks, \
                {1}(Spent) Spent, \
                {1}(Total_Conversion) Total_Conversion, \
                {1}(Approved_Conversion) Approved_Conversion \
                FROM df \
                GROUP BY {0}".format(
                    feat, agg
                )
            ).toPandas(),
            ax=axes[i],
        )
        if scale:
            axes[i].set_yscale(scale)
        i += 1
    sns.despine()


def var2(feat1, feat2, scale=None, agg="avg"):
    spark.sql(
        "SELECT {0}, {1}, \
                    {2}(Impressions) Impressions, \
                    {2}(Clicks) Clicks, \
                    {2}(Spent) Spent, \
                    {2}(Total_Conversion) Total_Conversion, \
                    {2}(Approved_Conversion) Approved_Conversion \
                    FROM df \
                    GROUP BY {0}, {1}".format(
            feat1, feat2, agg
        )
    ).show()
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(100, 20))
    i = 0
    for col in [
        "Impressions",
        "Clicks",
        "Spent",
        "Total_Conversion",
        "Approved_Conversion",
    ]:
        sns.barplot(
            x=feat1,
            y=col,
            hue=feat2,
            data=spark.sql(
                "SELECT {0}, {1}, \
                        {2}(Impressions) Impressions, \
                        {2}(Clicks) Clicks, \
                        {2}(Spent) Spent, \
                        {2}(Total_Conversion) Total_Conversion, \
                        {2}(Approved_Conversion) Approved_Conversion \
                        FROM df \
                        GROUP BY {0}, {1}".format(
                    feat1, feat2, agg
                )
            ).toPandas(),
            ax=axes[i],
        )
        if scale:
            axes[i].set_yscale("log")
        axes[i].set_ylabel(col, fontsize=40)
        axes[i].set_xlabel(feat1, fontsize=40)
        axes[i].legend(fontsize=40)
        i += 1


# ## Categorical Variables

# ### Campaign

# In[10]:


import matplotlib.pyplot as plt

uni_eda("xyz_campaign_id", "cat", agg='count', palette=['#BB2649','#FFE4EA','#2B526B'])


# In[11]:


var2('xyz_campaign_id', 'age', scale='log')


# In[12]:


var2('xyz_campaign_id', 'gender', scale='log')


# In[13]:


var2('xyz_campaign_id', 'interest', scale='log')


# ### Age

# In[14]:


uni_eda("age", "cat", agg='count', palette=['#BB2649','#FFE4EA', '#91AEB9' ,'#2B526B'])


# In[15]:


uni_eda("age", "cat", agg='avg')


# The age 45-49 is spent on a lot, got a lot of impressions and clicks but less enquirement and sales.
# In contrast, the group 30-34 have highest conversions though spent amount is less than 45-49

# In[16]:


var2('age', 'xyz_campaign_id', scale='log')


# In[17]:


var2('age', 'gender', scale='log')


# In[18]:


uni_eda("gender", "cat", agg='count', palette=['#BB2649','#2B526B'])


# ### Gender

# In[19]:


uni_eda("gender", "cat", agg='count', palette=['#BB2649','#2B526B'])


# Although Female is invested in more, and they have higher impressions/clicks but Male tends to buy the product after clicks more.

# In[20]:


uni_eda("gender", "cat", agg='avg')


# ### Interest

# In[21]:


uni_eda("interest", "cat", agg='count', palette=sns.diverging_palette(203.5, 9.5))


# It is worthnoting that the interest 100-114 is more focused on despite that their amount is less compare to other interests.

# In[22]:


uni_eda("interest", "cat", agg='avg')


# ### Spent

# In[23]:


sns.set_style("whitegrid")
sns.displot(data=df.toPandas(), x="Spent", kde=True, rug=False, color="#BB2649")
plt.savefig("spent_uni.png")


# ## Correlogram

# In[24]:


g=sns.PairGrid(df.toPandas(), vars=['Spent', 'Impressions', 'Clicks', 'Total_Conversion', 'Approved_Conversion'], diag_sharey=False, corner=True)
g.map_lower(sns.kdeplot, color="#BB2649", fill=True)
g.map_diag(sns.histplot, color="#BB2649", kde=True)
plt.savefig("correl.png")


# In[25]:


spark.stop()


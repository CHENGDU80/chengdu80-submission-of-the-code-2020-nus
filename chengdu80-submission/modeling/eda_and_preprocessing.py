import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

trans_df = pd.read_csv("../../data/raw/transaction_data.tsv", sep="\t")
trans_df.head()

trans_df['TICKER'].value_counts()

ir_df = pd.read_csv("../../data/raw/industrial_relation.tsv", sep="\t", encoding= 'unicode_escape')
ir_df.head()

segment_df = pd.read_csv("../../data/processed/industrial_relation_sector.tsv", sep="\t", encoding= 'unicode_escape')

segment_df.head()

segment_df['Sector'].value_counts()

from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()

file_path = "../../data/raw/2012_financial_news/*/*"

rdd = sc.wholeTextFiles(file_path)

sdf = spark.createDataFrame(rdd)
sdf.createOrReplaceTempView("news")

sdf.write.parquet("../../data/raw/data.parquet")

sdf = spark.read.parquet("../../data/raw/data.parquet")

sdf.count()


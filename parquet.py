import pandas

path1 = "cyberthreat.parquet"
path2 = "data.parquet"
path3 = "RT_IOT2022.parquet"

data1 = pandas.read_csv("cyberthreat.csv")
data2 = pandas.read_csv("data.csv")
data3 = pandas.read_csv("RT_IOT2022.csv")

data1.to_parquet(path1)
data2.to_parquet(path2)
data3.to_parquet(path3)
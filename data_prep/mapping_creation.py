from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize SparkSession (if not already initialized)
spark = SparkSession.builder.appName("MappingTable").getOrCreate()

def extract_val_name_pairs(df):
    columns = df.columns
    val_cols = [col for col in columns if col.endswith('_val')]
    name_cols = [col for col in columns if col.endswith('_name')]
    val_prefixes = set([col[:-4] for col in val_cols])
    name_prefixes = set([col[:-5] for col in name_cols])
    common_prefixes = val_prefixes.intersection(name_prefixes)
    pair_dfs = []
    for prefix in common_prefixes:
        val_col = f"{prefix}_val"
        name_col = f"{prefix}_name"
        pair_df = df.select(
            F.col(val_col).alias('val'),
            F.col(name_col).alias('name')
        )
        pair_dfs.append(pair_df)
    return pair_dfs

# List of your DataFrames
dataframes = [df1, df2, df3]  # Replace with your actual DataFrames

all_pair_dfs = []
for df in dataframes:
    pair_dfs = extract_val_name_pairs(df)
    all_pair_dfs.extend(pair_dfs)

if all_pair_dfs:
    mapping_table = reduce(lambda df1, df2: df1.union(df2), all_pair_dfs)
else:
    from pyspark.sql.types import StringType, StructField, StructType
else:
    from pyspark.sql.types import StringType, StructField, StructType
    schema = StructType([
        StructField('val', StringType(), True),
        StructField('name', StringType(), True)
    ])
    mapping_table = spark.createDataFrame([], schema)

# Remove null 'val' entries
mapping_table = mapping_table.filter(F.col('val').isNotNull())

# Group by 'val' and aggregate
aggregated_mapping_table = mapping_table.groupBy('val').agg(
    F.collect_set('name').alias('names'),
    F.countDistinct('name').alias('name_count')
)

# Add 'is_duplicate' flag
aggregated_mapping_table = aggregated_mapping_table.withColumn(
    'is_duplicate',
    F.when(F.col('name_count') > 1, True).otherwise(False)
)

# Optionally select a 'name' (e.g., the first one)
aggregated_mapping_table = aggregated_mapping_table.withColumn(
    'name',
    F.expr("names[0]")
)

# Show the final mapping table
aggregated_mapping_table.select('val', 'name', 'names', 'name_count', 'is_duplicate').show(truncate=False)

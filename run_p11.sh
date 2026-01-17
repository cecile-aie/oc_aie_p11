#!/bin/bash
# Mode client
# spark-submit \
#   --master yarn \
#   --deploy-mode client \
#   --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=/usr/bin/python3 \
#   --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3 \
#   /home/hadoop/p11_app.py

# Mode cluster
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=/usr/bin/python3 \
  --conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3 \
  s3://ociae-p11/scripts/p11_app.py

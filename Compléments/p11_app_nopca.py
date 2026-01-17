import re, io, boto3
from urllib.parse import unquote

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, udf
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT

# === 🔁 Spark init
spark = SparkSession.builder.appName("P11AppRawFeatures").getOrCreate()
sc = spark.sparkContext

# === 🧠 Broadcast du modèle MobileNetV2
def build_model():
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=x)

model_weights = build_model().get_weights()
bc_model_weights = sc.broadcast(model_weights)

# === Chargement image S3
def load_image_from_s3(s3_path):
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3 = boto3.client("s3")
    image_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return load_img(io.BytesIO(image_bytes), target_size=(224, 224))

# === Feature extractor UDF
model_instance = None
def extract_features(path):
    global model_instance
    if model_instance is None:
        model_instance = build_model()
        model_instance.set_weights(bc_model_weights.value)
    try:
        img = load_image_from_s3(path)
        arr = img_to_array(img)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        features = model_instance.predict(arr, verbose=0)
        return Vectors.dense(features.flatten())
    except Exception:
        return Vectors.dense([0.0] * 1280)

extract_udf = udf(extract_features, VectorUDT())
unquote_udf = udf(lambda s: unquote(s))

# === 🔁 Traitement pour chaque jeu : train, test
for set_name in ["train", "test"]:
    print(f"=== Traitement du set : {set_name} ===")

    PATH_Data = f"s3://ociae-p11/images/{set_name}"
    PATH_Result = f"s3://ociae-p11/results_{set_name}_raw/"

    # 📥 Chargement des fichiers image
    df = spark.read.format("image") \
        .option("recursiveFileLookup", "true") \
        .load(PATH_Data)

    df = df.select(
        unquote_udf(input_file_name()).alias("path"),
        unquote_udf(regexp_extract(input_file_name(), rf"images/{set_name}/([^/]+)/[^/]+$", 1)).alias("label")
    )

    # 🧠 Features
    df_feat = df.withColumn("features", extract_udf("path")).cache()

    # 🧪 Normalisation StandardScaler (pas de PCA)
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_feat)
    df_scaled = scaler_model.transform(df_feat).cache()

    # 💾 Enregistrement brut (non réduit)
    df_scaled.select("path", "label", "scaled_features") \
             .write.mode("overwrite") \
             .parquet(PATH_Result)

    print(f"✅ Sauvegarde terminée dans : {PATH_Result}")

spark.stop()

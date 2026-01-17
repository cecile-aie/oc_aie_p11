import tensorflow as tf
import re

# Pour EMR : on travaille directement avec S3
PATH_Data = 's3://ociae-p11/images/test'
PATH_Result = 's3://ociae-p11/results'
# Extrait 'train' ou 'test' depuis PATH_Data (ex: 's3://bucket/images/train')
SET_NAME = re.search(r'images/([^/]+)', PATH_Data).group(1)


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("P11App").getOrCreate()

from pyspark.sql.functions import input_file_name, regexp_extract, udf
from urllib.parse import unquote

# UDF pour décoder les chemins encodés (ex: %20 → espace)
unquote_udf = udf(lambda s: unquote(s))

df = spark.read.format("image") \
    .option("recursiveFileLookup", "true") \
    .load(PATH_Data)

# Remplacement du chemin encodé par sa version décodée
from pyspark.sql.functions import lit

df = df.select(
    unquote_udf(input_file_name()).alias("path"),
    unquote_udf(regexp_extract(input_file_name(), rf"images/{SET_NAME}/([^/]+)/[^/]+$", 1)).alias("label")
)

df.select("path", "label").show(10, truncate=False)

from pyspark.sql import functions as F
# Sélection de lignes au hasard
df_sample = df #.orderBy(F.rand()).limit(1000)

# ⚠️ Mise en cache
# définition d'un partitionnement .repartition()
df_sample = df_sample.repartition(40).cache()


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ⚠️ Pas de Sequential → ça cause souvent des erreurs silencieuses
def build_mobilenetv2_model_imagenet():
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    return model


model = build_mobilenetv2_model_imagenet()
weights = model.get_weights()
# 📡 Diffusion les poids avec Spark
bc_model_weights = spark.sparkContext.broadcast(weights)


def load_image_from_s3(s3_path):
    import boto3
    import io
    from tensorflow.keras.preprocessing.image import load_img

    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()

    return load_img(io.BytesIO(image_bytes), target_size=(224, 224))


def log_s3_error(path, message):
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket="ociae-p11",  # adapte ici si tu changes de bucket
            Key=f"logs/errors/{path.split('/')[-1]}.log",
            Body=message.encode("utf-8")
        )
    except Exception as e:
        pass  # ignore les erreurs de log pour ne pas casser les workers


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np

# Reconstruction locale à partir des poids broadcastés
def build_model_from_weights(weights):
    model = build_mobilenetv2_model_imagenet()
    model.set_weights(weights)
    return model

# UDF avec initialisation une seule fois par worker
model_instance = None  # stocké localement sur le worker

def extract_features(path):
    global model_instance
    try:
        import numpy as np
        from tensorflow.keras.preprocessing.image import img_to_array
        import tensorflow as tf
        from pyspark.ml.linalg import Vectors

        if model_instance is None:
            model_instance = build_model_from_weights(bc_model_weights.value)

        img = load_image_from_s3(path)
        img_array = img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        features = model_instance.predict(img_array, verbose=0)

        return Vectors.dense(features.flatten())

    except Exception as e:
        log_s3_error(path, f"❌ Erreur image : {str(e)}")
        return Vectors.dense([0.0] * 1280)


from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT
# 🔁 Intégration dans Spark avec UDF et ⚠️ Mise en cache
extract_udf = udf(extract_features, VectorUDT())
df_features = df_sample.withColumn("features", extract_udf("path")).cache()


from pyspark.ml.feature import StandardScaler

# Normalisation des features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(df_features)
df_scaled = scaler_model.transform(df_features)
# ⚠️ Mise en cache
df_scaled.cache()


from pyspark.ml.feature import PCA
# PCA.1 Calcul de la PCA avec le même nombre de dimensions que train
num_features = len(df_scaled.select("scaled_features").first()["scaled_features"])
pca = PCA(k=412, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(df_scaled)


df_pca = pca_model.transform(df_scaled)

# ⚠️ Mise en cache
df_pca.cache()


# Sauvegarde du résultat final (3 colonnes) en Parquet
df_pca.select("path", "label", "pca_features") \
      .write.mode("overwrite") \
      .parquet(PATH_Result)


# Lire pour tester que l'écriture a bien fonctionné
df_parquet = spark.read.parquet(PATH_Result)
df_parquet.show(5, truncate=False)


spark.stop()
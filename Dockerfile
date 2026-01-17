FROM python:3.10-slim

# Dépendances système
RUN apt-get update && apt-get install -y procps --no-install-recommends \
    openjdk-17-jdk \
    curl \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Environnement Java & Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_VERSION=3.5.0
ENV SPARK_PACKAGE=spark-3.5.0-bin-hadoop3

# Installation de Spark depuis archive stable
RUN curl -fSL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz -o spark.tgz && \
    tar -xzf spark.tgz -C /opt && \
    mv /opt/${SPARK_PACKAGE} /opt/spark && \
    rm spark.tgz

# Variables pour PySpark
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Dossier pour Spark event logs
RUN mkdir -p /spark-events

# Désactiver les logs GPU de TensorFlow (niveau WARNING uniquement)
ENV TF_CPP_MIN_LOG_LEVEL=2

# Installer les bibliothèques Python (CPU only)
RUN pip install --no-cache-dir \
    jupyterlab \
    pandas \
    numpy \
    pillow \
    tensorflow-cpu==2.17.0 \
    pyarrow \
    pyspark==3.5.0 \
    scikit-learn \
    matplotlib \
    seaborn \
    s3fs \
    scipy

# Dossier de travail
WORKDIR /workspace
VOLUME /workspace

# Ports exposés
EXPOSE 8888   
EXPOSE 4040    
EXPOSE 18080   

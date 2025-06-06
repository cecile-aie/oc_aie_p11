# oc_aie_p11
Projet OC parcours ingénieur IA - Réalisez un traitement big data dans le cloud

🧠 Résumé des attendus du projet
🎯 Objectif
Mettre en place une chaîne de traitement d’images de fruits dans un environnement Big Data, pour poser les bases d’une future architecture de reconnaissance d’images à grande échelle, sans réentraîner de modèle 

📌 Ce qu'il faut faire concrètement
✅ 1. Compléter le script PySpark
On pars d’un notebook existant et tu dois le compléter avec deux éléments clés :

✅ Diffusion des poids du modèle MobileNet sur le cluster (broadcast TensorFlow)

✅ Réduction de dimension via une PCA distribuée en PySpark

Il n'est pas d'entraîner MobileNet, mais il faut :

Charger un modèle pré-entraîné (ex : MobileNetV2(weights="imagenet")).

Utiliser ce modèle pour extraire des vecteurs de features sur les images.

Transférer ces features dans un DataFrame Spark.

Diffuser les poids (ou le modèle lui-même) sur le cluster pour montrer que tu maîtrises les traitements distribués.

Appliquer une PCA sur ces vecteurs dans Spark.

✅ 2. Travailler dans un environnement Cloud
Déployer un environnement AWS EMR (ou Databricks).

Stocker tes images et tes résultats dans S3.

Utiliser EMR uniquement pour les tests et démonstrations, pour minimiser le coût.

T’assurer que les serveurs sont situés en Europe (conformité RGPD).

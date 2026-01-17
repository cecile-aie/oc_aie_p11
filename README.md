# Traitement Big Data pour la reconnaissance d’images de fruits 🍎🍌🍇

**Projet OpenClassrooms Ingénieur IA - P11**  
**Auteur : Cécile MAYER**  
**Date : Août 2025**  

---

## 📌 Objectif du projet

Ce projet vise à **mettre en place une architecture Big Data** pour la reconnaissance d’images de fruits, capable de **supporter un passage à l’échelle**. Il s’inscrit dans le cadre d’un POC pour l’entreprise fictive *Fruits!*, une start-up AgriTech développant des outils de reconnaissance visuelle de fruits, notamment à travers une future application mobile grand public.

---

## 🧱 Démarche et méthodologie

Ce travail repose sur la **reprise, l’adaptation et l’optimisation** d’un notebook initial, conçu pour un environnement Big Data. La démarche suit une logique progressive :

### 1. 📥 Chargement & préparation des données
- Données image (Kaggle fruits-360) pré-organisées en `train/` et `test/` (~140 000 images)
- Extraction des labels depuis les chemins
- Partitionnement intelligent des données pour le calcul distribué

### 2. 🧠 Extraction de features via MobileNetV2
- Suppression de la couche finale du modèle (`include_top=False`)
- Diffusion des poids du modèle avec `sc.broadcast()` pour chaque worker Spark avec instanciation unique locale du modèle
- Traitement batch d’images avec `UDF` dans Spark

### 3. ⚙️ Réduction de dimension par PCA
- Standardisation des vecteurs
- Calcul du nombre optimal de composantes expliquant ≥ 90% de la variance
- Application de la PCA sur `train` et `test` avec les mêmes transformations

### 4. ☁️ Passage à l’échelle sur AWS EMR
- Tests locaux : 1 000 à 5 000 images
- Déploiement sur EMR avec JupyterHub (mode interactif) et Spark-submit (mode client puis cluster)
- Évaluation du temps de traitement, scalabilité
- Adaptation du nombre de workers et de partitions pour optimiser les performances

### 5. ✨ Bonus - Mise en oeuvre d'un algorithme de classification
- Test de régression logistique, RandomForrest, GradientBoosting
- Métriques globales, par classes
- Focus sur les catégories mal classées (matrice de confusion, TSNE)
---

## 🏗️ Infrastructure Big Data

- **Stockage :** S3 (`s3://ociae-p11`)
- **Traitement distribué :** AWS EMR (cluster Spark)
- **Visualisation & suivi par SSH et/ou tunneling SSH:**
  - Spark UI (ports 18080 / 20888)
  - JupyterHub (port 9443)
- **Traitement batch final :** `spark-submit` (mode client/cluster)

---

## 📁 Structure du dépôt

```
.
├──gitlab-ci.yml            # Script de déploiement/run depuis Gitlab
├── p11_app.py              # Script principal PySpark optimisé
├── run_p11.sh              # Script d'exécution (Spark-submit)
├──emr_config.json          # Configuration EMR (accès s3)
├──bootstrap-emr.sh         # Script d'amorçage du cluster 
├── p11_optimisé_local.ipynb    # Notebook de traitement local
├── p11_optimisé_EMR.ipynb      # Notebook EMR / JupyterHub
├── p11_classification.ipynb    # Débuts de tests de classification
├── P8_Notebook_Linux_EMR_PySpark_V1.0.ipynb  # Notebook original à reprendre
└── P8_Notebook_Linux_EMR_PySpark_V2.0 (PCA).ipynb  # Notebook original avec réduction PCA
├──Dockerfile               # Conteneur pour l'exécution locale
├──docker-compose.yml       # Ajout de services au docker local

```


---

## 🚀 Exécution du script sur EMR

### Pré-requis :
- Un cluster EMR actif avec Spark
- Données accessibles sur S3 (`s3://ociae-p11/images/`)

### Modes d'exécution :

- ▶️ **Via GitLab CI/CD**   
  Utilise le fichier [`🟡 .gitlab-ci.yml`](.gitlab-ci.yml) pour lancer automatiquement le traitement Big Data après chaque push.

- 🖥️ **En ligne de commande via SSH**  
  Exécution manuelle avec [`🟢 run_p11.sh`](./run_p11.sh), en mode client ou cluster.

Les résultats sont sauvegardés en format Parquet dans :
- `s3://ociae-p11/results_train`
- `s3://ociae-p11/results`

---

## 🧪 Résultats clés

| Mode                | Volume     | Workers | Partitions | Temps approx. |
|---------------------|------------|---------|------------|----------------|
| Local               | 1 000 img  | N/A     | 10         | ~50 sec        |
| EMR JupyterHub      | 1 000 img  | 5       | 10         | ~1 min         |
| EMR Spark-submit    | 50 000 img | 5 → 8   | 100        | ~7 min 30      |
| EMR Spark-submit    | 103 993 img| 8       | 200        | ~13 min        |

---

## ✅ Points forts techniques

- 💡 Optimisation mémoire : `model_instance` instancié une fois par worker
- 🔄 Mise en cache stratégique (`.cache()`) des DataFrames
- 📦 Broadcast des poids du modèle pour éviter les surcharges
- 📊 Réduction de dimension dynamique (PCA ≥ 90% de variance)
- ☁️ Traitement scalable et industrialisable (via `spark-submit`)

---

## 📌 Perspectives

- Amélioration du classifieur en aval (boosting, fine-tuning)
- Affinage de la réduction de dimension (t-SNE, UMAP ?)
- Intégration future dans une API ou une application mobile

---

## 🔐 Respect du RGPD

- Traitements réalisés dans des clusters situés sur le territoire européen
- Pas de stockage local de données personnelles
- Instance EMR maintenue active uniquement pour les tests/démos

---

## 📽️ Présentation du projet

[👉 Accéder à la présentation de synthèse (OneDrive)](https://1drv.ms/p/c/08F813C23A12D604/EXtK3sqeYFVMphQhEuaSfAYBH4BfFI8xSb7vYOU6vY40AQ?e=IqaUex)

Ce document utilisé en soutenance aborde :
- L'architecture choisie
- Le processus de traitement
- Les résultats obtenus
- Des pistes d'amélioration

---

## 🔐 Accès au bucket S3 (lecture seule)

Un utilisateur IAM nommé `s3-readonly` a été configuré avec les permissions minimales pour accéder aux buckets S3 `ociae-p11` et de logs EMR.

👤 **Accès AWS Console – Projet P11**

- **URL de connexion :** https://908027391515.signin.aws.amazon.com/console
- **Nom d’utilisateur :** s3-readonly
- **Mot de passe temporaire :** OC-aie#11

➡️ Vous devrez définir votre propre mot de passe lors de la première connexion.

Vous aurez accès en lecture seule au bucket `s3://ociae-p11` (par la console ou l’interface S3).


Pour toute question, n’hésitez pas à me contacter !  
🚜🍏 *Fruits! pour la planète... et le cloud.* ☁️🌱

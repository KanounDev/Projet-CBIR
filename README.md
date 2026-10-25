# CBIR - Content-Based Image Retrieval System

> Moteur de recherche d'images par similarité visuelle et mots-clés pour le dataset MIRFLICKR-1M (1 million d'images).

## 📋 Table des matières

- [Architecture du projet](#-architecture-du-projet)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure des données](#-structure-des-données)
- [API Endpoints](#-api-endpoints)
- [Descripteurs disponibles](#-descripteurs-disponibles)
- [Métriques de distance](#-métriques-de-distance)
- [Résultats d'évaluation](#-résultats-dévaluation)
- [Limitations connues](#-limitations-connues)

---

## 🏗 Architecture du projet

```
┌────────────────┐   requête (mots-clés,    ┌─────────────────────────────┐
│  Client web    │   image, descripteurs)   │  Serveur FastAPI            │
│ HTML/CSS/JS    │ ───────────────────────► │  (Python 3.11+)             │
│                │                          │                             │
│                │ ◄─────────────────────── │  ┌───────────────────────┐  │
│                │   liste d'URLs triées    │  │ Routes API (JSON)     │  │
│                │   + scores               │  ├───────────────────────┤  │
└────────────────┘                          │  │ Services:            │  │
                                            │  │  - text_search.py    │  │
                                            │  │  - search_service.py │  │
                                            │  │  - visual_search.py  │  │
                                            │  ├───────────────────────┤  │
                                            │  │ Indexing:            │  │
                                            │  │  - index_manager.py  │  │
                                            │  │  - faiss_index.py    │  │
                                            │  │  - vptree.py         │  │
                                            │  │  - distances.py      │  │
                                            │  └──────────┬────────────┘  │
                                            └─────────────┼───────────────┘
                                                          │
                          ┌───────────────────────────────┼──────────────────┐
                          │                               │                  │
                  ┌───────▼────────┐            ┌─────────▼────────┐  ┌──────▼──────────┐
                  │ MongoDB        │            │ Fichiers .npy    │  │ Images/vignettes│
                  │                │            │ (descripteurs)   │  │ (fichiers .jpg) │
                  │ - chemins      │            │                  │  │                 │
                  │ - URLs Flickr  │            │ - vgg16_gap512   │  │ - thumbs1-33/   │
                  │ - tags         │            │ - edgehistogram  │  │ - thumbs34-66/  │
                  │ - licences     │            │ - homogtexture   │  │ - thumbs67-100/ │
                  └────────────────┘            └──────────────────┘  └─────────────────┘
```

### Structure des répertoires

```
projet-CBIR/
├── start.bat                    # Script de lancement (Windows)
├── README.md                    # Ce fichier
├── plan_projet_cbir.md          # Plan de développement
│
├── backend/
│   ├── main.py                  # Point d'entrée FastAPI
│   ├── requirements.txt         # Dépendances Python
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes.py        # Endpoints API
│   │   │   └── upload.py        # Upload d'images
│   │   ├── db/
│   │   │   └── mongodb.py       # Connexion MongoDB
│   │   ├── schemas/
│   │   │   └── search.py        # Modèles Pydantic
│   │   ├── services/
│   │   │   ├── search_service.py    # Service de recherche unifié
│   │   │   ├── text_search.py       # Recherche par mots-clés
│   │   │   └── visual_search.py     # Recherche par similarité
│   │   ├── descriptors/
│   │   │   └── vgg.py           # Extracteur VGG16
│   │   └── indexing/
│   │       ├── index_manager.py # Gestionnaire d'index
│   │       ├── faiss_index.py   # Index FAISS (production)
│   │       ├── vptree.py        # VP-Tree (démonstration)
│   │       └── distances.py     # Fonctions de distance
│   ├── scripts/
│   │   ├── parse_descriptors.py # Parsing MPEG-7 → .npy
│   │   ├── build_indexes.py     # Construction des index
│   │   ├── build_metadata.py    # Tags + licences → metadata
│   │   ├── load_mongo.py        # Chargement MongoDB
│   │   └── comprehensive_evaluation.py  # Évaluation complète
│   └── tests/
│       └── test_search_contracts.py    # Tests unitaires du contrat et VP-Tree
│   └── evaluation_results/
│       ├── evaluation_results.json
│       └── evaluation_report.md
│
├── frontend/
│   ├── index.html               # Interface web
│   ├── style.css
│   └── app.js                   # Logique JavaScript
│
├── data/
│   ├── mirflickr/
│   │   └── raw/                 # Données brutes (À TÉLÉCHARGER)
│   │       ├── features_edgehistogram/
│   │       ├── features_homogeneoustexture/
│   │       ├── tags/
│   │       ├── license/
│   │       ├── thumbs1-33/
│   │       ├── thumbs34-66/
│   │       └── thumbs67-100/
│   ├── features/                # Descripteurs extraits (générés par scripts)
│   │   ├── vgg16_gap512_1M.npy
│   │   ├── edgehistogram.npy
│   │   └── homogeneoustexture.npy
│   ├── indexes/                 # Index de recherche (générés par scripts)
│   │   ├── vgg16_euclidean.faiss
│   │   ├── edgehistogram_euclidean.faiss
│   │   └── ...
│   ├── metadata/                # Métadonnées (générées par scripts)
│   │   └── metadata.jsonl
│   └── models/                  # Cache de poids VGG16 téléchargés
│
└── notebooks/
    └── exploration_donnees.ipynb
```

---

## ⚙️ Prérequis

### Logiciels requis

| Logiciel | Version | Usage |
|----------|---------|-------|
| **Python** | 3.11+ | Runtime principal |
| **MongoDB** | Version installée localement ; Docker Compose utilise 7.0 | Base de données métadonnées |
| **Git** | 2.0+ | Gestion de version |

### Bibliothèques Python principales

- **FastAPI** + Uvicorn (API web)
- **NumPy**, **SciPy** (calcul numérique)
- **FAISS** (index de similarité)
- **PyMongo** (MongoDB)
- **PyTorch** + torchvision (VGG16)
- **Pillow**, **OpenCV** (traitement d'images)

---

## 📥 Installation

### 1. Cloner le repository

```bash
git clone <repository-url>
cd projet-CBIR
```

### 2. Télécharger les données MIRFLICKR-1M

**Télécharger depuis** : https://press.liacs.nl/mirflickr/mirdownload.html

Les données MIRFLICKR ne sont pas incluses dans le dépôt Git et ne sont pas téléchargées par les scripts de démarrage. Il faut télécharger et extraire les cinq archives ci-dessous avant d'utiliser `start.bat` ou Docker Compose. Les scripts créent ensuite les descripteurs, index, métadonnées et documents MongoDB dérivés.

**Fichiers requis dans `data/mirflickr/raw/`** :

| Archive | Contenu | Taille | Obligatoire |
|---------|---------|--------|-------------|
| `mirflickr1m.zip` | Vignettes (thumbs1-33, thumbs34-66, thumbs67-100) | ~10 GB | ✅ Oui |
| `mirflickr1m_tags.zip` | Tags (mots-clés) | ~100 MB | ✅ Oui |
| `mirflickr1m_license.zip` | Licences et URLs Flickr | ~50 MB | ✅ Oui |
| `edgehistogram.zip` | Descripteurs Edge Histogram (MPEG-7) | ~600 MB | ✅ Oui |
| `homogentexture.zip` | Descripteurs Homogeneous Texture (MPEG-7) | ~200 MB | ✅ Oui |

**Structure attendue après extraction** :

```
data/mirflickr/raw/
├── thumbs1-33/          # 330,000 vignettes
├── thumbs34-66/         # 330,000 vignettes
├── thumbs67-100/        # 340,000 vignettes
├── tags/                # 1,000,000 fichiers .txt
├── license/             # 1,000,000 fichiers .txt
├── features_edgehistogram/
│   └── features_edgehistogram/
│       └── <sous-dossiers>/<id>.txt
└── features_homogeneoustexture/
    └── features_homogeneoustexture/
        └── <sous-dossiers>/<id>.txt
```

### 3. Démarrer l'application

Après avoir téléchargé et extrait les archives dans `data/mirflickr/raw/`, choisissez **une** méthode :

- **Windows sans Docker** : double-cliquez sur `start.bat`. Il crée un environnement virtuel, installe les dépendances manquantes et démarre MongoDB localement si nécessaire (MongoDB Community Server doit être installé et `mongod` disponible dans `PATH`).
- **Docker** : suivez la section « Déploiement avec Docker » ci-dessous. MongoDB est fourni par Compose.

Au premier démarrage, l'application génère les descripteurs, les index FAISS et les métadonnées manquants, puis charge MongoDB. Le calcul VGG16 et la construction des index peuvent prendre longtemps. Une connexion Internet est nécessaire une première fois pour télécharger les poids ImageNet de VGG16 dans le cache `data/models/torch/`.

---

## 🚀 Utilisation

### Lancement rapide (Windows)

Double-cliquez sur **`start.bat`** ou exécutez :

```bash
start.bat
```

Ce script prépare les données dérivées manquantes avant de lancer le backend et ouvre l'interface web. Les archives MIRFLICKR doivent déjà être extraites dans `data/mirflickr/raw/`.

Pour lancer manuellement depuis la racine du dépôt, activez votre environnement virtuel, démarrez MongoDB, puis exécutez :

```bash
python backend/scripts/prepare_data.py
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

La préparation réutilise les fichiers générés déjà présents et n'importe les métadonnées que si `cbir.images` est vide.

### Accès à l'application

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Interface web servie par FastAPI |
| `http://localhost:8000/docs` | Documentation API (Swagger) |
| `http://localhost:8000/api/health` | Vérification santé serveur |

L'interface et l'API sont servies par le même serveur. Les résultats visuels peuvent être relancés en cliquant sur une vignette.

---

## 📊 Structure des données

### Données téléchargées (obligatoires)

Ces données doivent être téléchargées depuis https://press.liacs.nl/mirflickr/mirdownload.html :

| Dossier | Description | Taille |
|---------|-------------|--------|
| `data/mirflickr/raw/thumbs*/` | Vignettes 64×64 px | ~10 GB |
| `data/mirflickr/raw/tags/` | Mots-clés (1 fichier/image) | ~100 MB |
| `data/mirflickr/raw/license/` | Métadonnées Flickr | ~50 MB |
| `data/mirflickr/raw/features_edgehistogram/` | Edge Histogram MPEG-7 | ~600 MB |
| `data/mirflickr/raw/features_homogeneoustexture/` | Homogeneous Texture MPEG-7 | ~200 MB |

### Données générées automatiquement au démarrage

Ces fichiers sont dérivés des archives dans `data/mirflickr/raw/`. `start.bat` et l'image Docker exécutent `backend/scripts/prepare_data.py` et ne recréent pas les éléments déjà présents.

| Dossier/Fichier | Script de génération | Taille |
|-----------------|----------------------|--------|
| `data/features/edgehistogram.npy`, `edgehistogram_ids.npy`, `homogeneoustexture.npy`, `homogeneoustexture_ids.npy` | `parse_descriptors.py` | ~1 GB |
| `data/features/vgg16_gap512_1M.npy` | `build_vgg16_features.py` (extraction depuis les vignettes) | ~2 GB |
| `data/indexes/*_euclidean.faiss` | `build_indexes.py` | ~3 GB |
| `data/metadata/metadata.jsonl` | `build_metadata.py` | ~200 MB |
| MongoDB (`cbir.images`) | `load_mongo.py` | ~500 MB |
| `data/models/torch/` (poids VGG16 téléchargés) | torchvision au premier calcul VGG16 | ~500 MB |

### Règle d'identifiants

**Important** : Les images MIRFLICKR sont numérotées de 1 à 1,000,000, mais les IDs dans la base de données et les descripteurs vont de 0 à 999,999.

```
ID = numéro_image - 1
```

Exemple : `im43.jpg` ↔ ID `42`

---

## 🔌 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/health` | GET | État du serveur |
| `/api/descriptors` | GET | Liste des descripteurs disponibles |
| `/api/images` | GET | Parcourir les images (pagination) |
| `/api/search` | POST | Recherche textuelle et/ou visuelle |
| `/api/search/upload` | POST | Recherche avec une image requête externe (VGG16) |

### Exemple de requête de recherche

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["sunset", "beach"],
    "keyword_mode": "and",
    "query_image_id": 42,
    "descriptors": ["vgg16"],
    "metrics": {"vgg16": "euclidean"},
    "index_type": "faiss",
    "k": 20
  }'
```

---

## 🎯 Descripteurs disponibles

| Descripteur | Dimensions | Type | Recommandé pour |
|-------------|------------|------|-----------------|
| **vgg16** | 512 | Deep Learning (CNN) | Similarité sémantique ✅ |
| **edgehistogram** | 150 | MPEG-7 | Patterns de contours |
| **homogeneoustexture** | 43 | MPEG-7 | Textures |

### Index de recherche

- **FAISS Flat** est l'index principal et recherche sur la base complète.
- **VP-Tree** est une démonstration euclidienne limitée aux 50 000 premières images. Sa construction n'est pas lancée automatiquement, car elle peut prendre longtemps. Pour le créer manuellement après la préparation des données, exécutez `python backend/scripts/build_vptree_indexes.py` depuis la racine du dépôt. Les autres métriques restent disponibles en brute-force ou avec FAISS selon leur support ; seules les distances métriques sont adaptées à l'élagage d'un VP-Tree. L'option apparaît dans l'interface seulement pour les arbres accompagnés d'un manifeste valide.
- **Recherche exhaustive** est une référence et peut être lente sur un million d'images.
- La fusion de plusieurs descripteurs calcule les distances de façon exhaustive. En recherche combinée texte + image, les mots-clés filtrent les candidats, puis les descripteurs les classent.

Les temps varient selon la machine et les paramètres de recherche. Les chiffres mesurés et leur protocole sont consignés dans le rapport d'évaluation ci-dessous ; ils ne constituent pas une garantie de latence.

---

## 📏 Métriques de distance

| Métrique | Compatible FAISS | Recommandée pour |
|----------|------------------|------------------|
| **euclidean** | ✅ Oui | Tous descripteurs |
| **cosine** | ✅ Oui | Tous descripteurs |
| **chi2** | ❌ Non | Edge Histogram (brute-force uniquement) |
| **l1** | ✅ Oui | MPEG-7 (FAISS Flat ou brute-force) |

VGG16 étant déjà L2-normalisé, sa recherche cosine réutilise l'index euclidien. Les index cosine/L1 optionnels des descripteurs MPEG-7 ne sont pas créés au démarrage. Pour les construire après la préparation des données :

```powershell
python backend\scripts\build_indexes.py --faiss --descriptors edgehistogram homogeneoustexture --metrics cosine l1
```

La recherche chi2 reste exhaustive. Les anciens index optionnels sans manifeste de construction sont considérés comme non vérifiés ; l'exécution de la commande les reconstruit avec le contrat courant avant qu'ils soient proposés dans l'interface.

---

## 📈 Résultats d'évaluation

### Performances des index

| Descripteur | Brute Force | FAISS | Speedup |
|-------------|-------------|-------|---------|
| vgg16 | 1714.61 ms | 328.42 ms | **5.2x** |
| edgehistogram | 392.97 ms | 92.85 ms | **4.2x** |

Ces valeurs proviennent de `backend/evaluation_results/evaluation_report.md`. Le rapport ne fournit pas de comparaison agrégée brute-force/FAISS équivalente pour Homogeneous Texture.
Ce sont des mesures historiques obtenues avant la correction du contrat de métriques ; elles sont conservées pour contexte et doivent être recalculées avant toute comparaison de performance dans le rapport final.

### Vérifications self-retrieval

Le taux de self-retrieval indique seulement si le vecteur de l'image requête revient en première position. C'est un contrôle de cohérence, **pas** une mesure de pertinence sémantique, de précision ou de qualité de recherche. Le rapport compare également les résultats FAISS au brute-force avec le Jaccard des ensembles top-k ; les résultats varient selon le descripteur et la métrique. L'évaluation ne comporte pas de jugements de pertinence humaine.

### Recherche textuelle

| Mode | Temps moyen | Exemple |
|------|-------------|---------|
| AND | ~5 ms | "sunset AND beach" |
| OR | ~2 ms | "sunset OR beach" |

---

## ⚠️ Limitations connues

### Données

- **Vignettes uniquement** : Les images font 64×64 px (pas d'images en pleine taille en local)
- **Pas de descripteur couleur** : MIRFLICKR ne fournit pas de descripteur couleur MPEG-7
- **URLs Flickr optionnelles** : Seulement ~58% des images ont une URL Flickr valide

### Index

- **VP-Tree** : Limité à 50,000 images (démonstration uniquement)
- **Chi-squared** : Disponible en recherche exhaustive uniquement.
- **Index optionnels hérités** : Les index FAISS cosine/L1 créés par les anciennes versions doivent être reconstruits avec `build_indexes.py` pour garantir que les vecteurs de requête et ceux indexés utilisent la même représentation.

### Performances

- **Premier démarrage lent** : Chargement des index en mémoire (~30 secondes)
- **Recherche brute-force** : Les temps dépendent du descripteur, du matériel et de la charge ; la comparaison présente dans le rapport est limitée à un petit ensemble de requêtes.

---

## 🐳 Déploiement avec Docker (Optionnel)

### Prérequis Docker

- **Docker** 20.10+ installé
- **Docker Compose** 2.0+ installé
- Au moins **8 GB de RAM** disponible pour Docker

### Architecture Docker

```
┌─────────────────────────────────────────────────────────┐
│  Docker Network: cbir-network                           │
│                                                          │
│  ┌──────────────────┐         ┌──────────────────┐     │
│  │  cbir-backend    │         │  cbir-mongo      │     │
│  │  Port: 8000      │◄───────►│  Port: 27017     │     │
│  │                  │         │                  │     │
│  │  Python 3.11     │         │  MongoDB 7.0     │     │
│  │  FastAPI         │         │                  │     │
│  └────────┬─────────┘         └──────────────────┘     │
│           │                                              │
│           │ Volumes montés (données non copiées)        │
│           ▼                                              │
│  ┌──────────────────────────────────────────────┐      │
│  │  ./data/features → /app/data/features        │      │
│  │  ./data/indexes → /app/data/indexes          │      │
│  │  ./data/mirflickr → /app/data/mirflickr      │      │
│  └──────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### Lancement avec Docker

```bash
# Construire l'image et démarrer les services
docker compose up --build -d

# Suivre la préparation des données et le démarrage
docker compose logs -f backend

# Arrêter les services
docker compose down

# Arrêter et supprimer les volumes
docker compose down -v
```

Le backend attend que MongoDB soit prêt, puis prépare automatiquement les fichiers absents et importe les métadonnées si la collection `cbir.images` est vide. Les fichiers générés sont écrits dans les dossiers `data/features/`, `data/indexes/`, `data/metadata/` et `data/models/` sur l'hôte ; les données brutes sont montées en lecture seule. Le premier démarrage peut prendre des heures pour extraire les caractéristiques VGG16 des 1 million de vignettes. Une connexion Internet est nécessaire pour télécharger les poids ImageNet de VGG16 si le cache est vide.

### Accès avec Docker

| Service | URL |
|---------|-----|
| Interface web et backend API | http://localhost:8000/ |
| API Docs | http://localhost:8000/docs |
| MongoDB | localhost:27017 |

### Volumes Docker

| Volume | Usage |
|--------|-------|
| `./data/features` | Descripteurs calculés (persistants, écriture backend) |
| `./data/indexes` | Index FAISS (persistants, écriture backend) |
| `./data/metadata` | Métadonnées générées (persistantes, écriture backend) |
| `./data/mirflickr` | Archives et fichiers bruts (montés en lecture seule) |
| `./data/models` | Cache des poids VGG16 (persistant) |
| `mongo-data` | Données MongoDB (persistant) |

### Avantages de Docker

- ✅ **Environnement reproductible** : Même configuration partout
- ✅ **Installation simplifiée** : Une seule commande `docker compose up --build`
- ✅ **MongoDB inclus** : Pas d'installation manuelle
- ✅ **Isolation** : Pas de conflit avec autres projets

### Limitations Docker

- ⚠️ **Premier démarrage lent** : Calcul de 1 million de vecteurs VGG16 et construction des index
- ⚠️ **Espace disque** : Plusieurs gigaoctets sont nécessaires en plus des archives MIRFLICKR
- ⚠️ **Mémoire requise** : Prévoir au moins 8 GB de RAM pour les index et le backend
- ⚠️ **Accès aux vignettes** : Montage de volumes requis

### Dépannage Docker

```bash
# Si MongoDB ne démarre pas
docker compose restart mongo

# Si le backend ne trouve pas les données
docker compose down
docker compose up --build -d

# Vérifier l'espace disque
docker system df

# Nettoyer les images inutilisées
docker system prune -a
```

---

## 🛠 Développement

### Tests automatisés

Les tests unitaires sont indépendants des données MIRFLICKR et de MongoDB :

```bash
cd backend
python -m unittest discover -s tests
```

Après le démarrage, vérifier aussi `http://localhost:8000/api/health` et tester une recherche depuis l'interface ou la documentation interactive sur `http://localhost:8000/docs`.

### Évaluation complète

```bash
cd backend
python scripts/comprehensive_evaluation.py
```

Les résultats sont sauvegardés dans `backend/evaluation_results/`.

---

## 📝 Licence

Ce projet utilise le dataset MIRFLICKR-1M sous licence Creative Commons.
Voir `data/mirflickr/raw/license/` pour les licences individuelles des images.

---

## 👤 Auteur

Projet académique - Système CBIR MIRFLICKR-1M

---

## 📚 Références

- [MIRFLICKR-1M Dataset](https://press.liacs.nl/mirflickr/)
- [MPEG-7 Descriptors](https://www.chiariglione.org/mpeg/standards/mpeg-7)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

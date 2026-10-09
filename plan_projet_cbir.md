# Plan de développement : moteur d'indexation et de recherche d'images Textuel/Visuel (CBIR)

> Document de pilotage pour guider Claude étape par étape. Chaque étape contient : objectif, outils, contexte à fournir à Claude, livrable et critère de validation. Aucun code n'est inclus.
>
> **Version mise à jour** : adaptée aux données réellement téléchargées (`data/mirflickr/raw`) et à la diapo d'évaluation (démo déployée sur PC LIASA).
>
> **Révision de cohérence** : alignée sur les 5 diapos du sujet (Application, Fonctionnalités, Architecture, Évaluation, Outils). Voir la section « Correspondance avec les diapos du sujet » dans la section 1.

---

## 0. Résumé du cahier des charges

**Objectif** : une application client/serveur de recherche d'images par mots-clés et/ou par similarité visuelle.

**Deux jeux de données** :
1. **MIRFLICKR-1M** : 1 million d'images. Ce qui est réellement disponible dans `data/mirflickr/raw` :
   - **2 descripteurs MPEG-7 fournis** : Edge Histogram (`features_edgehistogram`, **150 valeurs** par image : 80 locales + 5 globales + 65 semi-globales) et Homogeneous Texture (`features_homogeneoustexture`, **43 valeurs** par image). Un fichier texte par image, une valeur par ligne. Rien à calculer pour ceux-là, mais **aucun descripteur couleur n'est fourni**.
   - **Descripteurs VGG16 (deep learning)** : **512 dimensions** par image, extraits avec VGG16 pré-entraîné sur ImageNet (conv features + Global Average Pooling). Fichiers : `data/features/vgg16_gap512_1M.npy` (~2 GB), modèles dans `data/models/vgg16_features_imagenet.pth`. **Recommandé pour la similarité visuelle sémantique** (performances bien supérieures à MPEG-7).
   - **Tags** (`tags`) : source des mots-clés.
   - **Licences** (`license`) : fichier texte clé/valeur (date de dépôt, licence, URL de la licence Creative Commons, titre, nom/pseudo/identifiant du propriétaire). **Selon les fichiers, il contient aussi `Photo id`, `Photo url` (image Flickr) et `Web url` (page Flickr de la photo) ; ces trois lignes sont absentes de certains fichiers** (voir « Constats sur les données réelles » ci-dessous).
   - **Vignettes uniquement** (`thumbs1-33`, `thumbs34-66`, `thumbs67-100`) : pas d'images en pleine taille en local (l'exemple `im1.jpg` fait 64×64 px). L'affichage repose sur les vignettes ; l'image complète s'ouvre via l'URL Flickr quand elle existe dans le fichier `license` (pas pour toutes les images, et certains liens Flickr peuvent être morts).
2. **Base « Objects »** : images pour lesquelles les descripteurs sont à calculer (couleur, LBP, HOG, etc.).

**Fonctionnalités minimales** :
- Requête par mots-clés **ET/OU** par image (image de la base ou image issue d'un résultat précédent).
- L'utilisateur choisit le ou les descripteurs utilisés par le serveur (**MPEG-7** : Edge Histogram, Homogeneous Texture ; **Deep Learning** : VGG16).
- Le serveur calcule/utilise les descripteurs, cherche les images similaires et renvoie la liste des URLs.
- Affichage au client : image requête + images similaires triées par distance croissante (la plus ressemblante en premier ; la diapo dit « ordre croissant de ressemblance », à interpréter ainsi, à confirmer avec l'enseignant).
- Relance d'une recherche en cliquant sur une image résultat.
- Temps de recherche optimisé grâce à une bonne structure de l'espace des index (FAISS pour 1M+ vectors, VP-Tree pour démonstration).

**Livrables d'évaluation** (diapo du cours) : **rapport**, **démo déployée sur PC LIASA**, **code source + base de données**. Critères : méthodologie, conception, sérieux (assiduité).
> ⚠️ À confirmer : la diapo affiche une date limite du 26 octobre **2025** ; vérifier la date en vigueur.

**Contraintes** : client = application web ; serveur = Python (FastAPI, cité dans la diapo « Outils » ; Flask, Django ou Streamlit possibles) ; échange de données en JSON (XML autorisé) ; images stockées dans des répertoires, seules les URLs/chemins sont dans la base de données ; Docker éventuel pour le packaging et le déploiement ; déploiement simple, code non optimisé à 100 % (cette dernière précision ne figure pas sur les diapos).

---

## 1. Décisions techniques (à figer avant de commencer)

| Composant | Choix recommandé | Alternative |
|---|---|---|
| Langage serveur | Python 3.11+ | – |
| Framework API | **FastAPI** + Uvicorn | Flask, Django |
| Calcul numérique | **NumPy**, **SciPy** | – |
| Descripteurs image (Objects, et couleur sur vignettes MIRFLICKR si souhaité) | **scikit-image** (LBP, HOG), **OpenCV** (histogrammes couleur) | Pillow pour le chargement |
| Descripteurs MIRFLICKR | **Edge Histogram** + **Homogeneous Texture** (fournis, à parser) | – |
| **Descripteurs deep learning (MIRFLICKR)** | **VGG16** (conv features + Global Average Pooling → **512 dims**, L2-normalisés) | ResNet, EfficientNet |
| Index de similarité | **FAISS** (production, 1M+ vectors) + **VP-Tree** (démonstration, ≤100k vectors) | `scikit-learn NearestNeighbors`, `scipy.spatial.cKDTree` |
| Base de données | **MongoDB** (métadonnées : URLs, tags, licences) | Elasticsearch (lourd) |
| Accès MongoDB depuis Python | **pymongo** | motor (version asynchrone) |
| Recherche texte (mots-clés) | **Index texte MongoDB** (`$text`) ou index sur le champ liste de mots-clés (ET/OU) | Elasticsearch si besoin de plus de souplesse (pluriels, lemmes) |
| Client | **HTML + CSS + JavaScript** (vanilla) ou **Streamlit** si le temps manque | React (absent de la diapo « Outils », donc à valider) |
| Packaging | **Docker** (optionnel, « éventuellement » selon la diapo ; ne pas supposer qu'il est installé sur le PC LIASA) | – |
| Tests, évaluation, données | **pytest**, **Matplotlib**, **Pandas**, **Pillow**, **joblib** | – |
| Hébergement | **Démo sur PC LIASA** (principal, cf. diapo d'évaluation). Option bonus : backend sur Render, frontend sur Vercel/Cloudflare Pages | – |
| Environnement dev | **Anaconda/venv** + **Jupyter** pour les essais | – |

**Correspondance avec les diapos du sujet** :

| Élément | Diapos | Ce plan |
|---|---|---|
| Descripteurs MIRFLICKR | MPEG-7 fournis | Edge Histogram + Homogeneous Texture (MPEG-7) + **VGG16 (deep learning, recommandé)** |
| Descripteurs Objects | À calculer (couleur, LBP, HOG) | `color_hist`, `lbp`, `hog` |
| Client | Application web | HTML/CSS/JS (Streamlit en plan B) |
| Serveur | Node.js, IIS, Django, Flask, Streamlit ; web service FastAPI | Python + **FastAPI** |
| Échange | XML ou JSON | JSON |
| Base de données | Images en répertoires, URLs en BD ; MongoDB / Elasticsearch listés | **MongoDB** (URLs, tags, licences) |
| Index | VP-Tree en exemple | **FAISS** (production) + VP-Tree (démonstration) |
| Outils | Python, FastAPI, MongoDB, Elasticsearch, Anaconda/Jupyter/Spyder, Streamlit, Flask, Django, HTML/CSS/JS, SciPy, NumPy, Docker | Tous cohérents ; Elasticsearch, Flask, Django, Streamlit en alternatives |
| Évaluation | Rapport, démo sur PC LIASA, code source + BD | Étapes 12 et 13 |

Bibliothèques ajoutées par ce plan et absentes de la diapo « Outils » (justifiées par les descripteurs LBP/HOG/couleur, l'index, les descripteurs deep learning et l'évaluation) : **PyTorch**, **torchvision**, scikit-image, OpenCV, FAISS, scikit-learn, pymongo, Pillow, Pandas, joblib, pytest, Matplotlib.

**Décision à prendre tôt** : jeu de travail sur MIRFLICKR. Le choix actuel est d'utiliser **le MIRFLICKR-1M complet** en local. Garder un sous-ensemble de 10 000 à 50 000 images pour développer vite, tester chaque module et, si besoin, pour un déploiement en ligne limité en mémoire.

**Ordre de grandeur mémoire (à vérifier sur les vrais fichiers)** : 1 M de vecteurs de 150 dimensions (Edge Histogram) en float32 ≈ 600 Mo ; 43 dimensions (Homogeneous Texture) ≈ 172 Mo. Le VP-Tree sur 1 M de vecteurs est lourd à construire en Python pur : prévoir de le démontrer sur un sous-ensemble et d'utiliser FAISS / `NearestNeighbors` sur la base complète.

### Constats sur les données réelles (échantillons examinés à l'étape 2)

| Élément | Constat | Conséquence |
|---|---|---|
| Edge Histogram | 150 valeurs/fichier, une par ligne (80 locales, 5 globales, 65 semi-globales) ; valeurs d'environ 0,05 à 2 | Matrice 1 M × 150 ≈ 600 Mo en float32 ; le bloc de 5 valeurs globales a une échelle plus grande, à normaliser avant de combiner les distances |
| Homogeneous Texture | 43 valeurs/fichier, une par ligne, autour de 1 | Matrice 1 M × 43 ≈ 172 Mo |
| Tags | Fichier texte, un mot-clé par ligne (exemple : `explore`) | Parser simple ; un fichier peut contenir un seul mot ou être vide |
| Licence | Clé/valeur. Deux variantes vues : (a) date, licence, URL CC, titre, nom/pseudo/id du propriétaire ; (b) la même chose **plus** `Photo id`, `Photo url`, `Web url`. L'URL de la licence est celle de Creative Commons, pas celle de la photo | Le parser lit chaque ligne `clé : valeur` et laisse les champs Flickr vides s'ils manquent ; le champ « URL source » du modèle est donc **optionnel** ; compter (étape 2) combien de fichiers ont `Photo url` |
| Vignette | `im1.jpg` : 64×64 px | Affichage très petit : prévoir un agrandissement (le plan évite d'agrandir le fichier lui-même) |
| Numérotation | **Confirmée** : `license` va de `0` à `999999` (1 000 000 fichiers), les vignettes de `im1` à `im1000000` (1 000 000 fichiers) | id = numéro de l'image − 1 ; `im1.jpg` ↔ `0.txt` |
| Comptage des fichiers | Edge Histogram : 1 000 003 fichiers = 1 M + 2 `.DS_Store` + `feature_names.txt` ; Homogeneous Texture : 1 000 001 = 1 M + `feature_names.txt` ; `license` et `tags` : 1 M exactement ; vignettes : 330 000 + 330 000 + 340 000 | Les parsers ne retiennent que les fichiers nommés `<nombre>.txt` (ignorer `.DS_Store` et `feature_names.txt`) |
| Structure des dossiers | Dossiers imbriqués : `features_edgehistogram/features_edgehistogram/<sous-dossier>/<id>.txt` (idem pour la texture) | Parcours récursif (`os.walk`), jamais de chemin construit à la main |
| `feature_names.txt` | 150 noms pour l'Edge Histogram (16 blocs locaux × 5 directions, 5 globaux, 65 semi-globaux : 4 verticaux, 4 horizontaux, 4 coins, 1 centre) ; 43 noms pour la texture (DC, SD, 30 énergies = 5 échelles × 6 orientations, 5 valeurs par échelle, 6 par orientation) | Utile pour le rapport et pour pondérer les blocs de dimensions |
| URL Flickr | `Photo url` présente dans 2 880 fichiers sur les 5 000 premiers parcourus (≈ 58 %, échantillon non aléatoire) | Les liens Flickr restent optionnels ; refaire le comptage sur un échantillon aléatoire pour le rapport |

---

## 2. Architecture du projet

### 2.1 Vue d'ensemble

```
┌────────────────┐   requête (mots-clés,    ┌─────────────────────────────┐
│  Client web    │   image, descripteurs)   │  Serveur FastAPI (PC LIASA)  │
│ HTML/CSS/JS    │ ───────────────────────► │                              │
│                │                          │  ┌───────────────────────┐   │
│                │ ◄─────────────────────── │  │ Routes API (JSON)     │   │
│                │   liste d'URLs triées    │  ├───────────────────────┤   │
└────────────────┘   + scores               │  │ Service de recherche  │   │
                                            │  │  - texte              │   │
                                            │  │  - visuel             │   │
                                            │  │  - fusion ET/OU       │   │
                                            │  ├───────────────────────┤   │
                                            │  │ Extracteurs / parsers │   │
                                            │  │ de descripteurs       │   │
                                            │  ├───────────────────────┤   │
                                            │  │ Index (VP-Tree/FAISS) │   │
                                            │  └──────────┬────────────┘   │
                                            └─────────────┼────────────────┘
                                                          │
                          ┌───────────────────────────────┼─────────────────┐
                          │                               │                 │
                  ┌───────▼────────┐            ┌─────────▼───────┐  ┌──────▼────────────┐
                  │ Base de données│            │ Fichiers index  │  │ Stockage images   │
                  │ MongoDB        │            │ (.npy, .pkl)    │  │ vignettes locales │
                  │ URLs, tags,    │            │ descripteurs    │  │ + image complète  │
                  │ licences       │            │                 │  │ via URL Flickr    │
                  └────────────────┘            └─────────────────┘  └───────────────────┘
```

> Correspondance avec la diapo « Architecture » : la « Feature & Image Database » du schéma regroupe ici MongoDB (URLs, tags, licences), les fichiers de descripteurs `.npy`, les index sérialisés et le dossier d'images/vignettes. Tout cela fait partie du livrable « BD ».

### 2.2 Deux phases bien séparées

1. **Phase hors ligne (une seule fois)** : parsing des répertoires, tags et licences ; lecture des descripteurs MPEG-7 (MIRFLICKR) et calcul des descripteurs (Objects) ; construction des index ; remplissage de la base de données.
2. **Phase en ligne (à chaque requête)** : réception de la requête, calcul du descripteur de l'image requête si nécessaire, recherche dans l'index, fusion avec le texte, renvoi des résultats.

### 2.3 Arborescence proposée

```
cbir-project/
├── README.md
├── docker-compose.yml              (optionnel)
├── .env.example
│
├── data/                           (non versionné dans Git)
│   ├── mirflickr/
│   │   └── raw/                    (données telles que téléchargées)
│   │       ├── features_edgehistogram/
│   │       │   └── features_edgehistogram/   (puis sous-dossiers de fichiers)
│   │       ├── features_homogeneoustexture/
│   │       │   └── features_homogeneoustexture/
│   │       ├── license/
│   │       │   └── license/
│   │       ├── tags/
│   │       │   └── tags/
│   │       ├── thumbs1-33/
│   │       │   └── thumbs/                   (puis sous-dossiers d'images)
│   │       ├── thumbs34-66/
│   │       │   └── thumbs/
│   │       └── thumbs67-100/
│   │           └── thumbs/
│   ├── objects/
│   │   └── images/
│   ├── metadata/                   (table de correspondance id ↔ chemin/URL/tags)
│   ├── features/                   (descripteurs au format .npy)
│   └── indexes/                    (index sérialisés)
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                 (point d'entrée FastAPI, CORS, fichiers statiques des vignettes)
│   │   ├── config.py               (chemins, paramètres)
│   │   ├── api/
│   │   │   ├── routes_search.py    (recherche texte/image/mixte)
│   │   │   ├── routes_images.py    (liste, détail, images de la base)
│   │   │   └── routes_meta.py      (descripteurs disponibles, santé)
│   │   ├── schemas/                (modèles de requête/réponse)
│   │   ├── services/
│   │   │   ├── search_service.py   (orchestration des recherches)
│   │   │   ├── text_search.py      (mots-clés)
│   │   │   ├── visual_search.py    (similarité visuelle)
│   │   │   └── fusion.py           (combinaison ET/OU, pondération)
│   │   ├── descriptors/
│   │   │   ├── base.py             (interface commune d'un descripteur)
│   │   │   ├── mpeg7_parser.py     (Edge Histogram, Homogeneous Texture)
│   │   │   ├── tags_parser.py      (lecture des tags MIRFLICKR)
│   │   │   ├── color_hist.py
│   │   │   ├── lbp.py
│   │   │   └── hog.py
│   │   ├── indexing/
│   │   │   ├── distances.py        (euclidienne, cosinus, L1, chi²)
│   │   │   ├── vptree.py
│   │   │   ├── faiss_index.py      (ou knn_index.py)
│   │   │   └── index_manager.py    (chargement/choix de l'index)
│   │   └── db/
│   │       ├── database.py
│   │       └── models.py
│   ├── scripts/                    (exécutés hors ligne)
│   │   ├── 01_prepare_dataset.py
│   │   ├── 02_extract_descriptors.py
│   │   ├── 03_build_indexes.py
│   │   └── 04_fill_database.py
│   └── tests/
│
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   ├── js/
│   │   ├── api.js                  (appels au serveur)
│   │   ├── search.js               (formulaire, choix des descripteurs)
│   │   ├── results.js              (affichage, relance depuis un résultat)
│   │   └── main.js
│   └── assets/
│
├── notebooks/                      (essais Jupyter, comparaisons de descripteurs)
└── docs/                           (rapport, schémas, captures)
```

> La structure exacte sous chaque dossier (noms des sous-dossiers, extensions des fichiers) est à relever à l'étape 2 et à reporter ici.

### 2.4 Contrat d'API (à définir avant le code)

| Endpoint | Rôle | Entrées | Sortie |
|---|---|---|---|
| `GET /descriptors` | Liste des descripteurs disponibles par base | – | noms, dimension, bases concernées |
| `GET /images` | Parcourir/paginer les images de la base | page, base | URLs vignettes + mots-clés |
| `POST /search` | Recherche texte, image ou mixte | mots-clés, mode ET/OU, id ou fichier image, descripteurs choisis, nombre de résultats, distance | image requête + liste triée (URL vignette, URL source, score, mots-clés) |
| `GET /health` | Vérification du déploiement | – | statut |

Les vignettes sont servies par le backend (fichiers statiques) ; l'image en pleine taille s'ouvre via l'URL Flickr quand elle est présente dans le fichier `license` (pas toujours) ; sinon l'interface n'affiche que la vignette.

### 2.5 Modèle de données

- **Collection MongoDB `images`** (un document par image) : identifiant numérique, base d'origine (mirflickr/objects), chemin ou URL de la vignette, URL de l'image Flickr et URL de la page Flickr (`photo_url`, `web_url` ; MIRFLICKR uniquement, **optionnelles** car absentes de certains fichiers `license`), identifiant Flickr de la photo, licence et URL de la licence (MIRFLICKR), propriétaire (nom, id), largeur/hauteur, et **champ liste `keywords`** (mots-clés normalisés, avec un index pour la recherche ET/OU). Mots-clés MIRFLICKR : fichiers `tags` ; mots-clés Objects : noms de classes/dossiers.
- Pas de table de liaison : le modèle document de MongoDB rend la jointure image ↔ mot-clé inutile.
- **Fichiers de descripteurs** : un fichier par descripteur et par base, où la ligne *i* correspond à l'identifiant *i* de l'image.
- **Fichiers d'index** : un par descripteur (et éventuellement un par combinaison).
- **Règle d'identifiants MIRFLICKR** (à respecter partout) : **id = numéro de l'image − 1**. Les images (`imXXXX.jpg`) sont numérotées à partir de 1, alors que les tags, licences et descripteurs sont numérotés à partir de 0 (ex. `im609999.jpg` ↔ `609998.txt`). Les données sont réparties en 100 dossiers de 10 000 images.

---

## 3. Étapes de développement

### Étape 1 : Cadrage et préparation de l'environnement

- **Objectif** : projet initialisé, outils installés, données disponibles.
- **Actions** :
  1. Créer le dépôt Git et l'arborescence ci-dessus.
  2. Créer l'environnement (Anaconda ou venv) et installer les dépendances de base.
  3. Vérifier que `data/mirflickr/raw` est complet (descripteurs, tags, licences, les 3 archives de vignettes).
  4. Télécharger la base Objects.
- **Outils / sources** : Anaconda, Git, Jupyter, MIRFLICKR (https://press.liacs.nl/mirflickr/ : l'URL du sujet contient une faute, `.ntml` au lieu de `.html`), pages de la base Objects fournies par le professeur.
- **Contexte à donner à Claude** : le cahier des charges (captures), l'arborescence, les choix techniques de la section 1, ton niveau et ton système d'exploitation.
- **Livrable** : projet qui s'installe et s'exécute, fichier `requirements.txt`, README de départ.
- **Validation** : l'environnement se recrée depuis zéro avec une seule commande.

### Étape 2 : Exploration des données

- **Objectif** : comprendre exactement ce qui est fourni avant d'écrire le moindre module.
- **Actions** :
  1. Parcourir `raw/` : relever la structure réelle (sous-dossiers, nombre de fichiers par dossier, extensions) pour chaque dossier, y compris la particularité `thumbsX-XX/thumbs/`.
  2. Examiner les deux descripteurs MPEG-7 fournis (Edge Histogram, Homogeneous Texture) : dimension, format du fichier, plages de valeurs.
  3. Ouvrir quelques fichiers `tags` et `license` : format, présence (ou non) d'une URL Flickr, de la licence, et des tags.
  4. **Vérifier le mapping** : confirmer la règle id = numéro d'image − 1 sur plusieurs images, entre vignettes, tags, licences et descripteurs.
  5. Examiner la base Objects : nombre d'images, classes, taille, format, annotations disponibles.
  6. Décider du sous-ensemble de développement (la base complète reste le but).
- **Outils / sources** : Jupyter, Pandas, Pillow, documentation MPEG-7 (savvash.blogspot.com, cbir.info), documentation de MIRFLICKR.
- **Contexte à donner à Claude** : les **premières lignes d'un fichier de descripteurs**, d'un fichier de tags et d'un fichier de licence, la **liste des dossiers** et la documentation du format.
- **Livrable** : notebook d'exploration et note courte « données » (structure, formats, volumes, règle de mapping).
- **Validation** : tu peux afficher une vignette avec ses deux vecteurs de descripteur, ses tags et sa licence.

### Étape 3 : Préparation du jeu de données

- **Objectif** : un jeu propre et indexé par identifiants.
- **Actions** :
  1. Attribuer un identifiant numérique unique à chaque image (MIRFLICKR : id = numéro − 1 ; Objects : liste ordonnée).
  2. Construire le sous-ensemble de développement (échantillonnage aléatoire reproductible).
  3. **MIRFLICKR** : conserver les vignettes telles quelles (ne pas les agrandir). **Objects** : générer des miniatures (par exemple 256 px) pour accélérer l'affichage.
  4. Produire le fichier de métadonnées (identifiant, base, chemin vignette, URL source, licence, mots-clés).
- **Outils** : Python, Pillow, Pandas.
- **Contexte à donner à Claude** : résultats de l'étape 2 et le modèle de données (section 2.5).
- **Livrable** : script `01_prepare_dataset.py` et fichier de métadonnées.
- **Validation** : le nombre d'images, d'identifiants et de lignes de métadonnées est identique ; la vignette, les tags et le descripteur d'un même id correspondent bien à la même image.

### Étape 4 : Lecture des descripteurs MPEG-7 et des tags (MIRFLICKR)

- **Objectif** : transformer les descripteurs et tags fournis en structures exploitables.
- **Actions** :
  1. Écrire le module de lecture (`mpeg7_parser`) selon le format réel des fichiers, pour **Edge Histogram** et **Homogeneous Texture**.
  2. Normaliser chaque descripteur (taille fixe, type numérique, éventuelle normalisation).
  3. Sauvegarder une matrice `.npy` par descripteur, alignée sur les identifiants.
  4. Écrire `tags_parser` : lecture des tags et préparation pour la base de données.
- **Outils / sources** : Python, NumPy ; référence MPEG-7 (savvash.blogspot.com), LIRE (http://www.semanticmetadata.net/lire/) uniquement comme référence de lecture du format.
- **Contexte à donner à Claude** : exemple réel d'un fichier de chaque type, description du format de chaque descripteur.
- **Livrable** : modules `mpeg7_parser` et `tags_parser`, script `02_extract_descriptors.py` (partie MIRFLICKR), fichiers `.npy`.
- **Validation** : dimensions attendues respectées, aucune valeur manquante, vérification manuelle sur 5 images.

### Étape 5 : Calcul des descripteurs de la base Objects

- **Objectif** : calculer au moins 3 descripteurs (couleur, texture LBP, forme/contours HOG).
- **Actions** :
  1. Définir une interface commune : image en entrée → vecteur de taille fixe en sortie.
  2. Implémenter l'histogramme de couleur (espace HSV, quantification), LBP (histogramme uniforme) et HOG (redimensionnement fixe).
  3. Calculer sur toute la base et sauvegarder (calcul parallélisé avec `multiprocessing` ou `joblib`).
  4. Prévoir le **même extracteur pour les images envoyées par l'utilisateur** à l'exécution.
  5. *Optionnel* : calculer aussi ces descripteurs sur les vignettes MIRFLICKR, pour disposer d'un descripteur couleur sur cette base (absent des fichiers fournis).
- **Outils** : scikit-image, OpenCV, NumPy, joblib.
- **Contexte à donner à Claude** : l'interface commune, la taille standard des images, les paramètres souhaités (nombre de bins, etc.).
- **Livrable** : modules `color_hist`, `lbp`, `hog` et fichiers `.npy`.
- **Validation** : deux images visuellement proches ont une distance plus faible que deux images différentes (test sur quelques exemples).

### Étape 5b : Extraction des descripteurs VGG16 (Deep Learning) - RECOMMANDÉ

- **Objectif** : extraire des descripteurs de haute qualité (512 dimensions) avec un réseau pré-entraîné sur ImageNet.
- **Justification** : les descripteurs MPEG-7 (Edge Histogram, Homogeneous Texture) sont peu performants pour la similarité sémantique. VGG16 offre une similarité visuelle bien supérieure (testé et validé).
- **Actions** :
  1. Charger VGG16 pré-entraîné (ImageNet) via PyTorch/torchvision.
  2. Utiliser les features convolutionnels + Global Average Pooling → **512 dimensions** (plus compact que 4096-d FC).
  3. Prétraiter les vignettes 64×64 : upscale à 224×224, normalisation ImageNet (mean, std).
  4. Extraire par batches (GPU recommandé, CPU possible mais lent : ~2-3 jours pour 1M).
  5. L2-normaliser tous les vecteurs (distance euclidienne = similarité cosinus).
  6. Sauvegarder dans `data/features/vgg16_gap512_1M.npy` (~2 GB).
  7. Sauvegarder les poids du modèle dans `data/models/vgg16_features_imagenet.pth` pour usage hors ligne.
- **Outils** : **PyTorch**, **torchvision**, NumPy, GPU (optionnel mais recommandé).
- **Contexte à donner à Claude** : le code déjà écrit dans `backend/app/descriptors/vgg.py` et `backend/scripts/05_vgg_similar.py`.
- **Livrable** : `vgg16_gap512_1M.npy`, `vgg16_features_imagenet.pth`, script `05_extract_vgg.py` (si nouveau), module `backend/app/descriptors/vgg.py`.
- **Validation** : rechercher quelques images et vérifier que les résultats sont visuellement cohérents (bien meilleurs qu'avec MPEG-7). Script de test : `python backend/scripts/05_vgg_similar.py <image_id>`.

### Étape 6 : Distances et recherche par similarité (version de référence)

- **Objectif** : une recherche correcte, même lente, qui servira de référence.
- **Actions** :
  1. Implémenter les distances (euclidienne, L1, cosinus, chi²).
  2. Implémenter la recherche exhaustive (« brute force »), vectorisée avec NumPy.
  3. Gérer la **combinaison de plusieurs descripteurs** : normalisation des distances puis somme pondérée.
- **Outils** : NumPy, SciPy (`scipy.spatial.distance`).
- **Contexte à donner à Claude** : liste des descripteurs et leurs dimensions, formule de fusion souhaitée.
- **Livrable** : `distances.py`, `visual_search.py` (version simple).
- **Validation** : une image requête retrouve en premier rang elle-même (distance 0) et des images cohérentes ensuite.

### Étape 7 : Indexation pour réduire le temps de recherche

- **Objectif** : respecter l'exigence « temps de recherche optimal ».
- **Actions** :
  1. Implémenter le **VP-Tree** (construction, recherche des k plus proches voisins) pour un descripteur (démonstration sur ≤100k images).
  2. Implémenter **FAISS** (Flat index) pour la base complète de 1M (production, temps de requête ~1-10ms).
  3. Sérialiser les index sur disque et les charger au démarrage du serveur.
  4. **Comparer** : temps de la recherche exhaustive, du VP-Tree et de FAISS sur 100 requêtes, à plusieurs tailles de base (10k, 100k, 1M si possible) (tableau pour le rapport).
  5. Construire les index pour **VGG16** (recommandé), Edge Histogram et Homogeneous Texture.
- **Outils / sources** : Python, FAISS (`faiss-cpu`), scikit-learn, `joblib`/`pickle` ; idées d'indexation dans https://github.com/MKLab-ITI/multimedia-indexing (code Java, donc source d'inspiration uniquement).
- **Contexte à donner à Claude** : volumétrie (nombre de vecteurs, dimension), distance utilisée, mémoire disponible sur le PC cible.
- **Livrable** : `vptree.py`, `faiss_index.py`, `index_manager.py`, script `03_build_indexes.py`, tableau comparatif.
- **Validation** : les résultats de FAISS correspondent à ceux de la recherche exhaustive (même top-k, recall@k = 100%) et le temps de recherche est nettement inférieur.

### Étape 8 : Base de données et recherche textuelle

- **Objectif** : stocker les URLs et métadonnées et permettre la recherche par mots-clés.
- **Actions** :
  1. Créer la collection MongoDB, ses index (identifiant, mots-clés) et charger les métadonnées (script `04_fill_database.py`) : chemins des vignettes, URLs Flickr, licences, tags.
  2. Implémenter la recherche par mots-clés (normalisation, minuscules, pluriels éventuels).
  3. Implémenter le **mode ET/OU** entre plusieurs mots-clés.
  4. Implémenter la **fusion texte + visuel** : filtrer par mots-clés puis trier par similarité, ou combiner les scores.
- **Outils** : MongoDB, pymongo (Elasticsearch seulement en alternative).
- **Contexte à donner à Claude** : schéma, exemples de requêtes attendues, règle de fusion souhaitée.
- **Livrable** : `db/`, `text_search.py`, `fusion.py`.
- **Validation** : des requêtes test (un mot, deux mots en ET, deux mots en OU, mot + image) donnent des résultats cohérents.

### Étape 9 : API FastAPI

- **Objectif** : exposer les fonctionnalités via des endpoints JSON.
- **Actions** :
  1. Définir les schémas de requête et de réponse (section 2.4).
  2. Implémenter les routes `/search`, `/descriptors`, `/images`, `/health`.
  3. Servir les vignettes comme fichiers statiques.
  4. Gérer l'upload d'une image requête externe (optionnel) en plus du choix d'une image de la base.
  5. Configurer CORS pour autoriser le domaine du frontend.
  6. Charger les index et descripteurs **une seule fois au démarrage**.
  7. *Préparer Docker dès maintenant* : lire l'adresse MongoDB dans une variable d'environnement (`MONGO_URI`, valeur par défaut `mongodb://localhost:27017`) et le dossier des données dans une autre (`DATA_DIR`), sans chemin ni `localhost` écrits en dur.
- **Outils** : FastAPI, Uvicorn, Pydantic ; documentation automatique via `/docs` (Swagger).
- **Contexte à donner à Claude** : contrat d'API, services déjà écrits, format exact des réponses attendues par le client.
- **Livrable** : serveur fonctionnel en local.
- **Validation** : tous les endpoints sont testables depuis `/docs` et renvoient les résultats attendus.

### Étape 10 : Client web

- **Objectif** : interface qui couvre toutes les fonctionnalités minimales.
- **Éléments d'interface** :
  1. Champ de mots-clés et choix du mode ET/OU.
  2. Sélection de l'image requête (galerie de la base ou résultat précédent, éventuellement upload).
  3. Choix du ou des descripteurs (cases à cocher), de la distance et du nombre de résultats.
  4. Affichage de l'image requête puis des résultats triés avec leur score. Les vignettes MIRFLICKR étant petites, prévoir un agrandissement au survol ou au clic (CSS `image-rendering`/zoom) ; ouvrir l'image complète (`Photo url`) ou la page Flickr (`Web url`) seulement si ces champs existent pour l'image ; sinon masquer le lien.
  5. Clic sur un résultat pour relancer une recherche.
  6. États de chargement et messages d'erreur.
- **Outils** : HTML/CSS/JS avec `fetch` ; **Streamlit** comme plan B très rapide.
- **Contexte à donner à Claude** : contrat d'API, maquette en mots (disposition des zones), contraintes de style.
- **Livrable** : dossier `frontend/` fonctionnel contre le serveur local.
- **Validation** : scénario complet réussi : mots-clés → résultats → clic sur un résultat → nouvelle recherche.

### Étape 11 : Tests et évaluation

- **Objectif** : vérifier la qualité et la rapidité, et avoir des chiffres pour le rapport.
- **Actions** :
  1. Tests unitaires de base (descripteurs, distances, fusion, API) avec `pytest`.
  2. Évaluation de la pertinence : précision@k sur des requêtes de la base Objects (en utilisant les classes comme vérité terrain).
  3. Comparaison des descripteurs et de leurs combinaisons (sur MIRFLICKR : Edge Histogram, Homogeneous Texture, et leur combinaison).
  4. Mesure des temps de réponse.
- **Outils** : pytest, NumPy, Matplotlib.
- **Contexte à donner à Claude** : liste des classes, protocole d'évaluation souhaité.
- **Livrable** : tests, tableau et courbes de résultats.
- **Validation** : les mesures sont reproductibles par script.

### Étape 12 : Déploiement de la démo

- **Objectif** : une démonstration fonctionnelle, conforme aux livrables d'évaluation.
- **Actions** :
  1. **Cible principale : PC LIASA.** Installer l'environnement (et MongoDB, ou restaurer la base avec `mongodump`/`mongorestore`), copier le code, la base de données, les index, les descripteurs et les vignettes ; préparer un script de lancement unique (backend + frontend).
  2. Tester le scénario complet sur cette machine (mémoire, chemins, ports), idéalement avant la date de démo.
  3. *Optionnel (bonus) : Docker.* `Dockerfile` pour le backend, `docker-compose.yml` avec un service `mongo` (`MONGO_URI=mongodb://mongo:27017`), le dossier `data/` monté en **volume** (jamais copié dans l'image), `requirements.txt` généré avec `pip freeze`. Le mode principal de la démo reste Python local, car Docker n'est peut-être pas installé sur le PC LIASA.
  4. *Optionnel (bonus)* : version en ligne limitée à un sous-ensemble (backend sur Render, frontend sur Vercel ou Cloudflare Pages, images sur Cloudflare R2), en gardant le calcul lourd sur le backend.
- **Points d'attention** : taille des index et vignettes à copier sur le PC LIASA, versions de Python et des bibliothèques identiques, chemins relatifs dans la configuration. Pour la version en ligne : mise en veille du plan gratuit de Render (premier appel lent) et mémoire limitée (environ 512 Mo).
- **Contexte à donner à Claude** : caractéristiques du PC LIASA (OS, RAM, Python disponible), taille de l'index, fichiers de configuration actuels.
- **Livrable** : démo qui tourne sur le PC LIASA ; export de la base MongoDB (`mongodump`) fourni avec le code ; optionnellement une URL publique.
- **Validation** : le scénario complet de l'étape 10 fonctionne sur le PC LIASA.

### Étape 13 : Documentation et rapport

- **Objectif** : rendre le projet présentable (le rapport fait partie des livrables).
- **Contenu** : README (installation, lancement, déploiement), schéma d'architecture, description des données et des descripteurs (dont la règle id = n − 1 et les limites : vignettes seulement, pas de descripteur couleur fourni), tableaux de comparaison de l'étape 7 et de l'étape 11, méthodologie et choix de conception, limites connues et pistes d'amélioration, captures d'écran.
- **Sources utiles** : cbir.info (articles), documentation MPEG-7.
- **Contexte à donner à Claude** : résultats chiffrés, captures et décisions prises.
- **Livrable** : README, rapport final, code source et base de données.

---

## 4. Planning indicatif

| Rythme | Étapes | Durée |
|---|---|---|
| **Sprint 2 jours (MVP)** | Jour 1 : étapes 1 à 9 (moteur + API) ; Jour 2 : étapes 10 à 13 (client, tests rapides, déploiement, README) | 2 jours |
| **Rythme confortable** | Même ordre, avec tests complets et rapport détaillé | 3 à 4 jours |

Éléments à lancer en parallèle : exploration des données (étape 2) pendant que tu avances sur l'architecture et l'API. Prévoir une marge pour les calculs longs sur 1 M d'images (parsing, index).

---

## 5. Méthode pour guider Claude

1. **Un prompt par étape**, jamais tout le projet d'un coup.
2. **Toujours fournir** : l'objectif de l'étape, l'arborescence, les contrats d'API, les fichiers déjà produits et un échantillon réel de données.
3. **Fixer les contraintes** dans chaque prompt : Python 3.11, bibliothèques autorisées, code simple et commenté, pas de sur-optimisation.
4. **Tester chaque module** sur 1 000 images avant de passer à l'échelle.
5. **Coller les erreurs complètes** (trace entière) en cas de blocage.
6. **Demander à Claude de vérifier** les critères de validation de l'étape avant de passer à la suivante.
7. **Garder un fichier `DECISIONS.md`** avec les choix déjà faits (formats, noms de fichiers, paramètres, règle id = n − 1) pour le recoller en début de conversation.

### Modèle de prompt pour démarrer une étape

```
Contexte du projet : [coller la section 0 et l'arborescence]
Étape en cours : [numéro et titre]
Déjà réalisé : [liste des fichiers/modules terminés]
Données disponibles : [exemple réel, formats]
Contraintes : [outils autorisés, simplicité, mémoire]
Demande : [objectif de l'étape]
Livrable attendu : [fichiers à produire]
Critère de validation : [comment on vérifie que c'est réussi]
```

---

## 6. Ressources

| Ressource | Usage |
|---|---|
| MIRFLICKR (https://press.liacs.nl/mirflickr/) | Vignettes, descripteurs MPEG-7, tags, licences |
| savvash.blogspot.com (descripteurs MPEG-7) | Comprendre les descripteurs |
| cbir.info/articles | Théorie du CBIR pour le rapport |
| semanticmetadata.net/lire | Référence Java (format et descripteurs) |
| github.com/MKLab-ITI/multimedia-indexing | Idées d'indexation à grande échelle |
| multimedia-metadata.info, ee.cityu.edu.hk/~mirror | Outils et références complémentaires |
| image-net.org | Optionnel, non nécessaire pour ce projet |
| Anaconda | Environnement Python |

> Plusieurs de ces liens sont anciens : certains peuvent être indisponibles, prévois de passer par l'Internet Archive si besoin.
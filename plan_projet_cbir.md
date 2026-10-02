# Plan de développement : moteur d'indexation et de recherche d'images Textuel/Visuel (CBIR)

> Document de pilotage pour guider Claude étape par étape. Chaque étape contient : objectif, outils, contexte à fournir à Claude, livrable et critère de validation. Aucun code n'est inclus.

---

## 0. Résumé du cahier des charges

**Objectif** : une application client/serveur de recherche d'images par mots-clés et/ou par similarité visuelle.

**Deux jeux de données** :
1. **MIRFLICKR-1M** : 1 million d'images, avec descripteurs MPEG-7 fournis (rien à calculer).
2. **Base « Objects »** : images pour lesquelles les descripteurs sont à calculer (couleur, LBP, HOG, etc.).

**Fonctionnalités minimales** :
- Requête par mots-clés **ET/OU** par image (image de la base ou image issue d'un résultat précédent).
- L'utilisateur choisit le ou les descripteurs utilisés par le serveur.
- Le serveur calcule/utilise les descripteurs, cherche les images similaires et renvoie la liste des URLs.
- Affichage au client : image requête + images similaires triées par ressemblance.
- Relance d'une recherche en cliquant sur une image résultat.
- Temps de recherche optimisé grâce à une structure d'index (VP-Tree ou équivalent).

**Contraintes** : client = application web ; serveur = Python (FastAPI recommandé) ; échange en JSON ; images stockées dans des répertoires, seules les URLs sont dans la base de données ; déploiement simple (Render + Vercel/Cloudflare Pages), code non optimisé à 100 %.

---

## 1. Décisions techniques (à figer avant de commencer)

| Composant | Choix recommandé | Alternative |
|---|---|---|
| Langage serveur | Python 3.11+ | – |
| Framework API | **FastAPI** + Uvicorn | Flask, Django |
| Calcul numérique | **NumPy**, **SciPy** | – |
| Descripteurs image | **scikit-image** (LBP, HOG), **OpenCV** (histogrammes couleur) | Pillow pour le chargement |
| Index de similarité | **VP-Tree** (exigé par le sujet) pour la démonstration + **FAISS** ou `scikit-learn NearestNeighbors` pour la vitesse | `scipy.spatial.cKDTree` |
| Base de données | **SQLite** (simple) ou **MongoDB Atlas** (gratuit) | Elasticsearch (lourd, à éviter en gratuit) |
| Recherche texte | Index inversé maison ou **SQLite FTS5** | Elasticsearch, MongoDB text index |
| Client | **HTML + CSS + JavaScript** (vanilla) ou **Streamlit** si le temps manque | React |
| Packaging | **Docker** (optionnel) | – |
| Hébergement | Backend sur **Render**, frontend sur **Vercel** ou **Cloudflare Pages**, images sur **Cloudflare R2** | – |
| Environnement dev | **Anaconda/venv** + **Jupyter** pour les essais | – |

**Décision à prendre tôt** : sous-ensemble utilisé pour le développement et la démo en ligne (conseil : 10 000 à 50 000 images). La base complète de 1M reste en local.

---

## 2. Architecture du projet

### 2.1 Vue d'ensemble

```
┌────────────────┐   requête (mots-clés,    ┌─────────────────────────────┐
│  Client web    │   image, descripteurs)   │  Serveur FastAPI (Render)    │
│ (Vercel / CF   │ ───────────────────────► │                              │
│  Pages)        │                          │  ┌───────────────────────┐   │
│                │ ◄─────────────────────── │  │ Routes API (JSON)     │   │
│ HTML/CSS/JS    │   liste d'URLs triées    │  ├───────────────────────┤   │
└────────────────┘   + scores               │  │ Service de recherche  │   │
                                            │  │  - texte              │   │
                                            │  │  - visuel             │   │
                                            │  │  - fusion ET/OU       │   │
                                            │  ├───────────────────────┤   │
                                            │  │ Extracteurs de        │   │
                                            │  │ descripteurs          │   │
                                            │  ├───────────────────────┤   │
                                            │  │ Index (VP-Tree/FAISS) │   │
                                            │  └──────────┬────────────┘   │
                                            └─────────────┼────────────────┘
                                                          │
                          ┌───────────────────────────────┼─────────────────┐
                          │                               │                 │
                  ┌───────▼────────┐            ┌─────────▼───────┐  ┌──────▼───────┐
                  │ Base de données│            │ Fichiers index  │  │ Stockage     │
                  │ (SQLite/Mongo) │            │ (.npy, .pkl)    │  │ des images   │
                  │ URLs + mots-clés│           │ descripteurs    │  │ (dossier/R2) │
                  └────────────────┘            └─────────────────┘  └──────────────┘
```

### 2.2 Deux phases bien séparées

1. **Phase hors ligne (une seule fois)** : parsing/extraction des descripteurs, construction des index, remplissage de la base de données.
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
│   │   ├── images/                 (sous-ensemble ou base complète)
│   │   └── descriptors_mpeg7/      (fichiers fournis)
│   ├── objects/
│   │   └── images/
│   ├── features/                   (descripteurs au format .npy)
│   └── indexes/                    (index sérialisés)
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                 (point d'entrée FastAPI, CORS)
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
│   │   │   ├── mpeg7_parser.py     (lecture des fichiers fournis)
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

### 2.4 Contrat d'API (à définir avant le code)

| Endpoint | Rôle | Entrées | Sortie |
|---|---|---|---|
| `GET /descriptors` | Liste des descripteurs disponibles par base | – | noms, dimension, bases concernées |
| `GET /images` | Parcourir/paginer les images de la base | page, base | URLs + mots-clés |
| `POST /search` | Recherche texte, image ou mixte | mots-clés, mode ET/OU, id ou fichier image, descripteurs choisis, nombre de résultats, distance | image requête + liste triée (URL, score, mots-clés) |
| `GET /health` | Vérification du déploiement | – | statut |

### 2.5 Modèle de données

- **Table `images`** : identifiant, base d'origine (mirflickr/objects), URL ou chemin relatif, largeur/hauteur.
- **Table `keywords`** et **table de liaison** image ↔ mot-clé (ou un champ liste de mots-clés si MongoDB).
- **Fichiers de descripteurs** : un fichier par descripteur et par base, où la ligne *i* correspond à l'identifiant *i* de l'image.
- **Fichiers d'index** : un par descripteur (et éventuellement un par combinaison).

---

## 3. Étapes de développement

### Étape 1 : Cadrage et préparation de l'environnement

- **Objectif** : projet initialisé, outils installés, données en cours de téléchargement.
- **Actions** :
  1. Créer le dépôt Git et l'arborescence ci-dessus.
  2. Créer l'environnement (Anaconda ou venv) et installer les dépendances de base.
  3. **Lancer tout de suite le téléchargement de MIRFLICKR** (descripteurs d'abord, puis une ou deux archives d'images seulement).
  4. Télécharger la base Objects.
- **Outils / sources** : Anaconda, Git, Jupyter, MIRFLICKR (https://press.liacs.nl/mirflickr/ : l'URL du sujet contient une faute, `.ntml` au lieu de `.html`), pages de la base Objects fournies par le professeur.
- **Contexte à donner à Claude** : le cahier des charges (captures), l'arborescence, les choix techniques de la section 1, ton niveau et ton système d'exploitation.
- **Livrable** : projet qui s'installe et s'exécute, fichier `requirements.txt`, README de départ.
- **Validation** : l'environnement se recrée depuis zéro avec une seule commande.

### Étape 2 : Exploration des données

- **Objectif** : comprendre exactement ce qui est fourni avant d'écrire le moindre module.
- **Actions** :
  1. Lister les descripteurs MPEG-7 fournis (type, dimension, format du fichier) et vérifier le mapping entre un fichier de descripteurs et son image.
  2. Examiner la base Objects : nombre d'images, classes, taille, format, annotations disponibles.
  3. Vérifier la présence de mots-clés/annotations (sinon prévoir une source : noms de dossiers/classes, tags MIRFLICKR).
  4. Décider du sous-ensemble de travail.
- **Outils / sources** : Jupyter, Pandas, Pillow, documentation MPEG-7 (savvash.blogspot.com, cbir.info), documentation de MIRFLICKR.
- **Contexte à donner à Claude** : les **premières lignes d'un fichier de descripteurs**, la **liste des dossiers** et la documentation du format.
- **Livrable** : notebook d'exploration et note courte « données » (formats, volumes, descripteurs retenus).
- **Validation** : tu peux afficher une image avec son vecteur de descripteur et ses mots-clés.

### Étape 3 : Préparation du jeu de données

- **Objectif** : un jeu propre et indexé par identifiants.
- **Actions** :
  1. Attribuer un identifiant numérique unique à chaque image (une liste ordonnée sert de référence pour tout le projet).
  2. Construire le sous-ensemble (échantillonnage aléatoire reproductible).
  3. Générer des miniatures (par exemple 256 px) pour accélérer l'affichage et alléger le déploiement.
  4. Produire le fichier de métadonnées (identifiant, chemin, base, mots-clés).
- **Outils** : Python, Pillow, Pandas.
- **Contexte à donner à Claude** : résultats de l'étape 2 et le modèle de données (section 2.5).
- **Livrable** : script `01_prepare_dataset.py` et fichier de métadonnées.
- **Validation** : le nombre d'images, d'identifiants et de lignes de métadonnées est identique.

### Étape 4 : Lecture des descripteurs MPEG-7 (MIRFLICKR)

- **Objectif** : transformer les descripteurs fournis en matrices NumPy exploitables.
- **Actions** :
  1. Écrire le module de lecture (`mpeg7_parser`) selon le format réel des fichiers (XML ou binaire/texte).
  2. Normaliser chaque descripteur (taille fixe, type numérique, éventuelle normalisation).
  3. Sauvegarder une matrice par descripteur, alignée sur les identifiants.
- **Outils / sources** : Python, `xml.etree` ou `lxml`, NumPy ; référence MPEG-7 (savvash.blogspot.com), LIRE (http://www.semanticmetadata.net/lire/) uniquement comme référence de lecture du format.
- **Contexte à donner à Claude** : exemple réel d'un fichier, description du format de chaque descripteur.
- **Livrable** : module `mpeg7_parser`, script `02_extract_descriptors.py` (partie MIRFLICKR), fichiers `.npy`.
- **Validation** : dimensions attendues du descripteur respectées, aucune valeur manquante, vérification manuelle sur 5 images.

### Étape 5 : Calcul des descripteurs de la base Objects

- **Objectif** : calculer au moins 3 descripteurs (couleur, texture LBP, forme/contours HOG).
- **Actions** :
  1. Définir une interface commune : image en entrée → vecteur de taille fixe en sortie.
  2. Implémenter l'histogramme de couleur (espace HSV, quantification), LBP (histogramme uniforme) et HOG (redimensionnement fixe).
  3. Calculer sur toute la base et sauvegarder (calcul parallélisé avec `multiprocessing` ou `joblib`).
  4. Prévoir le **même extracteur pour les images envoyées par l'utilisateur** à l'exécution.
- **Outils** : scikit-image, OpenCV, NumPy, joblib.
- **Contexte à donner à Claude** : l'interface commune, la taille standard des images, les paramètres souhaités (nombre de bins, etc.).
- **Livrable** : modules `color_hist`, `lbp`, `hog` et fichiers `.npy`.
- **Validation** : deux images visuellement proches ont une distance plus faible que deux images différentes (test sur quelques exemples).

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
  1. Implémenter le **VP-Tree** (construction, recherche des k plus proches voisins) pour un descripteur.
  2. Ajouter une alternative rapide (FAISS ou `NearestNeighbors`) pour la version déployée.
  3. Sérialiser les index sur disque et les charger au démarrage du serveur.
  4. **Comparer** : temps de la recherche exhaustive, du VP-Tree et de FAISS sur 100 requêtes (tableau pour le rapport).
- **Outils / sources** : Python, FAISS (`faiss-cpu`), scikit-learn, `joblib`/`pickle` ; idées d'indexation dans https://github.com/MKLab-ITI/multimedia-indexing (code Java, donc source d'inspiration uniquement).
- **Contexte à donner à Claude** : volumétrie (nombre de vecteurs, dimension), distance utilisée, contrainte de mémoire de Render (environ 512 Mo en gratuit).
- **Livrable** : `vptree.py`, `faiss_index.py`, `index_manager.py`, script `03_build_indexes.py`, tableau comparatif.
- **Validation** : les résultats du VP-Tree correspondent à ceux de la recherche exhaustive (même top-k) et le temps de recherche est nettement inférieur.

### Étape 8 : Base de données et recherche textuelle

- **Objectif** : stocker les URLs et permettre la recherche par mots-clés.
- **Actions** :
  1. Créer le schéma et charger les métadonnées (script `04_fill_database.py`).
  2. Implémenter la recherche par mots-clés (normalisation, minuscules, pluriels éventuels).
  3. Implémenter le **mode ET/OU** entre plusieurs mots-clés.
  4. Implémenter la **fusion texte + visuel** : filtrer par mots-clés puis trier par similarité, ou combiner les scores.
- **Outils** : SQLite (FTS5) ou MongoDB Atlas, SQLAlchemy (si SQL), PyMongo (si Mongo).
- **Contexte à donner à Claude** : schéma, exemples de requêtes attendues, règle de fusion souhaitée.
- **Livrable** : `db/`, `text_search.py`, `fusion.py`.
- **Validation** : des requêtes test (un mot, deux mots en ET, deux mots en OU, mot + image) donnent des résultats cohérents.

### Étape 9 : API FastAPI

- **Objectif** : exposer les fonctionnalités via des endpoints JSON.
- **Actions** :
  1. Définir les schémas de requête et de réponse (section 2.4).
  2. Implémenter les routes `/search`, `/descriptors`, `/images`, `/health`.
  3. Gérer l'upload d'une image requête externe (optionnel) en plus du choix d'une image de la base.
  4. Configurer CORS pour autoriser le domaine du frontend.
  5. Charger les index et descripteurs **une seule fois au démarrage**.
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
  4. Affichage de l'image requête puis des résultats triés avec leur score.
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
  3. Comparaison des descripteurs et de leurs combinaisons.
  4. Mesure des temps de réponse.
- **Outils** : pytest, NumPy, Matplotlib.
- **Contexte à donner à Claude** : liste des classes, protocole d'évaluation souhaité.
- **Livrable** : tests, tableau et courbes de résultats.
- **Validation** : les mesures sont reproductibles par script.

### Étape 12 : Déploiement

- **Objectif** : une démonstration en ligne.
- **Actions** :
  1. Réduire le déploiement au sous-ensemble (index et miniatures légers).
  2. Conteneuriser le backend avec Docker (optionnel) et le déployer sur **Render**.
  3. Héberger les images sur **Cloudflare R2** (ou dans le dépôt si le volume est faible).
  4. Déployer le frontend sur **Vercel** ou **Cloudflare Pages** en pointant vers l'URL du backend.
  5. Régler les variables d'environnement et CORS.
- **Points d'attention** : mise en veille du plan gratuit de Render (premier appel lent), mémoire limitée, taille des images, limites de Vercel pour le calcul lourd (garder le calcul sur Render).
- **Contexte à donner à Claude** : plan et limites de chaque service, taille de l'index, fichiers de configuration actuels.
- **Livrable** : URL publique fonctionnelle.
- **Validation** : le scénario complet de l'étape 10 fonctionne en ligne.

### Étape 13 : Documentation et rapport

- **Objectif** : rendre le projet présentable.
- **Contenu** : README (installation, lancement, déploiement), schéma d'architecture, description des descripteurs, tableaux de comparaison de l'étape 7 et de l'étape 11, limites connues et pistes d'amélioration, captures d'écran.
- **Sources utiles** : cbir.info (articles), documentation MPEG-7.
- **Contexte à donner à Claude** : résultats chiffrés, captures et décisions prises.
- **Livrable** : README et rapport final.

---

## 4. Planning indicatif

| Rythme | Étapes | Durée |
|---|---|---|
| **Sprint 2 jours (MVP)** | Jour 1 : étapes 1 à 9 (moteur + API) ; Jour 2 : étapes 10 à 13 (client, tests rapides, déploiement, README) | 2 jours |
| **Rythme confortable** | Même ordre, avec tests complets et rapport détaillé | 3 à 4 jours |

Éléments à lancer en parallèle : téléchargement des données (étape 1) pendant que tu avances sur l'architecture et l'API.

---

## 5. Méthode pour guider Claude

1. **Un prompt par étape**, jamais tout le projet d'un coup.
2. **Toujours fournir** : l'objectif de l'étape, l'arborescence, les contrats d'API, les fichiers déjà produits et un échantillon réel de données.
3. **Fixer les contraintes** dans chaque prompt : Python 3.11, bibliothèques autorisées, code simple et commenté, pas de sur-optimisation.
4. **Tester chaque module** sur 1 000 images avant de passer à l'échelle.
5. **Coller les erreurs complètes** (trace entière) en cas de blocage.
6. **Demander à Claude de vérifier** les critères de validation de l'étape avant de passer à la suivante.
7. **Garder un fichier `DECISIONS.md`** avec les choix déjà faits (formats, noms de fichiers, paramètres) pour le recoller en début de conversation.

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
| MIRFLICKR (https://press.liacs.nl/mirflickr/) | Images et descripteurs MPEG-7 |
| savvash.blogspot.com (descripteurs MPEG-7) | Comprendre les descripteurs |
| cbir.info/articles | Théorie du CBIR pour le rapport |
| semanticmetadata.net/lire | Référence Java (format et descripteurs) |
| github.com/MKLab-ITI/multimedia-indexing | Idées d'indexation à grande échelle |
| multimedia-metadata.info, ee.cityu.edu.hk/~mirror | Outils et références complémentaires |
| image-net.org | Optionnel, non nécessaire pour ce projet |
| Anaconda | Environnement Python |

> Plusieurs de ces liens sont anciens : certains peuvent être indisponibles, prévois de passer par l'Internet Archive si besoin.

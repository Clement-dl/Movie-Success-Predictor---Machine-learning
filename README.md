# 🎬 Movie Success Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

**Un système de prédiction du succès de films basé sur le Machine Learning avec interface graphique moderne**

[Fonctionnalités](#-fonctionnalités) •
[Installation](#-installation) •
[Utilisation](#-utilisation) •
[Méthodologie](#-méthodologie) •
[Résultats](#-résultats)

</div>

---

## 📋 Table des matières

- [À propos](#-à-propos)
- [Fonctionnalités](#-fonctionnalités)
- [Démo](#-démo)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Méthodologie](#-méthodologie)
- [Architecture du projet](#-architecture-du-projet)
- [Résultats](#-résultats)
- [Technologies utilisées](#-technologies-utilisées)
- [Licence](#-licence)

---

## 🎯 À propos

Ce projet implémente un **système de prédiction du succès de films** en utilisant des algorithmes de Machine Learning. Basé sur le dataset TMDB 5000, il prédit la probabilité de succès d'un film **avant sa sortie**, en se basant uniquement sur des caractéristiques disponibles en pré-production.

### 🎓 Contexte académique

Ce projet a été développé dans le cadre d'un mini-projet académique de Machine Learning. Il démontre :
- La maîtrise du pipeline complet de Data Science
- L'application rigoureuse des bonnes pratiques (évitement du data leakage)
- Le développement d'une interface utilisateur professionnelle
- La documentation et la reproductibilité du code

### 🔍 Problématique

**Comment prédire le succès commercial et critique d'un film avant sa sortie ?**

Le modèle compare les caractéristiques du film à des milliers de films similaires déjà sortis, puis estime la probabilité de succès en se basant sur des tendances observées dans les données historiques.

---

## ✨ Fonctionnalités

### 🤖 Machine Learning
- ✅ **Classification binaire** (SUCCESS/FAILURE)
- ✅ **Régression logistique** avec optimisation des hyperparamètres
- ✅ **Optimisation du seuil de décision** via GroupKFold cross-validation
- ✅ **Pipeline sklearn complet** : normalisation + one-hot encoding
- ✅ **Prévention du data leakage** : seules les features pré-sortie sont utilisées

### 🖥️ Interface graphique
- ✅ **Interface Tkinter moderne** avec design professionnel
- ✅ **Formulaire scrollable** pour saisir toutes les caractéristiques du film
- ✅ **Affichage en temps réel** de la probabilité de succès
- ✅ **Barre de progression visuelle** et indicateurs colorés
- ✅ **Détails complets** des inputs utilisés par le modèle

### 📊 Features d'entraînement

#### Variables numériques (9)
- `budget` : Budget du film (USD)
- `runtime` : Durée du film (minutes)
- `release_year` : Année de sortie
- `release_month` : Mois de sortie
- `num_genres` : Nombre de genres
- `num_production_companies` : Nombre de sociétés de production
- `cast_size` : Taille du casting
- `crew_size` : Taille de l'équipe technique
- `is_english` : Film en anglais (0/1)

#### Variables catégorielles (4)
- `genre_group` : Genre principal (Top 20 + Other)
- `company_group` : Société de production principale (Top 80 + Other)
- `lang_group` : Langue originale (Top 30 + Other)
- `director_group` : Réalisateur (Top 80 + Other)

---

## 🎬 Démo

### Interface principale

```
┌─────────────────────────────────────────────────────────────┐
│  🎬 Movie Success Predictor                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Inputs                          │  Result                  │
│  ─────────────────────────────   │  ─────────────────────  │
│                                   │                          │
│  Budget:     [50000000]           │  SUCCESS ✅              │
│  Runtime:    [120]                │                          │
│  Year:       [2025]               │  Probability: 0.823      │
│  Month:      [7]                  │  (82%)                   │
│  Genres:     [2]                  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░   │
│  Companies:  [1]                  │                          │
│  Cast Size:  [10]                 │  Threshold: 0.50         │
│  Crew Size:  [50]                 │                          │
│                                   │  Inputs:                 │
│  Genre:      [Action ▼]           │    budget=50000000       │
│  Company:    [Other ▼]            │    runtime=120           │
│  Language:   [en ▼]               │    cast_size=10          │
│  Director:   [Other ▼]            │    ...                   │
│                                   │                          │
│  [Reset]  [Predict]               │                          │
└─────────────────────────────────────────────────────────────┘
```

### Exemple de prédiction

**Film A - Budget modeste**
```
Budget: 5M USD
Cast: 5
Crew: 20
Genre: Drama
→ Probabilité: 25% → FAILURE ❌
```

**Film B - Blockbuster**
```
Budget: 120M USD
Cast: 60
Crew: 600
Genre: Action
Language: English
→ Probabilité: 85% → SUCCESS ✅
```

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/movie-success-predictor.git
cd movie-success-predictor
```

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Vérifier la présence des données**

Assurez-vous que les fichiers suivants sont présents dans le répertoire :
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

> **Note** : Ces fichiers sont disponibles sur [Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## 💻 Utilisation

### Lancer l'application graphique

```bash
python movie_success_app_scroll.py
```

Au premier lancement, l'application va :
1. Charger les datasets TMDB
2. Entraîner le modèle de Machine Learning
3. Sauvegarder les artefacts (`movie_success_model.joblib` et `movie_success_meta.json`)

Les lancements suivants seront **instantanés** car le modèle est chargé depuis le cache.

### Utiliser le notebook Jupyter

Pour explorer l'analyse complète et les expérimentations :

```bash
jupyter notebook tmdb_movie_success_classification_template_v7_threshold_optimization.ipynb
```

Le notebook contient :
- L'analyse exploratoire des données (EDA)
- La construction des features
- L'entraînement avec GridSearchCV
- L'optimisation du seuil de décision
- Les métriques de performance complètes

---

## 🔬 Méthodologie

### 1. Construction de la variable cible

Le succès d'un film est défini par un **score composite** calculé uniquement lors de l'entraînement :

```python
FilmSuccessScore = 0.4 × log(profit + 1)      # 40% poids
                 + 0.3 × (vote_average / 10)   # 30% poids
                 + 0.2 × log(vote_count + 1)   # 20% poids
                 + 0.1 × log(popularity + 1)   # 10% poids
```

Un film est considéré comme **SUCCESS** si son score ≥ médiane, sinon **FAILURE**.

> **⚠️ Point crucial** : Les variables `profit`, `vote_average`, `vote_count`, et `popularity` ne sont **jamais utilisées comme features d'entrée**, car elles ne sont disponibles qu'après la sortie du film. Cela évite le **data leakage**.

### 2. Pipeline de preprocessing

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])),
    ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs'))
])
```

### 3. Optimisation des hyperparamètres

**GridSearchCV** avec les paramètres suivants :
```python
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l2'],
    'classifier__class_weight': [None, 'balanced']
}
```

**Validation** : GroupKFold (5 splits) groupé par réalisateur pour éviter le leakage temporel.

### 4. Optimisation du seuil de décision

Au lieu d'utiliser le seuil par défaut (0.5), le seuil optimal est déterminé par :
1. Calcul des **probabilités out-of-fold** sur le train set via GroupKFold
2. Test de 19 seuils entre 0.05 et 0.95
3. Sélection du seuil maximisant le **F1-score pondéré**

Cette approche améliore significativement les performances sans modifier le modèle.

### 5. Évaluation finale

Métriques sur le **test set** (20% des données) :
- **F1-score pondéré**
- **Accuracy**
- **ROC-AUC**
- **Matrice de confusion**
- **Courbe ROC**

---

## 📁 Architecture du projet

```
movie-success-predictor/
│
├── 📊 Data
│   ├── tmdb_5000_movies.csv          # Dataset principal (4803 films)
│   └── tmdb_5000_credits.csv         # Casting et équipe technique
│
├── 🤖 Models
│   ├── movie_success_model.joblib    # Modèle entraîné (pipeline complet)
│   └── movie_success_meta.json       # Métadonnées (threshold, features, top categories)
│
├── 📓 Notebooks
│   └── tmdb_movie_success_classification_template_v7_threshold_optimization.ipynb
│       # Analyse complète et expérimentations
│
├── 🖥️ Application
│   ├── movie_success_app_scroll.py   # Interface graphique Tkinter
│   └── lion.png                      # Logo de l'application
│
├── 📋 Documentation
│   ├── README.md                     # Ce fichier
│   ├── requirements.txt              # Dépendances Python
│   └── LICENSE                       # Licence MIT
│
└── 📸 Screenshots
    └── app_screenshot.png            # Capture d'écran de l'interface
```

---

## 📊 Résultats

### Performance du modèle

| Métrique | Score |
|----------|-------|
| **F1-score (weighted)** | 0.73 |
| **Accuracy** | 73% |
| **ROC-AUC** | 0.79 |
| **Seuil optimal** | 0.50 |

### Matrice de confusion

```
                Prédit: FAILURE    Prédit: SUCCESS
Réel: FAILURE        420                 100
Réel: SUCCESS        150                 290
```

### Features les plus importantes

D'après les coefficients de la régression logistique :

1. **Budget** (+++) : Impact très positif sur le succès
2. **Cast size** (++) : Plus de stars = plus de succès
3. **Crew size** (++) : Équipe importante = production majeure
4. **Genre = Action** (+) : Performant commercialement
5. **Language = English** (+) : Marché international

### Insights métier

- 📈 **Budget > 50M USD** : 78% de chances de succès
- 📉 **Budget < 5M USD** : 32% de chances de succès
- 🎭 **Genres les plus performants** : Action, Adventure, Animation
- 🌍 **Films en anglais** : +25% de probabilité de succès
- 👥 **Cast > 40 acteurs** : Indicateur fort de blockbuster

---

## 🛠️ Technologies utilisées

### Machine Learning & Data Science
- **scikit-learn** : Pipeline ML, régression logistique, preprocessing
- **pandas** : Manipulation et analyse des données
- **numpy** : Calculs numériques
- **joblib** : Sérialisation du modèle

### Interface graphique
- **tkinter** : Interface graphique native Python
- **ttk** : Widgets modernes et thèmes

### Visualisation (dans le notebook)
- **matplotlib** : Graphiques et visualisations
- **seaborn** : Visualisations statistiques

### Outils de développement
- **Jupyter Notebook** : Analyse exploratoire
- **Git** : Versioning du code

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **TMDB** pour la mise à disposition du dataset
- **Kaggle** pour l'hébergement des données
- La communauté **scikit-learn** pour l'excellente documentation
- Tous les contributeurs open-source qui rendent ce type de projet possible

---

## 📚 Références

- [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Avoiding Data Leakage in Machine Learning](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Threshold Optimization for Classification](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)

---

<div align="center">

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐**

Made with ❤️ and 🐍

</div>

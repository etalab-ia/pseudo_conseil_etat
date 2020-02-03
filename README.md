# Pseudonymisation au Conseil d'État

Package permettant l'exploitation et la transformation des fichiers DOC contenant les décisions de justice du Conseil d'État (CdE). Il sert à préparer les données pour la modélisation du langage (*language modelling*) et la reconnaissance d'entités nommées (*named entity recognition*). Il contient aussi le code pour entrainer un modèle basline Flair.

N.B. 1: Ce package n'est utilisable que pour les décisions de justice du Conseil d'État.

N.B. 2: All the tools can run parallelized

N.B. 3: Travail en cours.


## Utilisation

### **Preprocessing**
Le preprocessing consiste en plusieurs étapes:
1. Transformer les .DOC à .TXT:

```bash
python doc2txt.py <path du répertoire avec les fichiers .DOC>
```

2. Extraire les annotations depuis la base de données du CdE et générer les fichiers d'annotation .XML.

```bash
python table2xml.py <path de la table documents> <path du répertoire de sortie> 
```

3. Combiner les fichiers .TXT de l'etape 1 avec les .XML de l'étape 2 pour generer les fichiers NER standard CoNLL (un mot taggé par ligne)

```bash
python xml2conll.py <path du répertoire avec les fichiers .XML> 
```
4. Pour générer un dataset **stratifié** train, test, dev à partir des fichiers CoNLL de l'étape 3.

```bash
python split_dataset.py <path du répertoire avec les fichiers _CoNLL.txt> <path du répertoire de sortie>
```

### **Modélisation**

Le script de modélisation est dans le repertoire `/models/`.

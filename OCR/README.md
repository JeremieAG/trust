# Solution de l’équipe OCR
Résultats sans erreur avec `3` appels a l'API (incluant les tentatives infructueuses) et un temps de traitement de `62.06` secondes sur un MacBook Pro M1.
Les résultats sont générés dans le fichier `result.json`.
Commande pour lancer le script :
```bash
python solution.py DEMO_MinuteBook_FR.pdf -o result.json  
```
Il faut installer les dépendances listées dans le fichier `requirements.txt` avant d'exécuter le script.
(Python 3.8+ < 3.14 est requis)

## Paramètres
- `pdf_file`: Chemin vers le fichier PDF à traiter
- `-o, --output`: Chemin vers le fichier de sortie JSON (par défaut `results.json`)
- `--api-url`: URL de l'API (par défaut `https://ai-models.autocomply.ca`)
- `--api-key`: Clé API pour l'authentification (par défaut `sk-ac-7f8e9d2c4b1a6e5f3d8c7b9a2e4f6d1c`)
- `--dpi`: Résolution en DPI pour le rendu des pages PDF (par défaut `150`)
- `--model`: Modèle LLM à utiliser (par défaut `gemini-2.5-flash`)
- `--batch-size`: Nombre de pages à envoyer par requête à l'API (par défaut `100`)
- `-h, --help`: Affiche l'aide pour les paramètres

## Approche du problème
Initalement, nous n'avions pas accès a la route `/ask` de l’API, nous étions donc contraints d’utiliser la route `/process-pdf` pour demander au LLM de classifier les pages du PDF.
Nous voulions à tout prix minimiser le nombre d’appels à un API, puisque celui-ci peut engendrer des coûts rapidement élevés à l’utilisation.
Nous avons donc décidé d’utiliser `pytesseract` une librairie OCR open source fait par google pour extraire le texte des pages du PDF localement.
Ensuite, après une analyse rapide des pages, on en est venue à la conclusion que les détails clefs pour classifier les pages étaient situés dans les premières et dernières lignes de chaque page. Pour la selection des lignes, on applique un filtre pour ignorer les lignes vide, ou contenant que des caractères spéciaux (-, _, *, etc..), puisqu'il indique souvent juste une erreur de rendu OCR. Ensuite on regroupe les lignes ayant peut de mots (moins de 2 mots) en une seule ligne pour donner un peu plus de contexte au LLM sans trop augmenter la taille du prompt.

Initialement, avec la limitation de la route `/process-pdf`, nous avions mis en place une fonction pour transformer le texte des x premières pages en une image et de l’envoyer au LLM avec un prompt demandant de classifier les pages. Cette méthode, bien que fonctionnelle, était loin d’être optimale, car on convertissait des images en texte pour ensuite les retransformer en images.

Heureusement, nous avons pu accéder a la route `/ask` de l’API, ce qui nous a permis d’envoyer le texte extrait par `pytesseract` directement au LLM paquet de x pages avec une superposition entre les paquets pour rajouter un peu de contexte entre les paquets.

La construction de la requête a été relativement simple, puisque nous avions juste à donner les grandes lignes et les limites de la tache. On demandait donc au LLM de nous retourner un simili CSV avec le `pageNumber, categoryNumber, confidenceScore` pour chaque page. Ce qui nous permettait ensuite de facilement parser la réponse.

Une fois la réponse du LLM reçue, nous parisien le résultat pour le transformer en tuple `(page_idx, category_idx, confidence)` que nous passions ensuite dans une fenêtre défilant pour enlever des incohérences évidentes, si, par exemple on avait une petite section entourée d’une même catégorie, on la recalcifiait dans la catégorie environnante.

Par la suite on crée un JSON avec les catégories et quelques métadonnées qui vont nous permettre de trouver des incohérences évidentes dans les classifications.
```json
{
  "name": "Category Name",
  "startPage": 1,
  "endPage": 5,
  "numPages": 5,
  "averageConfidence": 92.5
}
```

Ensuite, on gère les discontinuités dans une boucle en redemandant au LLM de déclassifier les pages qui semblent mal classifiées en lui fournissant le contexte des catégories environnantes. On répète cette étape jusqu’à ce qu’il n’y ait plus de discontinuités détectées ou qu’on ait atteint un nombre maximum d’itérations.

La dernière étape de traitement permet de trouver de petites catégories isolées qui auraient pu être mal classifiées. Pour ce faire on utilise du fuzzy matching pour tenter de déclassifier ces pages en se basant sur les x premières lignes de chaque page.

Finalement, on génère le fichier `results.json` avec le format demandé.

## Améliorations possibles
- Utiliser une méthode plus robuste pour détecter les discontinuités et les petites catégories
- Améliorer la requête pour obtenir des résultats plus précis du LLM
- Optimiser la taille des paquets de pages envoyées au LLM pour minimiser le nombre d’appels API tout en maximisant la précision
- Crée plus de règles heuristiques pour détecter et corriger les erreurs de classification
- Utiliser plusieurs travailleurs pour la gestions des discontinuités
- Trouver un moyen d'éviter
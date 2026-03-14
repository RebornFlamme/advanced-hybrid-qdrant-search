# Recherches avancées Qdrant

## Contexte technique

**Qdrant** : les points représentent des paragraphes de documents.
- L'ID du point est le `paragraph_id` (ex: `12301` = document `123`, paragraphe `01`)
- Le payload contient :
  - `document_id` (int) — identifiant du document parent
  - `tags` (string) — tags sous la forme `"#TAG1, #TAG2"`

**Parquet** : fichier avec colonnes `document_id` (int) et `text` (string, texte complet du document).

**Retour attendu** : toujours une liste de `document_id`.

**App** : moteur de recherche dans une app Dash via un `dcc.Input`.

---

## Types de requêtes

### Requête simple
Toute saisie qui ne commence pas par `c:` est une recherche sémantique directe sur toute la collection.

```
Victoire au superbowl
```

### Requête complexe
Commence obligatoirement par `c:`. Composée d'un `pre:` optionnel et d'un `req:` obligatoire.

```
c: [PRE] REQ
```

---

## Syntaxe détaillée

### Prefetch — `pre:`
Optionnel. Récupère un sous-ensemble de vecteurs sur lequel `req:` opère ensuite.

**Deux types, toujours explicitement spécifiés :**

- `pre: sem: "..."` — prefetch sémantique, supporte `lim:` pour limiter le nombre de vecteurs récupérés
- `pre: keywords: (...)` — prefetch par mots-clés booléens, pas de `lim:` (la logique booléenne borne naturellement les résultats)

`tags:` peut être ajouté après `pre:` ou `req:` pour filtrer les résultats.

### Requête finale — `req:`
Obligatoire. Opère sur le prefetch s'il existe, sinon sur toute la collection.

**Deux types, toujours explicitement spécifiés :**

- `req: sem: "..."` — recherche sémantique. Supporte un exemple positif et un négatif : `"positif" NOT "négatif"`
- `req: keywords: (...)` — recherche par mots-clés booléens

`lim:` après `req:` limite le nombre de résultats finaux retournés.
`tags:` peut être ajouté après `req:` pour filtrer les résultats finaux.

### Filtre par tags — `tags:`
Applicable à `pre:` ou `req:` (ou les deux). Même syntaxe booléenne que `keywords:`.

```
tags: #TAG1
tags: (#TAG1 OR #TAG2) AND NOT #TAG3
```

### Opérateurs booléens (pour `keywords:` et `tags:`)
- `AND` — les deux conditions doivent être présentes
- `OR` — au moins une condition
- `NOT` — exclusion
- Parenthèses pour grouper : `("plage" OR "vacances") AND NOT "sport"`

---

## Grammaire résumée

```
simple_query  := <phrase libre>

complex_query := "c:" [pre_clause] req_clause

pre_clause    := "pre:" ( "sem:" quoted ["NOT" quoted] ["lim:" N]
                        | "keywords:" bool_expr )
                 ["tags:" bool_expr]

req_clause    := "req:" ( "sem:" quoted ["NOT" quoted]
                        | "keywords:" bool_expr )
                 ["lim:" N]
                 ["tags:" bool_expr]

bool_expr     := term (("AND" | "OR") term)*
term          := ["NOT"] (quoted | "(" bool_expr ")")
quoted        := '"' texte '"'
```

---

## Exemples

```
# Requête simple (sémantique directe)
Victoire au superbowl

# Prefetch keywords + req sémantique avec limite finale
c: pre: keywords: ("plage" OR "vacances") AND "sport" req: sem: "Crème de bronzage" lim: 50

# Prefetch keywords + req sémantique avec vecteur négatif
c: pre: keywords: ("plage" OR "vacances") AND NOT "sport" req: sem: "Crème de bronzage" NOT "Après soleil" lim: 50

# Prefetch sémantique limité à 50 vecteurs + req sémantique
c: pre: sem: "Il fait beau" lim: 50 req: sem: "Il fait chaud"

# Prefetch sémantique + req keywords
c: pre: sem: "Voyage en mer" lim: 100 req: keywords: "bateau" AND "tempête" lim: 20

# Filtre par tags sur req
c: pre: sem: "Nutrition sportive" lim: 50 req: sem: "Protéines" tags: #SPORT AND NOT #VEGAN lim: 30

# Filtre par tags sur pre et req
c: pre: keywords: "musculation" tags: #SPORT req: sem: "Récupération" tags: #NUTRITION lim: 25
```


## Instructions
Whenever corrected, after making a mistake or misinterpreting, add a section in here. (`CLAUDE.MD`) to instruct future sessions, avoiding the mistake again. 
ALWAYS use subagents where possible, parralell work is better.

### Tools

ALWAYS use `uv` in python, `uv add` for installs, NEVER `pip install` directly. `uv run example.py` for running, NEVER `python example.py` or `python3 example.py` directly.
Avoid editing `pyproject.toml` directly, where possible use `uv add`, `uv remove` etc.

Use `ruff` for formatting python files, run via `uv run ruff`. Run `uv ruff check` on any new files before running them or including them. Fix any warning errors before procedding. 

## Guidelines
* Use google docstring format
* Follow SOLID design principles where possible
* we do not care about test coverage too much, functionality is most important.
* if writing unit tests, always use pytest, run via uv run pytest
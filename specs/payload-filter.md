# Spécification — Filtrage par payload Qdrant

## Contexte

Actuellement, le filtrage au niveau sémantique (`pre:` et `req:`) est limité au champ `tags` via la clause `tags:` (match full-text sur un string).

Cette feature introduit un mécanisme générique de filtrage par valeur exacte sur **n'importe quel champ du payload Qdrant**, en inclusion ou exclusion.

---

## Objectif

Permettre à l'utilisateur de restreindre ou exclure des résultats selon les valeurs d'un champ payload arbitraire, directement dans la query string.

**Exemples de cas d'usage :**
- Ne chercher que parmi les documents d'un certain auteur : `filter: author IN ["Dupont", "Martin"]`
- Exclure certaines catégories : `filter: category NOT IN ["fiction", "poésie"]`
- Cibler une année précise : `filter: year IN [2023]`
- Exclure un document spécifique : `filter: document_id NOT IN [42]`

---

## Syntaxe

```
filter: <field> IN [<value>, ...]
filter: <field> NOT IN [<value>, ...]
```

### Règles

- `<field>` : nom exact du champ dans le payload Qdrant (sensible à la casse)
- `IN` / `NOT IN` : mots-clés case-insensitive
- `[<value>, ...]` : liste de valeurs entre crochets, séparées par des virgules
- Les valeurs peuvent être :
  - Des **strings** entre guillemets doubles : `"Dupont"`
  - Des **entiers** sans guillemets : `42`, `2023`
- Plusieurs clauses `filter:` peuvent être chaînées (toutes appliquées en AND)

---

## Intégration dans la query

La clause `filter:` est optionnelle et s'applique globalement à `pre:` ou `req:` (ou au niveau de la query entière pour les queries simples).

### Placement

```
simple_query  := <phrase> [filter: ...]

complex_query := "c:" [pre_clause] req_clause

pre_clause    := "pre:" (...) ["lim:" N] ["tags:" ...] ["filter:" ...]
req_clause    := "req:" (...) ["lim:" N] ["tags:" ...] ["filter:" ...]
```

---

## Exemples de requêtes

### Query simple avec filtre d'inclusion

```
Victoire au superbowl filter: category IN ["sport", "football américain"]
```

→ Recherche sémantique sur toute la collection, restreinte aux documents dont `category` vaut `"sport"` ou `"football américain"`.

---

### Query simple avec filtre d'exclusion

```
Recettes de cuisine filter: document_id NOT IN [12, 45, 78]
```

→ Recherche sémantique en excluant les documents 12, 45 et 78.

---

### Prefetch sémantique avec filtre sur req

```
c: pre: sem: "plage" lim: 50 req: sem: "bronzage" filter: author IN ["Dupont"] lim: 20
```

→ Prefetch sémantique sur "plage", puis re-rank sur "bronzage" en ne gardant que les documents de l'auteur "Dupont".

---

### Filtre sur pre et req différents

```
c: pre: sem: "nutrition" lim: 100 filter: year IN [2022, 2023] req: sem: "protéines" filter: category NOT IN ["publicité"] lim: 30
```

→ Le prefetch est restreint aux années 2022-2023 ; le req exclut la catégorie "publicité".

---

### Plusieurs filtres chaînés (AND implicite)

```
c: req: sem: "intelligence artificielle" filter: category IN ["science"] filter: year NOT IN [2020] lim: 25
```

→ Les deux `filter:` sont combinés en AND : `category = "science"` ET `year ≠ 2020`.

---

### Filtre avec entiers

```
c: req: keywords: "alcène" filter: document_id IN [14, 23, 98]
```

→ Recherche par mot-clé, restreinte à trois documents précis.

---

## Comportement Qdrant sous-jacent

| Clause | Traduit en |
|---|---|
| `filter: field IN ["a", "b"]` | `FieldCondition(key="field", match=MatchAny(any=["a", "b"]))` dans `must` |
| `filter: field NOT IN ["a", "b"]` | `FieldCondition(key="field", match=MatchAny(any=["a", "b"]))` dans `must_not` |
| `filter: field IN [1, 2]` | `FieldCondition(key="field", match=MatchAny(any=[1, 2]))` dans `must` |
| Plusieurs `filter:` | Chaque condition ajoutée dans `must` (AND) |

Les filtres `filter:` sont **combinés avec** les filtres `tags:` existants via AND.

---

## Ce que cette feature ne couvre pas

- Comparaisons de plage (`>`, `<`, `between`) — hors scope (feature séparée)
- Matching full-text sur un champ — déjà couvert par `tags:` et `keywords:`
- Logique OR entre plusieurs `filter:` différents — les clauses multiples sont toujours en AND

---

## Impact sur le parser

Nouveaux tokens à ajouter :
- `FILTER_COLON` : `filter:`
- `IN` : mot-clé case-insensitive
- `NOT_IN` : séquence `NOT IN` case-insensitive
- `LBRACKET` / `RBRACKET` : `[` / `]`
- `INTEGER` : `[0-9]+` (distinct de `QUOTED` pour les strings)

Nouvelle règle de grammaire :

```
filter_clause := "filter:" WORD ("IN" | "NOT" "IN") "[" value_list "]"
value_list    := value ("," value)*
value         := quoted | integer
```

## Impact sur l'executor

Ajouter une méthode `_filter_clauses_to_filter(clauses) -> Filter | None` qui construit un `Filter` Qdrant à partir de la liste des `FilterClause` parsées, puis le merge avec les filtres `tags:` existants via `_merge_filters`.

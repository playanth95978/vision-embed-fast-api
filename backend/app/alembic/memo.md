# 🧠 FastAPI + SQLModel + Alembic + pgvector — Mémo

## 📦 Stack

* FastAPI
* SQLModel / SQLAlchemy
* Alembic (migrations)
* PostgreSQL + pgvector

---

# 🚀 Workflow global

## 1. Modifier un modèle

Exemple :

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column

embedding: list[float] | None = Field(
    default=None,
    sa_column=Column(Vector(512), nullable=True)
)
```

---

## 2. Générer une migration

⚠️ La base doit être à jour !

```bash
alembic upgrade head
```

Puis :

```bash
alembic revision --autogenerate -m "add embedding"
```

---

## 3. ⚠️ Corriger la migration (OBLIGATOIRE avec pgvector)

Alembic ne gère pas :

* extension pgvector
* index vectoriel (ivfflat)

👉 Modifier le fichier généré :

```python
def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "imageupload",
        sa.Column("embedding", sa.Vector(512), nullable=True)
    )

    op.execute("""
        CREATE INDEX ix_image_embedding
        ON imageupload
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
```

---

## 4. Appliquer la migration

```bash
alembic upgrade head
```

---

# ❗ Problèmes fréquents

## 🔴 "Target database is not up to date"

```bash
alembic upgrade head
```

OU si DB déjà sync :

```bash
alembic stamp head
```

---

## 🔴 Erreur pgvector

```text
ValueError: expected ndim to be 1
```

👉 Mauvais import :

❌

```python
from pgvector.vector import Vector
```

✅

```python
from pgvector.sqlalchemy import Vector
```

---

## 🔴 Alembic freeze / bloque

Causes possibles :

* import lourd dans models (torch, clip…)
* import circulaire
* autogenerate + pgvector

👉 Solution :

```bash
alembic revision -m "manual"
```

Puis écrire SQL à la main

---

# ⚡ Bonnes pratiques

## ✔ Toujours utiliser Alembic

❌ `create_all()` en prod

---

## ✔ Index obligatoire pour embeddings

```sql
CREATE INDEX ix_image_embedding
ON imageupload
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

---

## ✔ Analyser la table après index

```sql
ANALYZE imageupload;
```

---

## ✔ Dimension embedding

| Modèle | Dimension |
| ------ | --------- |
| CLIP   | 512       |
| OpenAI | 1536      |
| BGE    | 768       |

---

## ✔ created_at propre

```python
created_at: datetime = Field(
    default_factory=get_datetime_utc,
    sa_column=Column(DateTime(timezone=True), nullable=False),
)
```

---

# 🧠 Notes importantes

* Alembic = source de vérité DB
* SQLModel = description du schéma
* pgvector nécessite SQL manuel
* Ne jamais faire confiance à autogenerate pour les index avancés

---

# 🏁 TL;DR

```bash
# 1. Sync DB
alembic upgrade head

# 2. Générer migration
alembic revision --autogenerate -m "..."

# 3. Corriger à la main (pgvector)

# 4. Appliquer
alembic upgrade head
```

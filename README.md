# vision-embed-fast-api

API **FastAPI** de recherche multimodale : indexation d'images par embeddings **CLIP** + pipeline **RAG hybride** (pgvector + BM25 ParadeDB, fusion RRF, reranking cross-encoder, génération Ollama/Mistral), avec un frontend **React 19**.

---

## 1. Architecture

```
                        ┌─────────────────────────────┐
                        │  Frontend React 19 + Vite   │
                        │  TanStack Router/Query      │
                        │  Tailwind 4 + shadcn/ui     │
                        │  client TS auto-généré      │
                        └──────────────┬──────────────┘
                                       │ HTTP  /api/v1  (JWT)
                        ┌──────────────▼──────────────┐
                        │      FastAPI (Granian)      │
                        │  SQLModel · Pydantic v2     │
                        ├─────────────────────────────┤
                        │ /login /users /items /utils │
                        │ /images   → CLIP            │
                        │ /rag      → ingestion       │
                        │ /chat     → SSE streaming   │
                        └───┬──────────────────┬──────┘
                            │                  │
             ┌──────────────▼───────┐   ┌──────▼────────────────────────┐
             │  Service image       │   │  Service RAG                  │
             │  CLIP ViT-B/32       │   │  ingestion → retrieval →      │
             │  (transformers/torch)│   │  RRF → rerank → LLM           │
             │  vecteurs 512d       │   │                               │
             └──────────────┬───────┘   └──┬─────────┬──────────┬───────┘
                            │              │         │          │
                            │        embeddings   reranker     LLM
                            │              │         │          │
                            │        ┌─────▼───┐ ┌───▼──────┐ ┌─▼──────────┐
                            │        │ Ollama  │ │ ONNX     │ │ Ollama     │
                            │        │ nomic / │ │ bge-rerank│ │ mistral    │
                            │        │ mxbai   │ │ base INT8│ │ ou Mistral │
                            │        └─────────┘ └──────────┘ │ cloud API  │
                            │                                 └────────────┘
             ┌──────────────▼─────────────────────────────────┐
             │       PostgreSQL — image paradedb/paradedb     │
             │  pgvector (cosine) + BM25 (pg_search ParadeDB) │
             │  imageupload(embedding 512)                    │
             │  vector_docs / vector_jira / vector_pdf   768  │
             │  vector_code                             1024  │
             │  migrations Alembic                            │
             └────────────────────────────────────────────────┘
```

### Pipeline RAG (`/api/v1/rag`, `/api/v1/chat`)

```
question ──► embedding Ollama (par source)
         ├─► recherche vectorielle pgvector (cosine, seuil 0.3, 30/source)
         └─► recherche lexicale BM25 ParadeDB (30/source)
                    │
                    ▼
   fusion RRF pondérée (k=60 ; GITHUB 1.2 · CONFLUENCE 1.0 · PDF 0.9 · JIRA 0.8)
                    │  top 20
                    ▼
   reranking cross-encoder bge-reranker-base INT8 (ONNX, 128 tokens) ──► top 5 (fallback 3)
                    │
                    ▼
   prompt + contexte ──► LLM (Ollama `mistral:latest` ou Mistral cloud)
                    │
                    ▼
   réponse en streaming SSE + sources citées
```

### Pipeline images (`/api/v1/images`)

```
upload image  ──► CLIP ViT-B/32 (image encoder) ──► vecteur 512d normalisé L2 ──► imageupload.embedding
requête texte ──► CLIP (text encoder)           ──► vecteur 512d ──► ORDER BY embedding <=> query
```

---

## 2. Stack technique & versions

### Backend (`backend/pyproject.toml`)

| Domaine | Techno | Version |
|---|---|---|
| Langage | Python | `>=3.10,<4.0` (image Docker : 3.10) |
| Gestion deps | uv (`uv.lock`) | 0.11+ |
| API | FastAPI `[standard]` | `>=0.114.2,<1.0` |
| Serveur ASGI | Granian | `>=2.7.3` |
| ORM / modèles | SQLModel · Pydantic | `>=0.0.21` · `>2.0` |
| Driver DB | psycopg (binary) | `>=3.1.13,<4` |
| Migrations | Alembic | `>=1.12.1,<2` |
| Vecteurs | pgvector | `>=0.4.2` |
| Vision | transformers · torch · pillow | `>=5.5` · `>=2.11` · `>=12.2` |
| Embeddings | ollama | `>=0.6.0` |
| LLM cloud | mistralai | `>=1.9.11` |
| Reranker | onnxruntime · tokenizers | `>=1.18` · `>=0.20` |
| Chunking / PDF | langchain-text-splitters · pypdf | `>=0.3.8` · `>=6.1.1` |
| Auth | pyjwt · pwdlib[argon2,bcrypt] | `>=2.8` · `>=0.3` |
| Observabilité | sentry-sdk[fastapi] | `>=2.0,<3` |
| Qualité | ruff · mypy (strict) · ty · pytest · coverage · prek | cf. `[dependency-groups]` |

### Modèles ML

| Usage | Modèle | Dimension |
|---|---|---|
| Image + texte→image | `openai/clip-vit-base-patch32` | 512 |
| Embeddings texte (Confluence, Jira, PDF) | `nomic-embed-text` (Ollama) | 768 |
| Embeddings code (GitHub) | `mxbai-embed-large` (Ollama) | 1024 |
| Reranking | `bge-reranker-base` INT8 ONNX (278 M) | — |
| Génération | `mistral:latest` (Ollama) ou `mistral-medium-latest` (cloud) | — |

### Frontend (`frontend/package.json`)

| Domaine | Techno | Version |
|---|---|---|
| UI | React · React DOM | `^19.1` · `^19.2` |
| Build | Vite · plugin-react-swc · TypeScript | `^7.3` · `^4.2` · `^5.9` |
| Routing / data | TanStack Router · Query · Table | `^1.163` · `^5.90` · `^8.21` |
| Style | Tailwind CSS 4 · shadcn/ui (Radix) · lucide-react | `^4.2` |
| Formulaires | react-hook-form · zod · @hookform/resolvers | `^7.68` · `^4.3` · `^5.2` |
| HTTP | axios + client généré via `@hey-api/openapi-ts` | `1.13.5` · `0.73.0` |
| Lint | Biome | `^2.3` |
| E2E | Playwright | `1.58.2` |
| Runtime dev | Node 22 / Bun 1.3 (`bun.lock`) | — |

### Infrastructure

| Composant | Image / outil |
|---|---|
| Base de données | `paradedb/paradedb:latest` (Postgres + pgvector + BM25), exposée sur `127.0.0.1:5433` |
| Admin DB | `adminer` |
| Reverse proxy | Traefik (`compose.traefik.yml`) |
| Mails de dev | Mailcatcher |
| Orchestration | Docker Compose (`compose.yml`, `compose.override.yml`) |
| CI/CD | GitHub Actions (`.github/`) |

---

## 3. Démarrage

### Prérequis

- Docker + Docker Compose
- [uv](https://docs.astral.sh/uv/) (backend), Node 22+ ou Bun (frontend)
- [Ollama](https://ollama.com) si vous utilisez le RAG en local
- Un fichier `.env` à la racine (voir §4)

### 3.1 Tout en Docker (le plus simple)

```bash
docker compose watch
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000 |
| Swagger / OpenAPI | http://localhost:8000/docs |
| Adminer | http://localhost:8080 |
| Mailcatcher | http://localhost:1080 |
| Traefik UI | http://localhost:8090 |

Logs : `docker compose logs backend` · Arrêt : `docker compose down`

### 3.2 Développement local (backend hors Docker)

Démarrer uniquement la base :

```bash
docker compose up -d db
```

Installer les dépendances Python :

```bash
cd backend && uv sync
```

Appliquer les migrations :

```bash
cd backend && uv run alembic upgrade head
```

Lancer l'API en mode dev (reload) :

```bash
cd backend && PYTHONUTF8=1 uv run fastapi dev app/main.py
```

> `PYTHONUTF8=1` est indispensable sur Windows : sans lui, le banner de démarrage de FastAPI
> (emoji 🚀) fait échouer le processus avec un `UnicodeEncodeError` en console cp1252.

Pré-télécharger le modèle CLIP en local (optionnel, évite le download au premier appel) :

```bash
cd backend && uv run python download.py
```

> En local hors Docker, mettre `POSTGRES_SERVER=localhost` et `POSTGRES_PORT=5433` dans `.env`.

### 3.3 Frontend

```bash
cd frontend && npm install && npm run dev
```

Régénérer le client TypeScript depuis l'OpenAPI (backend démarré) :

```bash
cd frontend && npm run generate-client
```

### 3.4 Modèle de reranking

Le cross-encoder (~280 Mo) n'est pas versionné. Il est téléchargé automatiquement au premier
reranking, mais mieux vaut le pré-installer :

```bash
./scripts/download-reranker.sh
```

### 3.5 Ollama (RAG local)

```bash
ollama pull nomic-embed-text && ollama pull mxbai-embed-large && ollama pull mistral
```

Pour utiliser Mistral cloud à la place : `RAG_LLM_PROVIDER=mistral` et `MISTRAL_API_KEY=...` dans `.env`.

---

## 4. Configuration (`.env`)

| Variable | Rôle | Défaut |
|---|---|---|
| `ENVIRONMENT` | `local` / `staging` / `production` | `local` |
| `SECRET_KEY` | Signature JWT | généré |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Durée du token | 8 jours |
| `FIRST_SUPERUSER` / `FIRST_SUPERUSER_PASSWORD` | Compte admin créé au prestart | — |
| `POSTGRES_SERVER` / `_PORT` / `_DB` / `_USER` / `_PASSWORD` | Connexion DB | `5432` (`5433` côté hôte) |
| `SQLALCHEMY_ECHO` | Écho SQL (ralentit fortement le démarrage) | `false` |
| `OLLAMA_BASE_URL` | Serveur Ollama | `http://localhost:11434` |
| `RAG_LLM_PROVIDER` | `ollama` ou `mistral` | `ollama` |
| `RAG_LLM_MODEL` | Modèle Ollama de génération | `mistral:latest` |
| `MISTRAL_API_KEY` / `MISTRAL_MODEL` | Mistral cloud | — / `mistral-medium-latest` |
| `RERANKER_ONNX_PATH` / `RERANKER_TOKENIZER_PATH` | Modèle et tokenizer de reranking (solidaires) | `models/rerankers/bge-base/…` |
| `RERANKER_MAX_LENGTH` | Troncature en tokens | `128` |
| `RERANKER_INTRA_OP_THREADS` | Threads ONNX par inférence (0 = tous) | `0` |
| `BACKEND_CORS_ORIGINS` | Origines autorisées | — |
| `SENTRY_DSN` | Monitoring | — |

> Ne committez jamais de secrets : `.env` en local uniquement, variables d'environnement en déploiement.

---

## 5. API (préfixe `/api/v1`)

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/login/access-token` | Authentification JWT |
| `*` | `/users`, `/items` | CRUD standard |
| `GET` | `/images/all` | Liste paginée des images |
| `POST` | `/images/` | Upload + embedding CLIP |
| `GET` | `/images/search` | Recherche sémantique texte→image |
| `POST` | `/rag/ingest/text` | Ingestion d'un texte (chunks 1000 / overlap 200, upsert par `docId`) |
| `POST` | `/rag/ingest/file` | Ingestion d'un fichier (PDF via pypdf) |
| `GET` | `/rag/merge` | Recherche hybride + RRF + reranking |
| `POST` | `/chat/stream` | Réponse LLM en streaming SSE avec sources |
| `GET` | `/chat/translate` | Traduction de la requête utilisateur |

Documentation interactive : http://localhost:8000/docs

---

## 6. Bancs de mesure

```bash
cd backend && uv run python scripts/bench_reranker.py
```

Précision (P@1, exact@5, nDCG@10, MRR sur corpus étiqueté) et débit CPU du reranker, même
protocole que les bancs Java de job-search-ai. Mesuré sur 16 cœurs, lots de 20/30/60 documents :

| modèle | ms/doc | docs/s | P@1 | nDCG@10 | MRR |
|---|---|---|---|---|---|
| bge-reranker-base INT8 ONNX @128 (production) | 46-57 | 17-22 | 1.00 | 0,945-0,989 | 1.00 |
| bge-reranker-v2-m3 torch @256 (ancien) | 128-159 | 6-8 | 1.00 | 0,948-0,983 | 1.00 |

Test RAG de bout en bout contre une API démarrée (ingestion + questions SSE + vérification
des sources) :

```bash
cd backend && uv run python scripts/smoke_rag.py
```

## 7. Tests & qualité

```bash
cd backend && uv run bash scripts/test.sh
```

```bash
cd frontend && npm run lint && npx playwright test
```

Hooks pre-commit :

```bash
uv run prek install -f
```

---

## 8. Déploiement

Voir [deployment.md](deployment.md) (Docker Compose + Traefik, HTTPS automatique) et [development.md](development.md) pour le détail du workflow local. CI/CD via GitHub Actions.

---

## Licence & contribution

[LICENSE](LICENSE) · [SECURITY.md](SECURITY.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

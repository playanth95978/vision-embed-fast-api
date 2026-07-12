"""Moitié "retrieval" du pipeline : recherche hybride pgvector + BM25 ParadeDB, fusion RRF
pondérée, extraction des sources et formatage du contexte.

Portage de ``HybridSimilaritySearchService``, ``ReciprocalRankFusionService`` et des méthodes
de retrieval de ``ChatService`` (Java). Utilise la Session SQLModel synchrone du projet.
"""

import json
from collections import OrderedDict
from urllib.parse import quote

from sqlalchemy import text
from sqlmodel import Session

from app.models import SourceInfo
from app.service.rag import embeddings
from app.service.rag.schemas import Doc
from app.service.rag.sources import (
    ALLOWED_TABLES,
    PER_SOURCE_LIMIT,
    RRF_K,
    SIMILARITY_THRESHOLD,
    SOURCES,
    WEIGHT_BY_TYPE,
    RagSource,
)


def _to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in embedding) + "]"


def _as_metadata(raw: object) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)  # type: ignore[arg-type]


class Retriever:
    """Recherche hybride multi-sources sur les tables pgvector du projet."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- Recherches unitaires -------------------------------------------------

    def search_vector(self, src: RagSource, query: str, limit: int) -> list[Doc]:
        """Similarité cosine pgvector, seuil 0.3 (SearchRequest.similarityThreshold).

        ``score = 1 - distance_cosine`` ; on filtre ``distance < 1 - seuil`` comme Spring AI.
        """
        if src.table not in ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {src.table}")
        embedding = _to_vector_literal(embeddings.embed(query, src.embedding_model))
        max_distance = 1.0 - SIMILARITY_THRESHOLD
        sql = text(
            f"SELECT id::text AS id, content, metadata, "  # noqa: S608 (table validée)
            f"(embedding <=> CAST(:vec AS vector)) AS distance "
            f"FROM {src.table} "
            f"WHERE (embedding <=> CAST(:vec AS vector)) < :maxd "
            f"ORDER BY distance LIMIT :lim"
        )
        rows = self._session.execute(
            sql, {"vec": embedding, "maxd": max_distance, "lim": limit}
        ).mappings()
        docs: list[Doc] = []
        for r in rows:
            docs.append(
                Doc(
                    id=str(r["id"]),
                    text=r["content"] or "",
                    metadata=_as_metadata(r["metadata"]),
                    score=1.0 - float(r["distance"]),
                )
            )
        return docs

    def search_bm25(self, src: RagSource, query: str, limit: int) -> list[Doc]:
        """BM25 full-text ParadeDB (opérateur ``|||``). Dégradation propre si l'index manque."""
        if src.table not in ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {src.table}")
        sql = text(
            f"SELECT id::text AS id, content, metadata, paradedb.score(id) AS score "  # noqa: S608
            f"FROM {src.table} WHERE content ||| :q ORDER BY score DESC LIMIT :lim"
        )
        try:
            rows = self._session.execute(sql, {"q": query, "lim": limit}).mappings().all()
        except Exception:
            # Pas d'index bm25 sur cette table -> vectoriel seul pour la source (comme le Java).
            self._session.rollback()
            return []
        docs: list[Doc] = []
        for r in rows:
            metadata = _as_metadata(r["metadata"])
            metadata.setdefault("score", float(r["score"]))
            metadata.setdefault("source", src.data_type)
            docs.append(
                Doc(id=str(r["id"]), text=r["content"] or "", metadata=metadata, score=float(r["score"]))
            )
        return docs

    # -- Recherche hybride multi-sources -------------------------------------

    def hybrid_search(self, query: str) -> list[list[Doc]]:
        """Pour chaque source : listes BRUTES vector + BM25 (metadata type/rank posées)."""
        rankings: list[list[Doc]] = []
        for src in SOURCES:
            vec_docs = self.search_vector(src, query, PER_SOURCE_LIMIT)
            bm25_docs = self.search_bm25(src, query, PER_SOURCE_LIMIT)
            for i, d in enumerate(vec_docs):
                d.metadata["type"] = src.data_type
                d.metadata["rank"] = i + 1
            for i, d in enumerate(bm25_docs):
                d.metadata["type"] = src.data_type
                d.metadata["rank_bm25"] = i + 1
            if vec_docs:
                rankings.append(vec_docs)
            if bm25_docs:
                rankings.append(bm25_docs)
        return rankings

    def fetch_documents(self, src: RagSource, query: str, limit: int) -> list[Doc]:
        """vector + BM25 fusionnés par RRF, pour une seule source (fallback)."""
        vec_docs = self.search_vector(src, query, limit)
        bm25_docs = self.search_bm25(src, query, limit)
        return reciprocal_rank_fusion([vec_docs, bm25_docs], limit)


def reciprocal_rank_fusion(rankings: list[list[Doc]], top_k: int) -> list[Doc]:
    """RRF pondéré par le poids de la source : ``score = poids * 1/(k + rang + 1)``.

    Portage exact de ReciprocalRankFusionService.rrf (k=60, poids par défaut 1.0).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Doc] = {}
    for ranking in rankings:
        for i, doc in enumerate(ranking):
            doc_map.setdefault(doc.id, doc)
            source_type = doc.metadata.get("type")
            weight = WEIGHT_BY_TYPE.get(source_type, 1.0) if source_type else 1.0
            scores[doc.id] = scores.get(doc.id, 0.0) + weight * (1.0 / (RRF_K + i + 1))
    ranked_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[i] for i in ranked_ids[:top_k]]


# -- Formatage contexte + sources (miroir de ChatService) --------------------


def format_document_with_metadata(doc: Doc) -> str:
    """Enrichit les documents Jira (autorité maintainer/user, statut...) comme le Java."""
    meta = doc.metadata
    if str(meta.get("source", "unknown")).lower() != "jira":
        return doc.text

    parts: list[str] = []
    is_comment = str(meta.get("isComment")) == "true"
    author_type = str(meta.get("authorType", "unknown"))
    if is_comment:
        parts.append(
            "### MAINTAINER COMMENT (HIGH AUTHORITY)\n"
            if author_type == "maintainer"
            else "### USER COMMENT (LOW AUTHORITY)\n"
        )
    else:
        parts.append("### OFFICIAL JIRA ISSUE DESCRIPTION\n")
    for key, label in (("issueKey", "Issue"), ("title", "Title"), ("status", "Status"), ("priority", "Priority")):
        if key in meta:
            parts.append(f"{label}: {meta[key]}\n")
    parts.append("\nCONTENT:\n")
    parts.append(doc.text)
    return "".join(parts)


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def _resolve_score(doc: Doc) -> float:
    meta = doc.metadata
    if isinstance(meta.get("rerankScore"), int | float):
        return float(meta["rerankScore"])
    if doc.score is not None:
        return float(doc.score)
    if isinstance(meta.get("score"), int | float):
        return float(meta["score"])
    return 0.0


def _build_source(title: str, source: str, url: str, score: float, meta: dict, file_id: str | None) -> SourceInfo:
    if str(meta.get("sourceKind")) != "vba_macro" or file_id is None:
        return SourceInfo(title=title, source=source, url=url, score=score)
    module = str(meta.get("module", ""))
    return SourceInfo(
        title=title,
        source=source,
        url=url,
        score=score,
        kind="code",
        module=module,
        startLine=_to_int(meta.get("startLine")),
        endLine=_to_int(meta.get("endLine")),
        viewUrl=f"/api/files/{file_id}/macros?module={quote(module)}",
    )


def _source_dedup_key(info: SourceInfo, url: str) -> str:
    base = url if url else info.source
    if info.kind == "code" and info.module:
        return f"{base}#module={info.module}"
    return base


def extract_sources(docs: list[Doc]) -> list[SourceInfo]:
    """Déduplication par document (LinkedHashMap Java -> OrderedDict), meilleur score conservé."""
    if not docs:
        return []
    deduped: "OrderedDict[str, SourceInfo]" = OrderedDict()
    for doc in docs:
        meta = doc.metadata
        title = str(meta.get("title") or meta.get("file_name") or "Document")
        source = str(meta.get("source") or meta.get("type") or "unknown")
        url = str(meta.get("url") or meta.get("link") or "")
        file_id = str(meta["fileId"]) if meta.get("fileId") is not None else None
        if not url and file_id:
            url = f"/api/files/{file_id}/download"
        info = _build_source(title, source, url, _resolve_score(doc), meta, file_id)
        deduped.setdefault(_source_dedup_key(info, url), info)
    return list(deduped.values())

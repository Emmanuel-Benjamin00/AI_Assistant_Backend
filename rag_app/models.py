from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVector, SearchVectorField
from django.db import models
from pgvector.django import HnswIndex, VectorField

# Text-search configuration for keyword search: stems words ("refunds" -> "refund")
# and drops English stop words.
SEARCH_CONFIG = "english"


class Document(models.Model):
    class Source(models.TextChoices):
        TEXT = "text", "Pasted text"
        PDF = "pdf", "PDF"
        DOCX = "docx", "Word document"
        PLAIN = "plain", "Text/Markdown file"

    title = models.CharField(max_length=255)
    source = models.CharField(max_length=10, choices=Source.choices, default=Source.TEXT)
    filename = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"{self.id}: {self.title}"


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    text = models.TextField()
    embedding = VectorField(dimensions=1536, null=True, blank=True)
    # Postgres computes this from `text` on every insert/update, so it can never go stale.
    search_vector = models.GeneratedField(
        expression=SearchVector("text", config=SEARCH_CONFIG),
        output_field=SearchVectorField(),
        db_persist=True,
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            # Approximate nearest-neighbour index so vector search stays fast as chunks grow.
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
            # Inverted index for full-text (keyword) search.
            GinIndex(name="chunk_search_vector_gin", fields=["search_vector"]),
        ]

    def __str__(self) -> str:
        return f"{self.id} (doc={self.document_id})"

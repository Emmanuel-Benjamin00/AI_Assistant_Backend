import httpx
from django.db import transaction
from pgvector.django import CosineDistance
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .chunking import chunk_text_paragraphs
from .models import Chunk, Document
from .serializers import AskSerializer, DocumentIngestSerializer
from .services import llm


class DocumentIngestView(APIView):
    def post(self, request):
        serializer = DocumentIngestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        title = serializer.validated_data["title"]
        text = serializer.validated_data["text"]

        chunk_texts = chunk_text_paragraphs(text)
        try:
            with transaction.atomic():
                doc = Document.objects.create(title=title)
                if not chunk_texts:
                    return Response(
                        {"document_id": doc.id, "chunks_created": 0},
                        status=status.HTTP_201_CREATED,
                    )
                vectors = llm.embed_texts(chunk_texts)
                if len(vectors) != len(chunk_texts):
                    raise ValueError("embedding count mismatch")
                Chunk.objects.bulk_create(
                    [
                        Chunk(document=doc, text=t, embedding=v)
                        for t, v in zip(chunk_texts, vectors, strict=True)
                    ]
                )
        except RuntimeError as e:
            return Response({"detail": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except httpx.HTTPStatusError as e:
            return Response(
                {"detail": "OpenAI request failed", "status_code": e.response.status_code},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        except ValueError as e:
            return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            {"document_id": doc.id, "chunks_created": len(chunk_texts)},
            status=status.HTTP_201_CREATED,
        )


class AskView(APIView):
    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        question = serializer.validated_data["question"]
        top_k = serializer.validated_data["top_k"]

        try:
            q_vec = llm.embed_texts([question])[0]
        except RuntimeError as e:
            return Response({"detail": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except httpx.HTTPStatusError as e:
            return Response(
                {"detail": "OpenAI request failed", "status_code": e.response.status_code},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        retrieved = list(
            Chunk.objects.filter(embedding__isnull=False)
            .order_by(CosineDistance("embedding", q_vec))[:top_k]
        )
        if not retrieved:
            return Response(
                {
                    "answer": "I don’t have any indexed content to search yet. Ingest documents first.",
                    "sources": [],
                },
                status=status.HTTP_200_OK,
            )

        context_blocks = []
        for i, ch in enumerate(retrieved, start=1):
            context_blocks.append(f"[{i}] (chunk_id={ch.id})\n{ch.text}")
        context = "\n\n".join(context_blocks)

        system = (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "If the answer is not in the context, say you don’t know. "
            "When possible, cite chunk numbers like [1], [2]."
        )
        user_msg = f"Context:\n{context}\n\nQuestion:\n{question}"

        try:
            answer = llm.chat(system, user_msg)
        except RuntimeError as e:
            return Response({"detail": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except httpx.HTTPStatusError as e:
            return Response(
                {"detail": "OpenAI request failed", "status_code": e.response.status_code},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        sources = [{"chunk_id": ch.id, "text": ch.text} for ch in retrieved]
        return Response({"answer": answer, "sources": sources}, status=status.HTTP_200_OK)

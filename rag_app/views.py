from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .chunking import chunk_text_paragraphs
from .models import Chunk, Document
from .serializers import DocumentIngestSerializer


class DocumentIngestView(APIView):
    def post(self, request):
        serializer = DocumentIngestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        title = serializer.validated_data["title"]
        text = serializer.validated_data["text"]

        doc = Document.objects.create(title=title)
        chunks = chunk_text_paragraphs(text)

        Chunk.objects.bulk_create([Chunk(document=doc, text=chunk) for chunk in chunks])

        return Response(
            {"document_id": doc.id, "chunks_created": len(chunks)},
            status=status.HTTP_201_CREATED,
        )

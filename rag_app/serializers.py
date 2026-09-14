from rest_framework import serializers

from .services import retrieval

MAX_DOCUMENT_CHARS = 100_000
MAX_QUESTION_CHARS = 2_000


class DocumentIngestSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    text = serializers.CharField(max_length=MAX_DOCUMENT_CHARS)


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    # Defaults to the file name without its extension.
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)


class AskSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=MAX_QUESTION_CHARS)
    top_k = serializers.IntegerField(required=False, default=5, min_value=1, max_value=10)
    mode = serializers.ChoiceField(choices=retrieval.MODES, required=False, default=retrieval.DEFAULT_MODE)
    rerank = serializers.BooleanField(required=False, default=False)


class AgentSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=MAX_QUESTION_CHARS)

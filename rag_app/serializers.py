from rest_framework import serializers


class DocumentIngestSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    text = serializers.CharField()


class AskSerializer(serializers.Serializer):
    question = serializers.CharField()
    top_k = serializers.IntegerField(required=False, default=5, min_value=1, max_value=20)


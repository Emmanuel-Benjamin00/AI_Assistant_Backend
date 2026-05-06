from rest_framework import serializers


class DocumentIngestSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    text = serializers.CharField()


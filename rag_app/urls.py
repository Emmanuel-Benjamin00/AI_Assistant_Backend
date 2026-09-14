from django.urls import path

from .views import (
    AgentView,
    AskStreamView,
    AskView,
    DocumentIngestView,
    DocumentUploadView,
    HealthView,
    LangChainAskView,
)

urlpatterns = [
    path("documents/", DocumentIngestView.as_view(), name="documents-ingest"),
    path("documents/upload/", DocumentUploadView.as_view(), name="documents-upload"),
    path("ask/", AskView.as_view(), name="ask"),
    path("ask/stream/", AskStreamView.as_view(), name="ask-stream"),
    path("ask/langchain/", LangChainAskView.as_view(), name="ask-langchain"),
    path("agent/", AgentView.as_view(), name="agent"),
    path("health/", HealthView.as_view(), name="health"),
]

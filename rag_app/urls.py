from django.urls import path

from .views import AskView, DocumentIngestView

urlpatterns = [
    path("documents/", DocumentIngestView.as_view(), name="documents-ingest"),
    path("ask/", AskView.as_view(), name="ask"),
]


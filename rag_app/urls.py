from django.urls import path

from .views import DocumentIngestView

urlpatterns = [
    path("documents/", DocumentIngestView.as_view(), name="documents-ingest"),
]


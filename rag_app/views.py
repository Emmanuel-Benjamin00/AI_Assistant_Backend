import json
import logging
from pathlib import Path

from django.db import DatabaseError, connection
from django.db.models import Count
from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document
from .serializers import AgentSerializer, AskSerializer, DocumentIngestSerializer, DocumentUploadSerializer
from .services import agent, indexing, langchain_rag, llm, rag, retrieval
from .services.extraction import MAX_UPLOAD_BYTES, ExtractionError, Section, extract_sections
from .throttling import ClientIPScopedRateThrottle, IngestKeyPermission

logger = logging.getLogger(__name__)

LLM_ERRORS = (llm.LLMConfigError, llm.LLMRequestError)


def _llm_error_response(error: Exception) -> Response:
    if isinstance(error, llm.LLMConfigError):
        logger.error("LLM is not configured: %s", error)
        return Response(
            {"detail": "The AI service is not configured on the server."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    if getattr(error, "status_code", None) == 429:
        return Response(
            {"detail": "The AI service is busy right now. Please try again in a minute."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(
        {"detail": "The AI service request failed. Please try again."},
        status=status.HTTP_502_BAD_GATEWAY,
    )


def _index_response(title, sections, **kwargs) -> Response:
    try:
        doc, chunk_count = indexing.index_document(title, sections, **kwargs)
    except indexing.NothingToIndex as e:
        return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except LLM_ERRORS as e:
        return _llm_error_response(e)
    return Response(
        {"document_id": doc.id, "chunks_created": chunk_count, "source": doc.source},
        status=status.HTTP_201_CREATED,
    )


class DocumentIngestView(APIView):
    permission_classes = [IngestKeyPermission]
    throttle_classes = [ClientIPScopedRateThrottle]
    throttle_scope = "ingest"

    def get_throttles(self):
        # Listing documents is cheap; only rate-limit ingest.
        return super().get_throttles() if self.request.method == "POST" else []

    def get(self, request):
        docs = Document.objects.annotate(chunk_count=Count("chunks")).order_by("-created_at")[:100]
        return Response(
            [
                {
                    "id": d.id,
                    "title": d.title,
                    "source": d.source,
                    "filename": d.filename,
                    "chunks": d.chunk_count,
                    "created_at": d.created_at,
                }
                for d in docs
            ]
        )

    def post(self, request):
        serializer = DocumentIngestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        return _index_response(data["title"], [Section(data["text"])])


class DocumentUploadView(APIView):
    """POST multipart/form-data with `file` (PDF, DOCX, TXT, MD) and an optional `title`."""

    parser_classes = [MultiPartParser]
    permission_classes = [IngestKeyPermission]
    throttle_classes = [ClientIPScopedRateThrottle]
    throttle_scope = "ingest"

    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        upload = serializer.validated_data["file"]
        filename = Path(upload.name).name[:255]

        if upload.size > MAX_UPLOAD_BYTES:
            return Response(
                {"detail": f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        try:
            source, sections = extract_sections(filename, upload.read())
        except ExtractionError as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        title = (serializer.validated_data.get("title") or "").strip() or Path(filename).stem[:255]
        return _index_response(title, sections, source=source, filename=filename)


class AskView(APIView):
    throttle_classes = [ClientIPScopedRateThrottle]
    throttle_scope = "ask"

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            results = retrieval.retrieve(data["question"], data["top_k"], data["mode"], data["rerank"])
            answer = rag.answer(data["question"], results)
        except LLM_ERRORS as e:
            return _llm_error_response(e)

        return Response(
            {"answer": answer, "mode": data["mode"], "sources": rag.serialize_sources(results)},
            status=status.HTTP_200_OK,
        )


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


class AskStreamView(APIView):
    """
    Same as /api/ask/, but the answer streams as Server-Sent Events:

        event: sources   {"mode": ..., "sources": [...]}   once, before any text
        event: token     {"text": "..."}                   many
        event: done      {}                                 once, at the end
        event: error     {"detail": "..."}                  instead of done, if it fails mid-answer

    Validation, retrieval and provider-connection errors happen before streaming starts, so
    they still return normal JSON error responses with the right HTTP status.
    """

    throttle_classes = [ClientIPScopedRateThrottle]
    throttle_scope = "ask"

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        question = data["question"]

        try:
            results = retrieval.retrieve(question, data["top_k"], data["mode"], data["rerank"])
            deltas = rag.answer_stream(question, results) if results else iter([rag.NO_CONTENT_ANSWER])
            # Pull the first delta now: a provider failure then becomes a JSON error, not a broken stream.
            first = next(deltas, None)
        except LLM_ERRORS as e:
            return _llm_error_response(e)

        def events():
            yield _sse("sources", {"mode": data["mode"], "sources": rag.serialize_sources(results)})
            try:
                if first is not None:
                    yield _sse("token", {"text": first})
                for text in deltas:
                    yield _sse("token", {"text": text})
            except llm.LLMRequestError as e:
                logger.warning("Answer stream failed: %s", e)
                yield _sse("error", {"detail": "The answer was interrupted. Please try again."})
                return
            yield _sse("done", {})

        response = StreamingHttpResponse(events(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        # Stops reverse proxies (nginx, some load balancers) from buffering the stream.
        response["X-Accel-Buffering"] = "no"
        return response


class AgentView(APIView):
    """Tool-calling agent: the model chooses between searching, listing and reading documents."""

    throttle_classes = [ClientIPScopedRateThrottle]
    throttle_scope = "agent"

    def post(self, request):
        serializer = AgentSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            result = agent.run_agent(serializer.validated_data["question"])
        except LLM_ERRORS as e:
            return _llm_error_response(e)
        return Response(result, status=status.HTTP_200_OK)


class LangChainAskView(APIView):
    """The /api/ask/ flow rebuilt with LangChain LCEL, for comparison."""

    throttle_classes = [ClientIPScopedRateThrottle]
    throttle_scope = "ask"

    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            result = langchain_rag.ask(data["question"], data["top_k"], data["mode"], data["rerank"])
        except LLM_ERRORS as e:
            return _llm_error_response(e)
        return Response(
            {
                "answer": result["answer"],
                "mode": data["mode"],
                "pipeline": "langchain",
                "sources": rag.serialize_sources(result["results"]),
            },
            status=status.HTTP_200_OK,
        )


class HealthView(APIView):
    """Liveness check for Azure App Service: the app is up and can reach the database."""

    throttle_classes = []

    def get(self, request):
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        except DatabaseError:
            logger.exception("Health check: database unreachable")
            return Response(
                {"status": "unhealthy", "database": "unreachable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response({"status": "ok"})

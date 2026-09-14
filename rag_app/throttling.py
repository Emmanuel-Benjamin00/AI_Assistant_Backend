import hmac

from django.conf import settings
from rest_framework.permissions import SAFE_METHODS, BasePermission
from rest_framework.throttling import ScopedRateThrottle


def client_ip(request) -> str:
    """
    The caller's IP address.

    Azure App Service appends the real client to X-Forwarded-For as "ip:port", so the
    last entry is trusted (earlier entries are client-supplied) and the port is dropped;
    otherwise every new connection would look like a new client.
    """
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
    addr = forwarded.split(",")[-1].strip() if forwarded else request.META.get("REMOTE_ADDR", "")
    return strip_port(addr)


def strip_port(addr: str) -> str:
    if addr.startswith("["):  # [IPv6]:port
        end = addr.find("]")
        return addr[1:end] if end != -1 else addr
    if addr.count(":") == 1:  # IPv4:port
        return addr.split(":", 1)[0]
    return addr  # bare IPv4 or IPv6


class ClientIPScopedRateThrottle(ScopedRateThrottle):
    def get_ident(self, request):
        return client_ip(request)


class IngestKeyPermission(BasePermission):
    """
    Writes need the X-Ingest-Key header to match INGEST_API_KEY, unless ingest is open.

    Ingest is open by default in development and locked in production, so a public demo
    cannot be filled with junk or used to spend embedding credits.
    """

    message = "Adding documents is locked on this deployment. You can still ask questions."

    def has_permission(self, request, view):
        if request.method in SAFE_METHODS or settings.INGEST_OPEN:
            return True
        expected = settings.INGEST_API_KEY
        provided = request.headers.get("X-Ingest-Key", "")
        return bool(expected) and hmac.compare_digest(provided.encode(), expected.encode())

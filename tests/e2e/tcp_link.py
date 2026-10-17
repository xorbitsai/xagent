"""A TCP link that the test can cut and restore, in front of one service."""

from __future__ import annotations

import select
import socket
import socketserver
import threading
from urllib.parse import urlsplit, urlunsplit


class _LinkConnection(socketserver.BaseRequestHandler):
    def handle(self):
        link = self.server
        if not link.enabled.is_set():
            return  # Accepted, then closed at once: the service looks down.
        upstream = socket.create_connection(link.upstream, timeout=3)
        upstream.settimeout(None)
        peers = [self.request, upstream]
        with link.lock:
            link.peers.update(peers)
        try:
            while link.enabled.is_set():
                ready, _, _ = select.select(peers, [], [], 0.1)
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(data)
        except (OSError, ValueError):
            pass  # A cut deliberately closes active sockets from another thread.
        finally:
            with link.lock:
                link.peers.difference_update(peers)
            upstream.close()


class ControllableTcpLink(socketserver.ThreadingTCPServer):
    """Forward ``url``'s host:port; ``url`` keeps credentials and path.

    ``cut()`` closes every forwarded connection and refuses new ones until
    ``restore()``. Only clients given ``self.url`` are affected; the service
    itself stays up for everyone else.
    """

    daemon_threads = True

    def __init__(self, url: str, *, default_port: int):
        parsed = urlsplit(url)
        self.upstream = (parsed.hostname, parsed.port or default_port)
        self.enabled = threading.Event()
        self.enabled.set()
        self.lock = threading.Lock()
        self.peers: set[socket.socket] = set()
        super().__init__(("127.0.0.1", 0), _LinkConnection)
        credentials = parsed.netloc.rpartition("@")[0]
        authority = f"127.0.0.1:{self.server_address[1]}"
        if credentials:
            authority = credentials + "@" + authority
        self.url = urlunsplit(parsed._replace(netloc=authority))
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def cut(self) -> None:
        self.enabled.clear()
        with self.lock:
            for peer in self.peers:
                try:
                    peer.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def restore(self) -> None:
        self.enabled.set()

    def close(self) -> None:
        self.cut()
        self.shutdown()
        self.thread.join(5)
        self.server_close()

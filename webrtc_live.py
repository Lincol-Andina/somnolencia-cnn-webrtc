"""webrtc_live.py
=============================================================================
Estado compartido y configuración ICE para la detección en vivo por WebRTC.

Diseño offline-first:
- Modo "lan": no usa STUN/TURN externos. WebRTC utiliza candidatos host, por
  lo que funciona entre PC y celular en la misma red sin necesidad de internet.
- Modo "internet": usa STUN y, si está configurado, TURN para atravesar NAT.
  El TURN puede definirse por variables de entorno o habilitarse con un
  servidor público de respaldo (best-effort).
=============================================================================
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_STUN_SERVERS = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
    "stun:stun.cloudflare.com:3478",
]

# Servicio público de TURN para pruebas. Es best-effort: puede estar caído,
# bloqueado por la red o tener latencia alta. Solo se usa si el usuario lo
# habilita explícitamente en la interfaz o con USE_PUBLIC_TURN=1.
PUBLIC_TURN_ICE_SERVERS = [
    {
        "urls": [
            "turn:openrelay.metered.ca:80",
            "turn:openrelay.metered.ca:443",
            "turns:openrelay.metered.ca:443",
        ],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    }
]


def _usable_ipv4(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return (
        ip.version == 4
        and not ip.is_loopback
        and not ip.is_unspecified
        and not ip.is_link_local
    )


def _add_candidate(candidates: List[str], ip: str) -> None:
    if _usable_ipv4(ip) and ip not in candidates:
        candidates.append(ip)


def get_lan_ips() -> List[str]:
    """Devuelve todas las IPv4 privadas/utilizables del equipo.

    No envía tráfico: usa la tabla de rutas local y la resolución del hostname.
    El orden prioriza la interfaz de salida por defecto y luego las demás.
    """
    candidates: List[str] = []

    # La tabla de rutas funciona aunque no haya internet, siempre que exista
    # una puerta de enlace local. connect() en UDP no envía paquetes.
    for target in ("8.8.8.8", "10.255.255.255", "192.168.255.255"):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect((target, 1))
                _add_candidate(candidates, str(sock.getsockname()[0]))
            finally:
                sock.close()
        except Exception:
            continue

    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            _add_candidate(candidates, ip)
    except Exception:
        pass

    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        for info in infos:
            _add_candidate(candidates, info[4][0])
    except Exception:
        pass

    private_first = []
    others = []
    for ip in candidates:
        try:
            (private_first if ipaddress.ip_address(ip).is_private else others).append(ip)
        except ValueError:
            continue

    return private_first + others


def get_lan_ip() -> str:
    """Devuelve la IP LAN preferida sin depender de internet."""
    ips = get_lan_ips()
    return ips[0] if ips else "127.0.0.1"


def _split_urls(value: Optional[Any]) -> List[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def _turn_servers(
    turn_urls: Optional[Any] = None,
    turn_username: Optional[str] = None,
    turn_credential: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Construye ICE servers TURN desde parámetros o variables de entorno."""
    urls = _split_urls(turn_urls) or _split_urls(
        os.getenv("TURN_URL") or os.getenv("TURN_URLS")
    )
    if not urls:
        return []

    username = turn_username if turn_username not in (None, "") else os.getenv("TURN_USERNAME", "")
    credential = (
        turn_credential
        if turn_credential not in (None, "")
        else os.getenv("TURN_PASSWORD", os.getenv("TURN_CREDENTIAL", ""))
    )

    server: Dict[str, Any] = {"urls": urls}
    if username:
        server["username"] = username
    if credential:
        server["credential"] = credential
    return [server]


def _public_turn_servers() -> List[Dict[str, Any]]:
    return [dict(server) for server in PUBLIC_TURN_ICE_SERVERS]


def build_rtc_configuration(
    network_mode: str = "lan",
    use_public_turn: bool = False,
    stun_urls: Optional[Any] = None,
    turn_urls: Optional[Any] = None,
    turn_username: Optional[str] = None,
    turn_credential: Optional[str] = None,
) -> Dict[str, Any]:
    """Devuelve la configuración ICE para ``webrtc_streamer``.

    Parámetros
    ----------
    network_mode:
        ``"lan"`` para misma red sin dependencia de internet.
        ``"internet"`` para STUN/TURN en redes distintas o con NAT.
    use_public_turn:
        Añade el TURN público de respaldo si no hay un TURN configurado.
    stun_urls, turn_urls, turn_username, turn_credential:
        Valores explícitos (por ejemplo, desde ``st.secrets``). Si están vacíos,
        se usan las variables de entorno ``STUN_URL``, ``TURN_URL``,
        ``TURN_USERNAME`` y ``TURN_PASSWORD``/``TURN_CREDENTIAL``.
    """
    mode = (network_mode or "lan").strip().lower()
    ice_servers: List[Dict[str, Any]] = []

    if mode == "internet":
        servers = _split_urls(stun_urls) or _split_urls(os.getenv("STUN_URL")) or list(
            DEFAULT_STUN_SERVERS
        )
        ice_servers.append({"urls": servers})

        turn_servers = _turn_servers(turn_urls, turn_username, turn_credential)
        if turn_servers:
            ice_servers.extend(turn_servers)
        elif use_public_turn or os.getenv("USE_PUBLIC_TURN", "0") == "1":
            ice_servers.extend(_public_turn_servers())
    else:
        # Modo LAN: si el administrador definió un TURN explícito, se respeta
        # para redes WiFi con aislamiento de clientes; si no, se usan solo
        # candidatos host y no se contacta ningún servidor externo.
        ice_servers.extend(_turn_servers(turn_urls, turn_username, turn_credential))

    return {"iceServers": ice_servers, "iceTransportPolicy": "all"}


@dataclass
class LiveState:
    """Estado thread-safe compartido entre el callback WebRTC y la interfaz."""

    threshold: float = 0.50
    window: int = 10
    sound_enabled: bool = True
    prob: float = 0.0
    counter: int = 0
    alert_active: bool = False
    fps: float = 0.0
    event_id: int = 0
    last_ts: float = 0.0
    frames: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def configure(
        self,
        threshold: Optional[float] = None,
        window: Optional[int] = None,
        sound_enabled: Optional[bool] = None,
    ) -> None:
        with self._lock:
            if threshold is not None:
                self.threshold = float(threshold)
            if window is not None:
                self.window = max(1, int(window))
            if sound_enabled is not None:
                self.sound_enabled = bool(sound_enabled)

    def update(self, prob: float, now: Optional[float] = None) -> Dict[str, Any]:
        """Actualiza el acumulador temporal y devuelve una instantánea."""
        timestamp = time.time() if now is None else float(now)
        value = float(prob)

        with self._lock:
            if self.last_ts > 0.0:
                delta = timestamp - self.last_ts
                if delta > 0.0:
                    instant_fps = 1.0 / delta
                    self.fps = instant_fps if self.fps <= 0.0 else (0.85 * self.fps + 0.15 * instant_fps)
            self.last_ts = timestamp
            self.frames += 1
            self.prob = value

            if value >= self.threshold:
                self.counter += 1
            else:
                self.counter = max(0, self.counter - 1)

            was_active = self.alert_active
            self.alert_active = self.counter >= self.window
            if self.alert_active and not was_active:
                self.event_id += 1

            return self._snapshot_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "window": self.window,
            "sound_enabled": self.sound_enabled,
            "prob": self.prob,
            "counter": self.counter,
            "alert_active": self.alert_active,
            "fps": self.fps,
            "event_id": self.event_id,
            "last_ts": self.last_ts,
            "frames": self.frames,
        }

    def is_streaming(self, timeout: float = 2.0) -> bool:
        with self._lock:
            return self.last_ts > 0.0 and (time.time() - self.last_ts) <= timeout

    def reset(self) -> None:
        with self._lock:
            self.prob = 0.0
            self.counter = 0
            self.alert_active = False
            self.fps = 0.0
            self.last_ts = 0.0
            self.frames = 0

"""browser_alarm.py
=============================================================================
Alarma sonora ejecutada exclusivamente en el navegador del usuario.

- No usa ``winsound``, ``pygame`` ni audio del servidor.
- Funciona sin internet: el WAV se embebe en base64 y se reproduce con Web
  Audio API.
- En modos en vivo (WebRTC) un marcador oculto indica alerta activa y el JS
  repite el tono mientras la somnolencia persista.
- En modos estáticos (imagen/dataset) se dispara un único tono por evento.
- Incluye fallback a oscilador si el navegador no puede decodificar el WAV.
=============================================================================
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Optional, Union

import streamlit as st
import streamlit.components.v1 as components


ALARM_SIGNAL_ID = "ddd-alarm-signal"

_RUNTIME_TEMPLATE = """
<script>
(function () {
    var parentWin = window.parent;
    if (!parentWin) return;
    if (parentWin.__dddAlarmInstalled) return;
    parentWin.__dddAlarmInstalled = true;

    var doc = parentWin.document;
    var state = {
        ctx: null,
        buffer: null,
        decoding: false,
        b64: "__ALARM_B64__",
        lastRepeat: 0,
        lastEvent: null,
        plays: 0
    };

    function ensureContext() {
        if (state.ctx) return state.ctx;
        var AC = parentWin.AudioContext || parentWin.webkitAudioContext;
        if (!AC) return null;
        try {
            state.ctx = new AC();
        } catch (e) {
            return null;
        }
        return state.ctx;
    }

    function decode() {
        if (state.buffer || state.decoding || !state.b64) return;
        var ctx = ensureContext();
        if (!ctx || !ctx.decodeAudioData) return;
        state.decoding = true;
        try {
            var raw = parentWin.atob(state.b64);
            var bytes = new Uint8Array(raw.length);
            for (var i = 0; i < raw.length; i++) {
                bytes[i] = raw.charCodeAt(i);
            }
            ctx.decodeAudioData(
                bytes.buffer.slice(0),
                function (decoded) {
                    state.buffer = decoded;
                    state.decoding = false;
                },
                function () {
                    state.decoding = false;
                }
            );
        } catch (e) {
            state.decoding = false;
        }
    }

    function unlock() {
        var ctx = ensureContext();
        if (!ctx) return Promise.resolve(false);
        decode();
        if (ctx.state === "suspended") {
            return ctx.resume().then(function () { return true; }).catch(function () { return false; });
        }
        return Promise.resolve(true);
    }

    function fallbackBeep(ctx) {
        try {
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.type = "square";
            osc.frequency.value = 2200;
            gain.gain.value = 0.28;
            osc.connect(gain);
            gain.connect(ctx.destination);
            var now = ctx.currentTime;
            osc.start(now);
            osc.stop(now + 0.35);
        } catch (e) {}
    }

    function play() {
        var ctx = ensureContext();
        if (!ctx) return;
        var run = function () {
            state.plays += 1;
            if (state.buffer) {
                try {
                    var src = ctx.createBufferSource();
                    var gain = ctx.createGain();
                    src.buffer = state.buffer;
                    gain.gain.value = 0.95;
                    src.connect(gain);
                    gain.connect(ctx.destination);
                    src.start();
                } catch (e) {
                    fallbackBeep(ctx);
                }
            } else {
                fallbackBeep(ctx);
                decode();
            }
        };
        if (ctx.state === "suspended") {
            ctx.resume().then(run).catch(function () {});
        } else {
            run();
        }
    }

    parentWin.__dddAlarm = {
        unlock: unlock,
        play: play,
        getStatus: function () {
            return {
                contextState: state.ctx ? state.ctx.state : null,
                bufferReady: !!state.buffer,
                decoding: state.decoding,
                lastEvent: state.lastEvent,
                lastRepeat: state.lastRepeat,
                plays: state.plays
            };
        },
        playOnce: function (eventId) {
            if (state.lastEvent === eventId) return;
            state.lastEvent = eventId;
            unlock().then(function () { play(); });
        }
    };

    ["pointerdown", "touchstart", "keydown"].forEach(function (evt) {
        doc.addEventListener(evt, function () { unlock(); }, { passive: true });
    });

    if (!parentWin.__dddAlarmTimer) {
        parentWin.__dddAlarmTimer = parentWin.setInterval(function () {
            try {
                var el = doc.getElementById("__SIGNAL_ID__");
                if (!el) return;
                var active = el.getAttribute("data-active") === "1";
                var sound = el.getAttribute("data-sound") !== "0";
                if (!active || !sound) {
                    state.lastRepeat = 0;
                    return;
                }
                var now = Date.now();
                if (now - state.lastRepeat >= 1000) {
                    state.lastRepeat = now;
                    unlock().then(function () { play(); });
                }
            } catch (e) {}
        }, 250);
    }

    try {
        ensureContext();
        decode();
    } catch (e) {}
})();
</script>
"""


def _alarm_base64(alarm_path: Union[str, Path] = "alarm.wav") -> str:
    try:
        data = Path(alarm_path).read_bytes()
        return base64.b64encode(data).decode("ascii")
    except Exception:
        return ""


def inject_alarm_runtime(alarm_path: Union[str, Path] = "alarm.wav") -> None:
    """Inyecta el runtime de audio una sola vez en el documento padre."""
    b64 = _alarm_base64(alarm_path)
    html = (
        _RUNTIME_TEMPLATE.replace("__ALARM_B64__", b64)
        .replace("__SIGNAL_ID__", ALARM_SIGNAL_ID)
    )
    components.html(html, height=0)


def render_alarm_signal(
    active: bool,
    sound_enabled: bool = True,
    event_id: Union[int, str] = 0,
) -> None:
    """Marcador oculto que el JS del navegador consulta periódicamente."""
    st.markdown(
        (
            f'<div id="{ALARM_SIGNAL_ID}" '
            f'data-active="{1 if active else 0}" '
            f'data-sound="{1 if sound_enabled else 0}" '
            f'data-event="{event_id}" '
            'style="display:none"></div>'
        ),
        unsafe_allow_html=True,
    )


def trigger_alarm_once(event_id: Optional[Union[int, str]] = None) -> None:
    """Dispara un único tono en el navegador para un evento estático."""
    if event_id is None:
        event_id = f"{time.time_ns()}"

    payload = json.dumps(str(event_id))
    html = (
        "<script>"
        "(function () {"
        f"var id = {payload};"
        "var tries = 0;"
        "function go() {"
        "  var target = window.parent;"
        "  if (target && target.__dddAlarm) {"
        "    target.__dddAlarm.playOnce(id);"
        "  } else if (tries++ < 40) {"
        "    setTimeout(go, 50);"
        "  }"
        "}"
        "go();"
        "})();"
        "</script>"
    )
    components.html(html, height=0)

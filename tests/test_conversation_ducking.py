"""Unit tests for conversation-aware ducking helpers (prompt 7)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_cowatcher.config import Settings
from ai_cowatcher.main import create_app

_ROOT = Path(__file__).resolve().parents[1]
_DUCK_JS = _ROOT / "ai_cowatcher" / "web" / "conversation_ducking.js"


def _node_available() -> bool:
    try:
        subprocess.run(
            ["node", "-e", "process.exit(0)"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False


def _run_duck_js(snippet: str) -> dict:
    """Evaluate pure helpers via Node (CommonJS export)."""
    script = f"""
const duck = require({json.dumps(str(_DUCK_JS))});
const result = (() => {{ {snippet} }})();
process.stdout.write(JSON.stringify(result));
"""
    proc = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(proc.stdout)


@pytest.mark.skipif(not _node_available(), reason="node required for ducking pure JS unit tests")
def test_compute_rms_silence_and_tone():
    out = _run_duck_js(
        """
        const silent = new Float32Array(64);
        const loud = new Float32Array(64);
        for (let i = 0; i < 64; i++) loud[i] = i % 2 === 0 ? 0.5 : -0.5;
        return {
          silent: duck.computeRms(silent),
          loud: duck.computeRms(loud),
        };
        """
    )
    assert out["silent"] == 0
    assert out["loud"] > 0.4


@pytest.mark.skipif(not _node_available(), reason="node required for ducking pure JS unit tests")
def test_target_gains_duck_and_ambient():
    out = _run_duck_js(
        """
        const hard = duck.targetGains({
          enabled: true,
          reasons: ["ask"],
          vadSpeechActive: false,
        });
        const vad = duck.targetGains({
          enabled: true,
          reasons: [],
          vadSpeechActive: true,
        });
        const ambient = duck.targetGains({
          enabled: true,
          reasons: ["ambient"],
          vadSpeechActive: false,
        });
        const off = duck.targetGains({
          enabled: false,
          reasons: ["ask"],
          vadSpeechActive: true,
        });
        return { hard, vad, ambient, off, cfg: duck.DUCK_CONFIG };
        """
    )
    assert out["hard"]["program"] == pytest.approx(out["cfg"]["DUCK_GAIN"])
    assert out["hard"]["tts"] == pytest.approx(out["cfg"]["DUCK_GAIN"])
    assert out["vad"]["program"] == pytest.approx(out["cfg"]["DUCK_GAIN"])
    assert out["ambient"]["program"] == pytest.approx(out["cfg"]["AMBIENT_GAIN"])
    assert out["ambient"]["tts"] == pytest.approx(1.0)
    assert out["off"]["program"] == 1.0
    assert out["off"]["tts"] == 1.0


@pytest.mark.skipif(not _node_available(), reason="node required for ducking pure JS unit tests")
def test_vad_threshold_and_hangover():
    out = _run_duck_js(
        """
        const floor = duck.updateNoiseFloor(0.01, 0.01, false);
        const thr = duck.vadThreshold(floor);
        const hang = duck.hangoverElapsed(true, 0, duck.DUCK_CONFIG.SPEECH_HANGOVER_MS + 1);
        const notYet = duck.hangoverElapsed(true, 0, 10);
        return {
          thr,
          hang,
          notYet,
          min: duck.DUCK_CONFIG.VAD_MIN_THRESHOLD,
          micEcho: duck.MIC_CONSTRAINTS.audio.echoCancellation,
        };
        """
    )
    assert out["thr"] >= out["min"]
    assert out["hang"] is True
    assert out["notYet"] is False
    assert out["micEcho"] is True


def test_watch_serves_ducking_helpers_and_session_hooks():
    app = create_app(Settings(MOCK_MODE=True))
    client = TestClient(app)

    js = client.get("/watch/conversation_ducking.js")
    assert js.status_code == 200
    assert "targetGains" in js.text
    assert "computeRms" in js.text
    assert "echoCancellation" in js.text
    assert "DUCK_GAIN" in js.text

    page = client.get("/watch")
    assert page.status_code == 200
    body = page.text
    assert "/watch/conversation_ducking.js" in body
    assert "ensureAudioSession" in body
    assert "applyUnifiedGains" in body
    assert "createMediaElementSource" in body
    assert "programGain" in body
    assert "vadSpeechActive" in body
    assert "processVadRms" in body
    assert "setDuckReason" in body
    assert "duckWhileTalking" in body
    # Auto-listen stays off by default (testing-friendly).
    assert 'id="autoListenMic"' in body
    assert "cowatcher.auto_listen_mic" in body
    # Must not barge-in cancel TTS solely for VAD ducking.
    assert "does not cancel speech" in body or "Does not cancel" in body

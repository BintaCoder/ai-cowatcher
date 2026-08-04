"""Unit tests for SmartAIDucker state machine + /watch wiring."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
_DUCKER_JS = _ROOT / "ai_cowatcher" / "web" / "smart_ai_ducker.js"
_WATCH_HTML = _ROOT / "ai_cowatcher" / "web" / "watch.html"


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


def _run_ducker_js(snippet: str) -> dict:
    """Evaluate SmartAIDucker via Node (CommonJS export) with a fake media element."""
    script = f"""
const api = require({json.dumps(str(_DUCKER_JS))});
const {{ SmartAIDucker, DEFAULTS, MIC_CONSTRAINTS, timeConstantFor, findAirPodsOutput, UNIFIED_SAMPLE_RATE }} = api;

class FakeMedia {{
  constructor() {{
    this.volume = 1;
    this.crossOrigin = null;
    this.paused = false;
    this.ended = false;
    this.muted = false;
    this.readyState = 2;
  }}
  play() {{ return Promise.resolve(); }}
}}

Promise.resolve()
  .then(() => {{ {snippet} }})
  .then((result) => {{
    process.stdout.write(JSON.stringify(result));
  }})
  .catch((err) => {{
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  }});
"""
    proc = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(proc.stdout)


def _watch_asset_client() -> TestClient:
    """Minimal app: watch HTML + SmartAIDucker JS only (no full create_app)."""
    from ai_cowatcher.api.watch_routes import (
        watch_page,
        watch_smart_ai_ducker_js,
        watch_conversation_ducking_js,
    )

    app = FastAPI()
    app.add_api_route("/watch", watch_page, methods=["GET"])
    app.add_api_route("/watch/smart_ai_ducker.js", watch_smart_ai_ducker_js, methods=["GET"])
    app.add_api_route(
        "/watch/conversation_ducking.js",
        watch_conversation_ducking_js,
        methods=["GET"],
    )
    return TestClient(app)


@pytest.mark.skipif(not _node_available(), reason="node required for SmartAIDucker unit tests")
def test_defaults_and_mic_constraints():
    out = _run_ducker_js(
        """
        return {
          duckGain: DEFAULTS.duckGain,
          holdMs: DEFAULTS.holdMs,
          duckAttack: DEFAULTS.duckAttackSec,
          recovery: DEFAULTS.recoverySec,
          btOffset: DEFAULTS.bluetoothOffsetMs,
          sampleRate: DEFAULTS.sampleRate,
          echo: MIC_CONSTRAINTS.audio.echoCancellation,
          ns: MIC_CONSTRAINTS.audio.noiseSuppression,
          agc: MIC_CONSTRAINTS.audio.autoGainControl,
          latencyIdeal: MIC_CONSTRAINTS.audio.latency.ideal,
          tau: timeConstantFor(0.15),
        };
        """
    )
    assert out["duckGain"] == pytest.approx(0.15)
    assert out["holdMs"] == 1500
    assert out["duckAttack"] == pytest.approx(0.15)
    assert out["recovery"] == pytest.approx(0.6)
    assert out["btOffset"] == 150
    assert out["sampleRate"] == 48000
    assert out["echo"] is True
    assert out["ns"] is True
    assert out["agc"] is False
    assert out["latencyIdeal"] == pytest.approx(0.01)
    assert out["tau"] == pytest.approx(0.05)


@pytest.mark.skipif(not _node_available(), reason="node required for SmartAIDucker unit tests")
def test_state_matrix_user_or_ai_or_pipeline():
    out = _run_ducker_js(
        """
        const d = new SmartAIDucker(new FakeMedia(), { holdMs: 1500 });
        d.onUserSpeechStart();
        const afterUser = d.getState();
        d.onAISpeechStart();
        const both = d.getState();
        d.onUserSpeechEnd();
        const aiOnly = d.getState();
        d.onAISpeechEnd();
        const recovering = d.getState();
        d.setPipelineActive(true);
        const pipeline = d.getState();
        d.setPipelineActive(false);
        const afterPipe = d.getState();
        return {
          afterUser,
          both,
          aiOnly,
          recovering,
          pipeline,
          afterPipe,
          duckGain: d.opts.duckGain,
        };
        """
    )
    assert out["afterUser"]["userSpeaking"] is True
    assert out["afterUser"]["ducked"] is True
    assert out["afterUser"]["gain"] == pytest.approx(out["duckGain"])
    assert out["both"]["userSpeaking"] is True and out["both"]["aiSpeaking"] is True
    assert out["both"]["ducked"] is True
    assert out["aiOnly"]["userSpeaking"] is False
    assert out["aiOnly"]["aiSpeaking"] is True
    assert out["aiOnly"]["ducked"] is True
    assert out["recovering"]["ducked"] is False
    assert out["recovering"]["recovering"] is True
    assert out["pipeline"]["pipelineActive"] is True
    assert out["pipeline"]["ducked"] is True
    assert out["afterPipe"]["pipelineActive"] is False
    assert out["afterPipe"]["recovering"] is True


@pytest.mark.skipif(not _node_available(), reason="node required for SmartAIDucker unit tests")
def test_crosstalk_cancels_recover_generation():
    out = _run_ducker_js(
        """
        const d = new SmartAIDucker(new FakeMedia(), { holdMs: 50 });
        d.onUserSpeechStart();
        d.onUserSpeechEnd();
        const genDuringHold = d._generation;
        const recovering = d.getState().recovering;
        d.onUserSpeechStart();
        const after = d.getState();
        return {
          recovering,
          genDuringHold,
          genAfter: d._generation,
          ducked: after.ducked,
          recoveringAfter: after.recovering,
          gain: after.gain,
          duckGain: d.opts.duckGain,
        };
        """
    )
    assert out["recovering"] is True
    assert out["genAfter"] > out["genDuringHold"]
    assert out["ducked"] is True
    assert out["recoveringAfter"] is False
    assert out["gain"] == pytest.approx(out["duckGain"])


@pytest.mark.skipif(not _node_available(), reason="node required for SmartAIDucker unit tests")
def test_hold_then_recover_gain():
    out = _run_ducker_js(
        """
        return new Promise((resolve) => {
          const d = new SmartAIDucker(new FakeMedia(), { holdMs: 40, recoverySec: 0.05 });
          d.onAISpeechStart();
          d.onAISpeechEnd();
          setTimeout(() => {
            const st = d.getState();
            resolve({
              recovering: st.recovering,
              ducked: st.ducked,
              gain: st.gain,
              full: d.opts.fullGain,
            });
          }, 80);
        });
        """
    )
    assert out["recovering"] is False
    assert out["ducked"] is False
    assert out["gain"] == pytest.approx(out["full"])


@pytest.mark.skipif(not _node_available(), reason="node required for SmartAIDucker unit tests")
def test_bluetooth_recovery_padding_and_sink_bind():
    out = _run_ducker_js(
        """
        const d = new SmartAIDucker(new FakeMedia(), {
          holdMs: 1500,
          bluetoothOffsetMs: 200,
        });
        const base = d.getEffectiveHoldMs();
        d.setBluetoothPadding(true);
        const padded = d.getEffectiveHoldMs();
        const sinks = [];
        d.mediaElement.setSinkId = (id) => { sinks.push('el:' + id); return Promise.resolve(); };
        // Simulate AudioContext.setSinkId without real Web Audio.
        d.ctx = { setSinkId: (id) => { sinks.push('ctx:' + id); return Promise.resolve(); } };
        return d.bindToAirPods('airpods-device-1').then(() => {
          const st = d.getState();
          return {
            base,
            padded,
            effective: st.effectiveHoldMs,
            padding: st.bluetoothPadding,
            sinkId: st.sinkId,
            sinks,
            unified: UNIFIED_SAMPLE_RATE,
          };
        });
        """
    )
    assert out["base"] == 1500
    assert out["padded"] == 1700
    assert out["effective"] == 1700
    assert out["padding"] is True
    assert out["sinkId"] == "airpods-device-1"
    assert "el:airpods-device-1" in out["sinks"]
    assert "ctx:airpods-device-1" in out["sinks"]
    assert out["unified"] == 48000


@pytest.mark.skipif(not _node_available(), reason="node required for SmartAIDucker unit tests")
def test_find_airpods_output_helper():
    out = _run_ducker_js(
        """
        const devices = [
          { kind: 'audioinput', deviceId: 'mic', label: 'MacBook Mic' },
          { kind: 'audiooutput', deviceId: 'speakers', label: 'MacBook Speakers' },
          { kind: 'audiooutput', deviceId: 'buds', label: 'Binta\\'s AirPods Pro' },
        ];
        const hit = findAirPodsOutput(devices);
        const none = findAirPodsOutput([
          { kind: 'audiooutput', deviceId: 'spk', label: 'Built-in Output' },
        ]);
        return { id: hit && hit.deviceId, none };
        """
    )
    assert out["id"] == "buds"
    assert out["none"] is None


def test_watch_html_wires_smart_ai_ducker():
    body = _WATCH_HTML.read_text(encoding="utf-8")
    assert "/watch/smart_ai_ducker.js" in body
    assert "conversation_ducking.js" not in body
    assert "ensureSmartDucker" in body
    assert "SmartAIDucker" in body
    assert "maybeBindAirPodsOutput" in body
    assert "bindToAirPods" in body
    assert "DUCK_BT_OFFSET_MS" in body
    assert "DUCK_SAMPLE_RATE" in body
    assert "processVadRms" not in body
    assert "vadSpeechActive" not in body
    assert "setDuckReason" in body
    assert "startPushToTalk" in body
    assert 'setDuckReason("ptt"' in body or "setDuckReason(\"ptt\"" in body
    assert "does not cancel" in body.lower()
    assert "cowatcher.auto_listen_mic" in body
    assert 'id="holdTalkBtn"' in body


def test_watch_serves_smart_ai_ducker_asset():
    client = _watch_asset_client()
    js = client.get("/watch/smart_ai_ducker.js")
    assert js.status_code == 200
    assert "SmartAIDucker" in js.text
    assert "onUserSpeechStart" in js.text
    assert "onAISpeechStart" in js.text
    assert "setPipelineActive" in js.text
    assert "cancelScheduledValues" in js.text
    assert "setTargetAtTime" in js.text
    assert "echoCancellation" in js.text
    assert "autoGainControl: false" in js.text or "autoGainControl:!1" in js.text.replace(" ", "")
    assert "bindToAirPods" in js.text
    assert "setSinkId" in js.text
    assert "sampleRate" in js.text
    assert "bluetoothOffsetMs" in js.text
    assert "holdMs" in js.text

    page = client.get("/watch")
    assert page.status_code == 200
    assert "/watch/smart_ai_ducker.js" in page.text
    assert "processVadRms" not in page.text
    assert "maybeBindAirPodsOutput" in page.text
"""Slack alert when a prediction says an engine has used (about) 80% of its life.

Estimated life used = cycle / (cycle + predicted RUL): the share of the engine's estimated total life already consumed. For a test
engine the true total life is unknown, so this is always an estimate from the model, never a measurement.

Alerts post through a Slack incoming webhook whose URL is read from an environment variable (never from code, config or logs). Without
it the service runs in dry-run mode: it decides and records what it would send, but sends nothing. An alert problem must never break a
prediction, so the tool layer calls ``on_prediction`` inside a try/except.

An engine alerts once per crossing: it re-arms only after the estimate falls below ``rearm_below`` (hysteresis, so a slider dragged
around 80% does not post repeatedly). The official test labels and true RUL are never read here or included in a message.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class AlertError(RuntimeError):
    """The alert could not be delivered (the message never includes the webhook URL)."""


def life_used(cycle: float, predicted_rul: float) -> float:
    """Share of the estimated total life already used: cycle / (cycle + predicted RUL). 0 when both are 0."""
    total = float(cycle) + float(predicted_rul)
    return 0.0 if total <= 0 else float(cycle) / total


class SlackWebhookSender:
    def __init__(self, url: str, timeout: float = 5.0):
        if not url.startswith("https://hooks.slack.com/"):
            raise ValueError("The Slack webhook URL must start with https://hooks.slack.com/")
        self._url, self.timeout = url, timeout

    def send(self, text: str) -> None:
        req = urllib.request.Request(self._url, data=json.dumps({"text": text}).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (host checked in __init__)
                status, body = getattr(r, "status", 200), r.read().decode("utf-8", "replace").strip()
        except Exception as e:  # never echo the exception text: it can contain the webhook URL
            raise AlertError(f"Slack webhook request failed ({type(e).__name__})") from None
        if status != 200 or body != "ok":
            raise AlertError(f"Slack rejected the alert (HTTP {status})")


@dataclass
class AlertEvent:
    status: str  # below_threshold | alerted | dry_run | already_alerted | send_failed
    key: str
    cycle: int
    life_used: float
    message: Optional[str] = None
    error: Optional[str] = None


class AlertService:
    def __init__(self, threshold: float = 0.80, rearm_below: float = 0.75, sender: Optional[Any] = None, state_path: Optional[Path] = None,
                 log_path: Optional[Path] = None, config_note: str = ""):
        if not 0 < rearm_below <= threshold <= 1:
            raise ValueError("Need 0 < rearm_below <= threshold <= 1")
        self.threshold, self.rearm_below, self.sender = threshold, rearm_below, sender
        self.state_path, self.log_path, self.config_note = state_path, log_path, config_note
        self.last_event: Optional[AlertEvent] = None
        self._dry: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self._fired: Dict[str, Dict[str, Any]] = self._load()

    # ------------------------------------------------------------------ construction and status
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "AlertService":
        from Agent.config import AGENT_DIR, resolve
        a = cfg.get("alerts", {})
        base = resolve(cfg["logging"]["trace_file"]).parent if cfg.get("logging", {}).get("trace_file") else AGENT_DIR / "logs"
        url, sender, note = os.environ.get(a.get("webhook_env", "SLACK_WEBHOOK_URL"), "").strip(), None, ""
        if url:
            try:
                sender = SlackWebhookSender(url)
            except ValueError as e:
                note = str(e)
        return cls(float(a.get("life_used_threshold", 0.80)), float(a.get("rearm_below", 0.75)), sender, base / "alert_state.json", base / "alerts.jsonl", note)

    @property
    def mode(self) -> str:
        return "slack" if self.sender is not None else "dry-run"

    def status_text(self) -> str:
        pct = f"{self.threshold * 100:.0f}%"
        if self.sender is not None:
            return f"Slack alerts ON: a message is posted once when an engine's estimated life used reaches {pct}."
        why = self.config_note or f"no webhook URL found in the environment"
        return f"Slack alerts in DRY-RUN ({why}): alerts at {pct} are decided and logged, not sent."

    # ------------------------------------------------------------------ the hook called after every prediction
    def on_prediction(self, ctx, out) -> AlertEvent:
        return self.evaluate(ctx.dataset_split, ctx.engine_id, ctx.cutoff_cycle, out.predicted_rul_cycles, out.support_rul_cycles,
                             out.demo_band, ctx.model_version, out.data_role)

    def evaluate(self, split: str, engine_id: int, cycle: int, predicted_rul: float, support_rul: float, band: str, model_version: str,
                 data_role: str = "") -> AlertEvent:
        pct, key = life_used(cycle, predicted_rul), f"{split}:{engine_id}:{model_version}"
        with self._lock:
            if pct < self.rearm_below:  # re-arm once the estimate is clearly back below the threshold
                self._dry.pop(key, None)
                if self._fired.pop(key, None) is not None:
                    self._save()
            if pct < self.threshold:
                return AlertEvent("below_threshold", key, cycle, pct)
            if key in self._fired or self._dry.get(key):
                ev = AlertEvent("already_alerted", key, cycle, pct)
                self.last_event = ev
                return ev
            text = self._message(split, engine_id, cycle, pct, predicted_rul, support_rul, band, model_version, data_role)
            if self.sender is None:
                self._dry[key] = True
                ev = AlertEvent("dry_run", key, cycle, pct, message=text)
            else:
                try:
                    self.sender.send(text)
                except Exception as e:
                    ev = AlertEvent("send_failed", key, cycle, pct, message=text, error=str(e))  # not marked fired, so the next prediction retries
                else:
                    self._fired[key] = {"cycle": cycle, "at": self._now()}
                    self._save()
                    ev = AlertEvent("alerted", key, cycle, pct, message=text)
            self.last_event = ev
            self._log(ev)
            return ev

    # ------------------------------------------------------------------ one summary for a whole fleet scan
    def send_fleet_summary(self, fingerprint: str, source: str, flagged_rows: list, n_scored: int, model_version: str, force: bool = False) -> AlertEvent:
        """One Slack message listing every engine at or above the threshold. Sent once per scanned file and model version unless ``force``.

        ``flagged_rows`` are dicts with engine_id, last_cycle, life_used_pct, predicted_rul and band, already sorted most-used first.
        """
        key = f"fleet:{fingerprint}:{model_version}"
        top = flagged_rows[0]["life_used_pct"] / 100.0 if flagged_rows else 0.0
        if not flagged_rows:
            return AlertEvent("below_threshold", key, 0, top)
        with self._lock:
            if not force and (key in self._fired or self._dry.get(key)):
                ev = AlertEvent("already_alerted", key, 0, top)
                self.last_event = ev
                return ev
            text = self._fleet_message(source, flagged_rows, n_scored, model_version)
            if self.sender is None:
                self._dry[key] = True
                ev = AlertEvent("dry_run", key, 0, top, message=text)
            else:
                try:
                    self.sender.send(text)
                except Exception as e:
                    ev = AlertEvent("send_failed", key, 0, top, message=text, error=str(e))
                else:
                    self._fired[key] = {"engines": len(flagged_rows), "at": self._now()}
                    self._save()
                    ev = AlertEvent("alerted", key, 0, top, message=text)
            self.last_event = ev
            self._log(ev)
            return ev

    def _fleet_message(self, source: str, rows: list, n_scored: int, version: str, max_listed: int = 15) -> str:
        pct = f"{self.threshold * 100:.0f}%"
        lines = [f":warning: *Fleet scan: {len(rows)} of {n_scored} engines at or above {pct} estimated life used (demo)* · {source}",
                 f"Model `{version}` · inspection review flagged for:"]
        for r in rows[:max_listed]:
            lines.append(f"• Engine {r['engine_id']}: {r['life_used_pct']:.0f}% at cycle {r['last_cycle']} · predicted RUL {r['predicted_rul']:.1f} cycles · {r['band']}")
        if len(rows) > max_listed:
            lines.append(f"…and {len(rows) - max_listed} more (see the downloadable table in the app).")
        lines.append("_Estimated life used = cycle ÷ (cycle + predicted RUL). This is a screening flag from a demo model, not a maintenance decision: estimates "
                     "typically err by about 15 cycles at an engine's last cycle and about 30 at arbitrary cycles._")
        return "\n".join(lines)

    # ------------------------------------------------------------------ internals
    def _message(self, split, engine_id, cycle, pct, pred, sup, band, version, role) -> str:
        lines = [f":warning: *Inspection review flagged (demo)* · {split} engine {engine_id}",
                 f"Estimated life used: *{pct * 100:.0f}%* at cycle {cycle} (alert threshold {self.threshold * 100:.0f}%)",
                 f"Predicted RUL: {pred:.1f} cycles · support model {sup:.1f} · demo band {band}",
                 f"Model `{version}`"]
        if role == "train_in_sample":
            lines.append("_This engine was part of the model's training data, so the estimate is optimistic._")
        lines.append("_Estimated life used = cycle ÷ (cycle + predicted RUL). This is a screening flag from a demo model, not a maintenance decision: estimates "
                     "typically err by about 15 cycles at an engine's last cycle and about 30 at arbitrary cycles._")
        return "\n".join(lines)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            return json.loads(self.state_path.read_text()) if self.state_path and self.state_path.exists() else {}
        except Exception:
            return {}  # a corrupt state file means "nothing alerted yet"

    def _save(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._fired))
            tmp.replace(self.state_path)
        except OSError:
            pass

    def _log(self, ev: AlertEvent) -> None:
        if not self.log_path:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps({"at": self._now(), "status": ev.status, "key": ev.key, "cycle": ev.cycle, "life_used": round(ev.life_used, 4),
                                    "error": ev.error}) + "\n")
        except OSError:
            pass

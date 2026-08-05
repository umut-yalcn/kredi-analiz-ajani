"""Guard: agent ile veri arasindaki zorunlu gecis noktasi.

Agentic sistemlerde model, hangi sorgunun calisacagina calisma aninda karar verir.
Bu esneklik kredi verisi gibi bir alanda dogrudan risk demektir: prompt injection,
halusinasyon ya da sadece kotu bir plan, kisisel veriyi disari sizdirabilir.

Bu katman modelin niyetine guvenmez. Her arac cagrisi buradan gecer ve dort kontrol
uygulanir:

  1. Kolon izni     - PII kolonlar analiz katmanina hicbir kosulda gecmez.
  2. k-anonimlik    - k'dan az satira dayanan hicbir toplulastirma dondurulmez.
  3. PII maskeleme  - cikti metninde PII deseni kalirsa maskelenir.
  4. Denetim kaydi  - her karar, gerekcesiyle birlikte kayda gecer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schema import ANALYZABLE_COLUMNS, BY_NAME, PII_COLUMNS

# Bir toplulastirmanin dondurulebilmesi icin gereken en az satir sayisi.
# Bunun altinda kalan gruplar tek bir kisiyi isaret edebilecegi icin bastirilir.
K_ANONYMITY_THRESHOLD = 20


class GuardViolation(Exception):
    """Guard bir istegi reddettiginde firlatilir. Agent bunu hata olarak gorur ve plan degistirir."""


@dataclass
class AuditEntry:
    timestamp: str
    action: str
    columns: tuple[str, ...]
    allowed: bool
    reason: str


@dataclass
class Guard:
    """Tek bir istek boyunca yasayan guvenlik baglami."""

    audit_log: list[AuditEntry] = field(default_factory=list)

    # --- 1. Kolon izni ---------------------------------------------------

    def check_columns(self, action: str, columns: list[str]) -> None:
        """Istenen kolonlarin analize acik olup olmadigini dogrular."""
        unknown = [c for c in columns if c not in BY_NAME]
        if unknown:
            self._record(action, columns, False, f"Bilinmeyen kolon: {', '.join(unknown)}")
            raise GuardViolation(
                f"Su kolonlar veri sozlugunde yok: {', '.join(unknown)}. "
                f"Kullanilabilir kolonlar: {', '.join(ANALYZABLE_COLUMNS)}"
            )

        blocked = [c for c in columns if c in PII_COLUMNS]
        if blocked:
            self._record(action, columns, False, f"PII kolon talebi: {', '.join(blocked)}")
            raise GuardViolation(
                f"Su kolonlar kisisel veri iceriyor ve analize kapalidir: {', '.join(blocked)}. "
                "Bu kolonlar hicbir kosulda sorgulanamaz. Analizini acik kolonlarla kur."
            )

        self._record(action, columns, True, "Kolon izni verildi")

    # --- 2. k-anonimlik --------------------------------------------------

    def check_group_sizes(self, action: str, group_counts: dict[Any, int]) -> list[Any]:
        """k esiginin altinda kalan gruplari dondurur; bunlar sonuctan cikarilir."""
        suppressed = [key for key, n in group_counts.items() if n < K_ANONYMITY_THRESHOLD]
        if suppressed:
            self._record(
                action,
                (),
                True,
                f"{len(suppressed)} grup k<{K_ANONYMITY_THRESHOLD} nedeniyle bastirildi",
            )
        return suppressed

    def check_row_count(self, action: str, n_rows: int) -> None:
        """Filtre sonucu cok daralmissa tekil kisiye inilebilir; bunu engelle."""
        if 0 < n_rows < K_ANONYMITY_THRESHOLD:
            self._record(action, (), False, f"Sonuc kumesi cok dar: {n_rows} satir")
            raise GuardViolation(
                f"Bu filtre yalnizca {n_rows} satir donduruyor. En az {K_ANONYMITY_THRESHOLD} "
                "satir gerekiyor, aksi halde sonuc tek bir kisiye indirgenebilir. "
                "Filtreyi genislet."
            )

    # --- 3. PII maskeleme ------------------------------------------------

    _PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("TCKN", re.compile(r"\b[1-9][0-9]{10}\b")),
        ("TELEFON", re.compile(r"\b(?:\+90|0)?5[0-9]{9}\b")),
        ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    )

    def mask(self, text: str) -> str:
        """Modelin urettigi metinde PII deseni kalmissa maskeler.

        Kolon izni katmani zaten bu veriyi agent'a hic vermiyor; bu, o katman
        atlatilirsa devreye giren ikinci savunma hatti.
        """
        masked = text
        for label, pattern in self._PII_PATTERNS:
            masked, n = pattern.subn(f"[{label}_MASKELENDI]", masked)
            if n:
                self._record("mask_output", (), True, f"{n} adet {label} maskelendi")
        return masked

    # --- 4. Denetim kaydi ------------------------------------------------

    def _record(self, action: str, columns: tuple[str, ...] | list[str], allowed: bool, reason: str) -> None:
        self.audit_log.append(
            AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                action=action,
                columns=tuple(columns),
                allowed=allowed,
                reason=reason,
            )
        )

    def audit_trail(self) -> list[dict[str, Any]]:
        """Denetim kaydini API cevabina eklenebilir bicimde dondurur."""
        return [
            {
                "timestamp": e.timestamp,
                "action": e.action,
                "columns": list(e.columns),
                "allowed": e.allowed,
                "reason": e.reason,
            }
            for e in self.audit_log
        ]

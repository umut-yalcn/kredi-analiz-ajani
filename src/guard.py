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

    #: Kolon basina daha once dondurulen sonuc kumesi boyutlari.
    #: Fark alma savunmasi bunu kullanir; bkz. check_overlap.
    _gecmis: dict[str, list[int]] = field(default_factory=dict, repr=False)

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

    # --- 2b. Fark alma (differencing) savunmasi ------------------------------

    def check_overlap(self, action: str, key: str, n_rows: int) -> None:
        """Ayni kolon uzerindeki ardisik sorgularin BIRBIRINE cok yakin olmasini engeller.

        Tek bir sorgunun k satir dondurmesi yetmiyor. Iki AYRI sorgu da k esigini
        gecebilir ama aralarindaki fark tek bir kisi olabilir:

            kredi_skoru < 1893  -> 4996 satir, ortalama X
            kredi_skoru <= 1893 -> 4997 satir, ortalama Y

        ortalama * satir_sayisi grup toplamini verdigi icin (4997*Y - 4996*X) o
        tek kisinin gelirini KESIN olarak verir. Gozlemlendi: cikarilan 31499.88,
        gercek 31500.0. Iki sorgu da guard'dan onay almisti.

        Bu yuzden ayni kolon uzerinde daha once dondurulen sonuc kumesi
        boyutlarini hatirliyoruz; yeni sorgu oncekilerden k'dan az farkliysa
        reddediliyor. Tam bir sorgu denetimi degil - ayni surec icinde
        calisiyor - ama saldiriyi tek istekte kurmayi imkansiz kiliyor ve her
        reddi denetim kaydina yaziyor.
        """
        oncekiler = self._gecmis.setdefault(key, [])
        for onceki in oncekiler:
            fark = abs(n_rows - onceki)
            if 0 < fark < K_ANONYMITY_THRESHOLD:
                self._record(
                    action,
                    (key,),
                    False,
                    f"Fark alma riski: onceki sorgu {onceki} satir, bu sorgu {n_rows} "
                    f"satir; fark {fark} < {K_ANONYMITY_THRESHOLD}",
                )
                raise GuardViolation(
                    f"Bu sorgu, ayni kolon uzerinde daha once calistirdigin bir sorgudan "
                    f"yalnizca {fark} satir farkli ({onceki} -> {n_rows}). Iki sonucun "
                    f"farki {K_ANONYMITY_THRESHOLD} kisiden az bir gruba isaret ediyor ve "
                    "bu gruptaki kisilerin degerleri cikarilabilir. Esigi belirgin sekilde "
                    "degistir ya da farkli bir analiz kur."
                )
        oncekiler.append(n_rows)

    # --- 3. PII maskeleme ------------------------------------------------

    _PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("TCKN", re.compile(r"\b[1-9][0-9]{10}\b")),
        # Bastaki \b yerine rakam-olmayan lookbehind kullaniliyor. \b, dizgenin
        # basinda "+" onunde sinir bulamadigi icin "+905551234567" HIC
        # eslesmiyordu - uluslararasi formattaki numara maskelemeden siziyordu.
        ("TELEFON", re.compile(r"(?<![0-9])(?:\+90|0)?5[0-9]{9}(?![0-9])")),
        ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    )

    # Bir sayiyi toplulastirma sonucu yapan birimler. TCKN veya telefon
    # numarasi bu kelimelerle nitelenmez; toplam tutar, adet ve sure nitelenir.
    _OLCU_BIRIMI = re.compile(
        r"^\s*(?:tl|try|₺|lira|adet|kayit|kayıt|basvuru|başvuru|kisi|kişi|musteri|"
        r"müsteri|müşteri|satir|satır|gun|gün|ay|yil|yıl|puan|krd)\b",
        re.IGNORECASE,
    )

    # Sayiyi bir olcume baglayan onek. "toplam 12345678901" bir kimlik numarasi
    # degil, bir toplamdir.
    _OLCU_ONEKI = re.compile(
        r"(?:toplam|ortalama|genel|medyan|std|sapma|adet|say[iı]s[iı]|"
        r"tutar[iı]?|hacim|bakiye|limit|portfoy|portföy)\s*[:=]?\s*$",
        re.IGNORECASE,
    )

    # Sayiyi acikca KIMLIK olarak niteleyen onek. Boyle bir etiket varsa
    # arkadaki birim kelimesi ne olursa olsun maskeleme yapilir.
    # Onceden yalnizca eslesmenin ARKASINA bakiliyordu; "TCKN: 12345678901 TL"
    # ya da "Musteri TCKN 12345678901 kisi" maskelenmeden geciyordu.
    _KIMLIK_ONEKI = re.compile(
        r"(?:tckn|t\.?c\.?\s*kimlik|kimlik\s*(?:no|numaras[iı])?|vatandaslik|"
        r"tel(?:efon)?|gsm|cep|numaras[iı])\s*[:=#]?\s*$",
        re.IGNORECASE,
    )

    def _olcum_mu(self, text: str, bas: int, son: int) -> bool:
        """Eslesen sayi, bir toplulastirma sonucu gibi mi duruyor?

        TCKN ve telefon desenleri buyuk tam sayilarla kacinilmaz olarak cakisir:
        11 haneli her sayi TCKN'ye, 5 ile baslayan 10 haneli her sayi telefona
        benzer. Bir kredi burosunda toplam portfoy buyuklugu rahatlikla bu
        aralia girer. Deseni gevsetmek yerine - ki bu gercek PII'yi kacirmak
        demek olurdu - eslesmenin cevresine bakiyoruz.
        """
        onceki = text[max(0, bas - 32) : bas]

        # Acik kimlik etiketi her seyi gecersiz kilar: "TCKN: ... TL" bir
        # toplam degil, etiketlenmis bir kimlik numarasidir.
        if self._KIMLIK_ONEKI.search(onceki):
            return False

        return bool(
            self._OLCU_BIRIMI.match(text[son : son + 24])
            or self._OLCU_ONEKI.search(onceki[-24:])
        )

    def mask(self, text: str) -> str:
        """Modelin urettigi metinde PII deseni kalmissa maskeler.

        Kolon izni katmani zaten bu veriyi agent'a hic vermiyor; bu, o katman
        atlatilirsa devreye giren ikinci savunma hatti.
        """
        masked = text
        for label, pattern in self._PII_PATTERNS:
            atlanan = 0

            def _degistir(m: re.Match[str], _label: str = label) -> str:
                nonlocal atlanan
                if _label != "EMAIL" and self._olcum_mu(m.string, m.start(), m.end()):
                    atlanan += 1
                    return m.group(0)
                return f"[{_label}_MASKELENDI]"

            masked, n = pattern.subn(_degistir, masked)
            maskelenen = n - atlanan
            if maskelenen:
                self._record("mask_output", (), True, f"{maskelenen} adet {label} maskelendi")
            if atlanan:
                self._record(
                    "mask_output",
                    (),
                    True,
                    f"{atlanan} adet {label} benzeri sayi olcum baglaminda oldugu icin korundu",
                )
        return masked

    # --- 4. Denetim kaydi ------------------------------------------------

    def note(self, action: str, columns: tuple[str, ...] | list[str], reason: str) -> None:
        """Reddetme olmayan bir guard kararini kayda gecirir.

        Ornegin bir uc degerin k esigi nedeniyle bastirilmasi: istek reddedilmiyor
        ama sonuctan bir sey cikariliyor. Bunun izlenebilir olmasi gerekir.
        """
        self._record(action, columns, True, reason)

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

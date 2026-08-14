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
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schema import ANALYZABLE_COLUMNS, BY_NAME, PII_COLUMNS

# Bir toplulastirmanin dondurulebilmesi icin gereken en az satir sayisi.
# Bunun altinda kalan gruplar tek bir kisiyi isaret edebilecegi icin bastirilir.
K_ANONYMITY_THRESHOLD = 20


class GuardViolation(Exception):
    """Guard bir istegi reddettiginde firlatilir. Agent bunu hata olarak gorur ve plan degistirir."""


#: Fark alma savunmasi icin SUREC GENELINDE paylasilan sorgu gecmisi.
#:
#: Guard nesnesi istek basina yasiyor; gecmis yalnizca orada tutulsaydi saldirgan
#: iki sorguyu iki AYRI istege bolerek savunmayi tamamen atlardi (olculdu: ayni
#: istekte engellenen saldiri, ayri isteklerde 31499.88 degerini yine cikardi).
#: Bu yuzden gecmis modul duzeyinde ve kilitli tutuluyor.
#:
#: SINIRI: tek surec icinde gecerlidir. Birden fazla uvicorn worker'i ya da
#: yeniden baslatma bu korumayi sifirlar. Gercek bir kurulumda sorgu denetimi
#: kalici bir depoda ve kullanici/oturum bazinda tutulmalidir.
_SORGU_GECMISI: dict[str, set[int]] = {}
_GECMIS_KILIDI = threading.Lock()

#: Kolon basina saklanan en fazla FARKLI sorgu boyutu. Sinirsiz buyumeyi
#: engeller. Kapasite dolunca yeni kayit eklenmez ama eskiler SILINMEZ -
#: eskiyi silmek savunmayi taskinla asilabilir kilardi.
GECMIS_SINIRI = 200


def gecmisi_temizle() -> None:
    """Surec genelindeki sorgu gecmisini bosaltir. Testler icin."""
    with _GECMIS_KILIDI:
        _SORGU_GECMISI.clear()


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
        self._record(
            action,
            (),
            True,
            f"{len(suppressed)} grup k<{K_ANONYMITY_THRESHOLD} nedeniyle bastirildi"
            if suppressed
            else f"{len(group_counts)} grubun hepsi k esigini gecti",
        )
        return suppressed

    def check_row_count(self, action: str, n_rows: int) -> None:
        """Filtre sonucu cok daralmissa tekil kisiye inilebilir; bunu engelle.

        Basarili kontrol de kayda gecer: denetim kaydi "her guard karari"
        iddiasini tasiyorsa, gecen kontroller de gorunmeli. Onceden yalnizca
        RETLER yaziliyordu; bir denetci "bu sorgu k kontrolunden gecti mi"
        sorusunu kayittan yanitlayamiyordu.
        """
        if 0 < n_rows < K_ANONYMITY_THRESHOLD:
            self._record(action, (), False, f"Sonuc kumesi cok dar: {n_rows} satir")
            raise GuardViolation(
                f"Bu filtre yalnizca {n_rows} satir donduruyor. En az {K_ANONYMITY_THRESHOLD} "
                "satir gerekiyor, aksi halde sonuc tek bir kisiye indirgenebilir. "
                "Filtreyi genislet."
            )
        self._record(action, (), True, f"k kontrolu gecildi: {n_rows} satir")

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
        reddediliyor.

        Gecmis SUREC GENELINDE tutuluyor, Guard ornegi basina degil: aksi halde
        saldirgan iki sorguyu iki ayri istege bolerek savunmayi atlardi (bu da
        olculdu - ayni degeri yine cikardi).
        """
        with _GECMIS_KILIDI:
            oncekiler = _SORGU_GECMISI.setdefault(key, set())
            catisma = any(
                0 < abs(n_rows - o) < K_ANONYMITY_THRESHOLD for o in oncekiler
            )
            if not catisma:
                # Kapasite dolduysa YENI kayit eklenmez ama eskiler DURUR.
                # Eskiyi silmek fail-open olurdu: saldirgan zararsiz bir sorguyu
                # tekrarlayip koruyucu kaydi gecmisten dusurebilirdi. Kume
                # kullanmak ayrica ayni sorgunun tekrarini bedelsiz kiliyor -
                # taskin artik gecmisi buyutmuyor.
                if len(oncekiler) < GECMIS_SINIRI:
                    oncekiler.add(n_rows)

        if not catisma:
            self._record(action, (key,), True, f"Fark alma kontrolu gecildi: {n_rows} satir")
            return

        # Ret gerekcesi ONCEKI sorgunun satir sayisini ACIKLAMAZ. Aciklasaydi,
        # baska bir kullanicinin sorgu boyutu bu mesaj uzerinden ogrenilebilirdi
        # - savunmanin kendisi bir yan kanal olurdu.
        self._record(
            action,
            (key,),
            False,
            f"Fark alma riski: '{key}' uzerinde daha onceki bir sorguyla arasindaki "
            f"fark {K_ANONYMITY_THRESHOLD} satirdan az",
        )
        raise GuardViolation(
            f"Bu sorgu, ayni kolon uzerinde daha once calistirilan bir sorguya cok "
            f"yakin bir sonuc kumesi donduruyor. Iki sonucun farki "
            f"{K_ANONYMITY_THRESHOLD} kisiden az bir gruba isaret ediyor ve o "
            "gruptaki kisilerin degerleri cikarilabilir. Esigi belirgin sekilde "
            "degistir ya da farkli bir analiz kur."
        )

    # --- 3. PII maskeleme ------------------------------------------------

    _PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("TCKN", re.compile(r"\b[1-9][0-9]{10}\b")),
        # Bastaki \b yerine rakam-olmayan lookbehind kullaniliyor. \b, dizgenin
        # basinda "+" onunde sinir bulamadigi icin "+905551234567" HIC
        # eslesmiyordu - uluslararasi formattaki numara maskelemeden siziyordu.
        ("TELEFON", re.compile(r"(?<![0-9])(?:\+90|0)?5[0-9]{9}(?![0-9])")),
        ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    )

    # BAGLAM SEZGISELI KALDIRILDI.
    #
    # Onceden, eslesmenin cevresinde "TL/adet/kisi" gibi bir birim ya da
    # "toplam/ortalama" gibi bir onek varsa maskeleme atlaniyordu; amac mesru
    # buyuk sayilarin maskelenmesini onlemekti. Bagimsiz denetimde bunun
    # atlatilabildigi gosterildi:
    #
    #     "TCKN degeri: 12345678901 TL"      -> maskelenmiyordu
    #     "Telefon degeri: 05551234567 adet" -> maskelenmiyordu
    #
    # Etiket listesini genisletmek cozum degil: sezgisel oldugu surece bir
    # sonraki ifade yine disarida kalir. Modelin ya da bir prompt enjeksiyonunun
    # sayinin sonuna bir birim eklemesi yetiyor.
    #
    # Bu KATMAN son savunma hatti. Burada yanlis pozitif (mesru bir toplamin
    # maskelenmesi) gorunur ve zararsizdir; yanlis negatif (bir TCKN'nin
    # sizmasi) gorunmez ve yikicidir. O yuzden desen eslesen her sayi
    # KOSULSUZ maskeleniyor.
    #
    # Bedeli: 11 haneli ya da 5 ile baslayan 10 haneli mesru bir toplam da
    # maskelenir. Bu veri setinde toplamlar o buyukluge ulasmiyor; ulasan bir
    # kurulumda toplamlar arac ciktisindaki YAPILANDIRILMIS alanlardan
    # okunmali, serbest metinden degil.

    def mask(self, text: str) -> str:
        """Modelin urettigi metinde PII deseni kalmissa maskeler.

        Kolon izni katmani zaten bu veriyi agent'a hic vermiyor; bu, o katman
        atlatilirsa devreye giren son savunma hatti. Kosulsuz maskeler -
        gerekcesi icin yukaridaki nota bak.
        """
        masked = text
        for label, pattern in self._PII_PATTERNS:
            masked, n = pattern.subn(f"[{label}_MASKELENDI]", masked)
            if n:
                self._record("mask_output", (), True, f"{n} adet {label} maskelendi")
        return masked

    # --- 4. Denetim kaydi ------------------------------------------------

    def reddet(self, action: str, columns: tuple[str, ...], reason: str) -> None:
        """Arac duzeyindeki bir REDDI kayda gecirir.

        Guard'in kendi kontrolleri disinda kalan dogrulamalar (kolon tipi vb.)
        de denetim izinde gorunmeli; onceden bu retler kayda hic girmiyordu ve
        denetim izi yalnizca "izin verildi" satirlarindan olusuyordu.
        """
        self._record(action, columns, False, reason)

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

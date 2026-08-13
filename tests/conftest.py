"""Testler icin veri hazirligi.

Taze bir klonda `data/` bostur - veri seti depoya konulmaz, uretilir. README
"pytest tests/ -q" komutunun API anahtari olmadan calistigini soyluyor; bunun
TAZE KLONDA da dogru olmasi gerekiyor.

Onceden oyle degildi: klonlayan biri once pytest calistirdiginda 17 test ham
bir FileNotFoundError ile dusuyordu. Veri uretimi artik burada, toplama
asamasindan once yapiliyor.

Uretec tohumlu; var olan veriye dokunulmaz, yalnizca eksikse uretilir.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
VERI = KOK / "data" / "kredi_basvurulari.csv"
URETEC = KOK / "scripts" / "generate_data.py"
SATIR = 5000


def pytest_configure(config) -> None:
    """Veri yoksa uretir. Testler toplanmadan once calisir."""
    if VERI.exists():
        return

    print(f"\n[conftest] Veri bulunamadi, uretiliyor: {VERI.name} ({SATIR} satir)")
    sonuc = subprocess.run(
        [sys.executable, str(URETEC), "--rows", str(SATIR)],
        cwd=KOK,
        capture_output=True,
        text=True,
    )
    if sonuc.returncode != 0:
        raise RuntimeError(
            f"Test verisi uretilemedi.\n"
            f"Komut: python scripts/generate_data.py --rows {SATIR}\n"
            f"{sonuc.stderr}"
        )
    print(f"[conftest] Hazir: {VERI}")

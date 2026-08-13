"""Komut satirindan hizli soru sormak icin.

    python cli.py "Hangi meslek grubunda temerrut orani en yuksek?"
    python cli.py            # etkilesimli mod
"""

from __future__ import annotations

import sys

from src.agent import ask


def show(question: str) -> None:
    result = ask(question)

    print("\n" + "=" * 70)
    print(result["cevap"])
    print("=" * 70)

    ozet = result.get("arac_ozeti", {})
    if result["kullanilan_araclar"]:
        print(f"\nKullanilan araclar ({ozet.get('basarili', 0)} basarili, "
              f"{ozet.get('hatali', 0)} hatali):")
        for i, call in enumerate(result["kullanilan_araclar"], 1):
            print(f"  {i}. {call['arac']}({call['girdi']})")

    if result.get("duzeltme_denemesi"):
        print(f"\nDuzeltmeye geri gonderme: {result['duzeltme_denemesi']} kez "
              "(dayanaksiz cevap yazmaya kalkisti)")

    reddedilen = [e for e in result["denetim_kaydi"] if not e["allowed"]]
    if reddedilen:
        print("\nGuard tarafindan reddedilen istekler:")
        for e in reddedilen:
            print(f"  - {e['action']}: {e['reason']}")
    print()


def main() -> None:
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return

    print("Soru yaz, cikmak icin bos birak.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        show(question)


if __name__ == "__main__":
    main()

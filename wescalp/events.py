"""Calendário de notícias marcadas dos EUA: comunicados do Fed (FOMC) e divulgação da inflação (CPI).

    python -m wescalp.events        # baixa ou atualiza data/factors/events.csv

- FOMC: só as reuniões que estavam na agenda (extraordinárias, canceladas, votos por escrito e teleconferências
  ficam de fora, porque não eram conhecidas antes). Vale o último dia da reunião, com o comunicado às 14h de Nova
  York (o horário de hoje; algumas reuniões antigas saíram em outro horário, o que só muda algo com o dia virando
  às 18h UTC). Fonte: as páginas do Fed, fomchistoricalAAAA.htm até 2020 e fomccalendars.htm de 2021 em diante.
- CPI: as datas de divulgação do índice de preços ao consumidor, às 8h30 de Nova York. Cada safra (vintage) nova do
  CPIAUCNS (sem ajuste sazonal) no ALFRED, o arquivo histórico do FRED, é uma divulgação; o site do BLS recusa
  acesso automático.
"""

import argparse
import csv
import logging
import re
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np

from .factors import DATA_DIR

log = logging.getLogger("wescalp.events")

NEW_YORK = ZoneInfo("America/New_York")
FOMC_TIME, CPI_TIME = time(14, 0), time(8, 30)
FIRST_YEAR = 2011
LAST_HISTORICAL = 2020  # depois disso, os anos estão na página de calendário
HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{}.htm"
CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
CPI_URL = "https://alfred.stlouisfed.org/series/downloaddata?seid=CPIAUCNS"
MONTHS = {m: i + 1 for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                          "dec"))}
WHEN = re.compile(r"^(?P<m1>[A-Za-z]+)(?:/(?P<m2>[A-Za-z]+))?\s+(?P<d1>\d+)"
                  r"(?:-(?:(?P<m3>[A-Za-z]+)\s+)?(?P<d2>\d+))?$")


def meeting_end(when: str, year: int) -> date:
    """"January 27-28", "Apr/May 30-1", "July 31-August 1", "March 15" → o último dia da reunião."""
    m = WHEN.match(re.sub(r"\s+", " ", when.strip()))
    if not m:
        raise ValueError(f"data de reunião desconhecida: {when!r}")
    month = m["m3"] or m["m2"] or m["m1"]
    return date(year, MONTHS[month[:3].lower()], int(m["d2"] or m["d1"]))


def parse_historical(html: str) -> list[date]:
    """Página fomchistoricalAAAA.htm → últimos dias das reuniões marcadas."""
    out = []
    for head in re.findall(r"<h5[^>]*>([^<]*)</h5>", html):
        head = re.sub(r"\s+", " ", head).strip()
        m = re.match(r"^(.*) Meeting - (\d{4})$", head)
        if m and "(" not in head:
            out.append(meeting_end(m[1], int(m[2])))
    return out


def parse_calendar(html: str) -> list[date]:
    """Página fomccalendars.htm → últimos dias das reuniões marcadas de cada ano (o * marca as com projeções)."""
    out = []
    for block in re.split(r'<div class="panel panel-default">', html)[1:]:
        y = re.search(r"(\d{4}) FOMC Meetings", block)
        if not y:
            continue
        rows = re.findall(r'fomc-meeting__month[^>]*><strong>([^<]*)</strong></div>\s*'
                          r'<div class="fomc-meeting__date[^>]*>([^<]*)<', block)
        for month, days in rows:
            if "(" not in days:
                out.append(meeting_end(f"{month} {days.replace('*', '').strip()}", int(y[1])))
    return out


def parse_cpi(html: str) -> list[date]:
    """Página de download do ALFRED → datas das safras do CPI, uma por divulgação."""
    m = re.search(r'<select id="form_selected_vintage_dates"[^>]*>(.*?)</select>', html, re.S)
    if not m:
        raise ValueError("lista de safras não encontrada")
    return [date.fromisoformat(d) for d in re.findall(r'<option value="(\d{4}-\d\d-\d\d)"', m[1])]


def at(day: date, hour: time) -> int:
    """Instante UTC (segundos) da hora de Nova York naquele dia, com o horário de verão de lá."""
    return int(datetime.combine(day, hour, NEW_YORK).timestamp())


def path(data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "events.csv"


def fetch(data_dir: Path = DATA_DIR) -> dict[str, int]:
    fomc: set[date] = set()
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for y in range(FIRST_YEAR, LAST_HISTORICAL + 1):
            r = client.get(HISTORICAL_URL.format(y))
            r.raise_for_status()
            fomc.update(parse_historical(r.text))
        r = client.get(CALENDAR_URL)
        r.raise_for_status()
        fomc.update(d for d in parse_calendar(r.text) if d.year > LAST_HISTORICAL)
        r = client.get(CPI_URL)
        r.raise_for_status()
        cpi = {d for d in parse_cpi(r.text) if d.year >= FIRST_YEAR}
    rows = sorted([("fomc", d, at(d, FOMC_TIME)) for d in fomc] + [("cpi", d, at(d, CPI_TIME)) for d in cpi])
    p = path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows((k, d.isoformat(), t) for k, d, t in rows)
    return {"fomc": len(fomc), "cpi": len(cpi)}


def load(data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
    """{"fomc": instantes UTC, "cpi": instantes UTC}, em ordem."""
    out: dict[str, list[int]] = {"fomc": [], "cpi": []}
    with open(path(data_dir), newline="") as f:
        for kind, _, t in csv.reader(f):
            out[kind].append(int(t))
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in out.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_DIR)
    args = p.parse_args()
    counts = fetch(args.data)
    ev = load(args.data)
    for k, n in counts.items():
        first, last = (datetime.fromtimestamp(int(x), timezone.utc).date() for x in (ev[k][0], ev[k][-1]))
        log.info("%s: %d datas, de %s a %s", k, n, first, last)


if __name__ == "__main__":
    main()

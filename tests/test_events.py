from datetime import date, datetime, timezone

import pytest

pytest.importorskip("numpy")

from wescalp import events  # noqa: E402

# Títulos reais das páginas fomchistoricalAAAA.htm (com os espaços duplos que elas têm).
HEADS = [
    "January 28-29 Meeting - 2020", "March 2 (unscheduled) Meeting - 2020", "March 17-18 (cancelled) Meeting - 2020",
    "March 19 (notation vote) - 2020", "July 31-August 1  Meeting - 2012", "March  4 (unscheduled) - 2014",
    "August 1 Conference Call - 2011", "Jan/Feb 31-1 Meeting - 2017", "April/May 30-1 Meeting - 2013",
    "March 15 Meeting - 2011",
]
HISTORICAL = "".join(f'<h5 class="panel-heading panel-heading--shaded">{h}</h5>' for h in HEADS)


def row(month: str, days: str) -> str:
    return (f'<div class="row fomc-meeting" "> <div class="fomc-meeting__month col-xs-5 col-sm-3 col-md-2">'
            f'<strong>{month}</strong></div> <div class="fomc-meeting__date col-xs-4 col-sm-9 col-md-10 col-lg-1">'
            f'{days}</div> <div class="col-xs-12 col-md-4 col-lg-2"> <strong>Statement:</strong></div></div>')


CALENDAR = ('<div class="panel panel-default"><div class="panel-heading"><h4><a id="42827">2025 FOMC Meetings</a></h4>'
            "</div>" + row("January", "28-29") + row("March", "18-19*") + row("August", "22 (notation vote)")
            + '</div><div class="panel panel-default"><div class="panel-heading"><h4><a id="39116">2024 FOMC Meetings'
            "</a></h4></div>" + row("Apr/May", "30-1") + row("Oct/Nov", "31-1") + "</div>")
CPI = ('<select id="form_units" name="form[units]"><option value="lin">x</option></select>'
       '<select id="form_selected_vintage_dates" name="form[selected_vintage_dates][]" size="10" multiple="multiple">'
       '<option value="2013-10-30">2013-10-30</option><option value="2025-12-18">2025-12-18</option></select>')


def test_historical_keeps_only_scheduled_meetings():
    assert events.parse_historical(HISTORICAL) == [
        date(2020, 1, 29), date(2012, 8, 1), date(2017, 2, 1), date(2013, 5, 1), date(2011, 3, 15)]


def test_calendar_page_years_and_month_pairs():
    assert events.parse_calendar(CALENDAR) == [date(2025, 1, 29), date(2025, 3, 19), date(2024, 5, 1), date(2024, 11, 1)]


def test_cpi_vintages():
    assert events.parse_cpi(CPI) == [date(2013, 10, 30), date(2025, 12, 18)]
    with pytest.raises(ValueError):
        events.parse_cpi("<html></html>")


def test_new_york_time_with_daylight_saving():
    assert events.at(date(2024, 1, 11), events.CPI_TIME) == datetime(2024, 1, 11, 13, 30, tzinfo=timezone.utc).timestamp()
    assert events.at(date(2024, 7, 31), events.FOMC_TIME) == datetime(2024, 7, 31, 18, tzinfo=timezone.utc).timestamp()


def test_unknown_meeting_text_fails_loudly():
    with pytest.raises(ValueError):
        events.meeting_end("sometime in March", 2020)

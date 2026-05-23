"""
RSS feed scraper for financial news (Reuters + Google News).
Returns a DataFrame with columns: ['date', 'title', 'summary', 'text', 'link', 'source']
"""

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?"
    "q=Sensex+OR+Nifty+OR+%22stock+market%22&"
    "hl=en-IN&gl=IN&ceid=IN:en"
)

RSS_SOURCES = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/marketsNews",
    GOOGLE_NEWS_RSS,
]

MAX_ITEMS_PER_FEED = 50

_TAGS_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_POS_WORDS = {
    "gain", "surge", "upbeat", "profit", "beat", "strong", "growth", "rally",
    "bullish", "upgrade", "outperform", "record", "robust", "improve", "higher",
    "boost", "optimism", "advance", "climb", "jump",
}
_NEG_WORDS = {
    "loss", "slump", "downgrade", "weak", "miss", "decline", "risk", "bearish",
    "fall", "plunge", "cut", "shortfall", "slowdown", "concern", "lower", "drop",
    "crisis", "volatility", "selloff", "slide",
}


def _fetch(url: str, timeout: int = 12) -> bytes | None:
    try:
        req = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
                )
            },
        )
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (HTTPError, URLError, TimeoutError) as e:
        print(f"[RSS Scraper] Fetch failed for {url}: {e}")
        return None


def _strip_html(s: str) -> str:
    s = _TAGS_RE.sub(" ", s or "")
    return _WS_RE.sub(" ", s).strip()


def _parse_rss(xml_bytes: bytes) -> list[dict]:
    if not xml_bytes:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    items = []
    channel = root.find("channel")
    if channel is not None:
        for it in channel.findall("item"):
            items.append(
                {
                    "title": (it.findtext("title") or "").strip(),
                    "summary": (it.findtext("description") or "").strip(),
                    "link": (it.findtext("link") or "").strip(),
                    "published": (it.findtext("pubDate") or "").strip(),
                }
            )
    else:
        ns = "{http://www.w3.org/2005/Atom}"
        for it in root.findall(f".//{ns}entry"):
            link_el = it.find(f"{ns}link")
            items.append(
                {
                    "title": (it.findtext(f"{ns}title") or "").strip(),
                    "summary": (it.findtext(f"{ns}summary") or "").strip(),
                    "link": (link_el.get("href", "").strip() if link_el is not None else ""),
                    "published": (it.findtext(f"{ns}updated") or "").strip(),
                }
            )

    cleaned = []
    for d in items[: (MAX_ITEMS_PER_FEED or len(items))]:
        title = _strip_html(d.get("title", ""))
        summ = _strip_html(d.get("summary", ""))
        if not (title or summ):
            continue
        cleaned.append(
            {
                "title": title,
                "summary": summ,
                "link": d.get("link", ""),
                "published": d.get("published", ""),
            }
        )
    return cleaned


def _parse_date(raw: str) -> datetime:
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def lexicon_label(text: str) -> int:
    """Weak pseudo-label: 1=positive, 0=negative, based on finance lexicon."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    score = sum(w in _POS_WORDS for w in words) - sum(w in _NEG_WORDS for w in words)
    return 1 if score >= 0 else 0


def scrape_rss_news(sources: list[str] = RSS_SOURCES) -> pd.DataFrame:
    """
    Fetch and parse RSS feeds into a structured DataFrame.

    Returns:
        pd.DataFrame with columns:
            ['date', 'title', 'summary', 'text', 'link', 'source', 'pseudo_label']
    """
    rows = []
    for src in sources:
        xml_bytes = _fetch(src)
        for item in _parse_rss(xml_bytes or b""):
            text = (item["title"] + ". " + item["summary"]).strip()
            dt = _parse_date(item["published"])
            rows.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "title": item["title"],
                    "summary": item["summary"],
                    "text": text,
                    "link": item["link"],
                    "source": src,
                }
            )
        time.sleep(0.5)

    if not rows:
        print("[RSS Scraper] No articles found. Using fallback dataset.")
        rows = [
            {
                "date": "2025-01-01",
                "title": "Market rally on positive earnings",
                "summary": "Banks lead gains.",
                "text": "Market rally on positive earnings. Banks lead gains.",
                "link": "",
                "source": "fallback",
            },
            {
                "date": "2025-01-01",
                "title": "Sensex slips amid global risk-off",
                "summary": "IT stocks drag indices lower.",
                "text": "Sensex slips amid global risk-off. IT stocks drag indices lower.",
                "link": "",
                "source": "fallback",
            },
        ]

    df = pd.DataFrame(rows).drop_duplicates(subset=["title", "summary", "link"])
    df["pseudo_label"] = df["text"].apply(lexicon_label)
    return df.reset_index(drop=True)

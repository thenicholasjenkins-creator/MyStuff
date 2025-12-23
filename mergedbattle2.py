
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mergedbattle.py — Unified results + anti-freeze watchdog
- Single unified CSV (leadership_results.csv) with EMAIL + LEADERSHIP events and a human-readable `formatted` column.
- Dual watchdogs: internal per-domain elapsed-time check + external asyncio.wait_for hard timeout per domain.
- Clear visited_pages at the start of each domain crawl.
- Safe BLOCK_PATTERNS, guarded route matcher, timezone-aware UTC timestamps.
"""

import asyncio
import nest_asyncio
import pandas as pd
import json
import os
import re
import sys
import time
import html
import io
import random
from typing import List, Tuple, Callable, Dict
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone
from tqdm import tqdm
from playwright.async_api import async_playwright
import argparse

# Optional PDF/OCR libs
PDF2IMAGE_AVAILABLE = False
PYTESSERACT_AVAILABLE = False
PIL_AVAILABLE = False
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    pass
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    pass
try:
    from PIL import Image, ImageOps, ImageFilter
    PIL_AVAILABLE = True
except Exception:
    pass

nest_asyncio.apply()

# -------------------------
# Base directory resolution
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LOG_DIR = os.path.join(BASE_DIR, "logs")
DEFAULT_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

 
DEBUG = True
HOST_STRICT = False
SAFE_TEXT_CHARS = 400_000

DEFAULT_TIME_LIMIT_SEC = 300
DEFAULT_CONCURRENCY = 6
DEFAULT_DEPTH_LIMIT = 3
DEFAULT_MAX_LINKS_PER_PAGE = 12
DEFAULT_MAX_FANOUT_PER_PAGE = 40

LOG_DIR = os.path.join(BASE_DIR, "logs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

UNIFIED_FILE = os.path.join(BASE_DIR, "leadership_results.csv")
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.jsonl")
LOG_FILE = os.path.join(LOG_DIR, "crawler.log")


found_emails: set = set()
email_records: List[Dict] = []
visited_pages: set = set()
leadership_records: List[Dict] = []
_leadership_keys: set = set()  # <--- add this
DOMAIN_START_TS: Dict[str, float] = {}


PDF_ENABLED = False
PDF_MAX_PER_DOMAIN = 2
PDF_MAX_SIZE_MB = 4.0
PDF_MAX_PAGES = 3
PDF_TIMEOUT_SEC = 25
PDF_COUNTS_BY_DOMAIN: Dict[str, int] = {}
PDF_OCR_ENABLED = True
OCR_DPI = 200
OCR_LANG = "eng"
OCR_OEM = 1
OCR_PSM = 6
OCR_MAX_PIXELS = 12_000_000
OCR_BW = True
STEALTH = True

# Logging
import logging
from logging.handlers import RotatingFileHandler

human_logger = logging.getLogger("crawler")
json_logger = logging.getLogger("progress")


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)

    human_logger.setLevel(logging.INFO)
    h_file = RotatingFileHandler(
        os.path.join(log_dir, "crawler.log"), maxBytes=5_000_000, backupCount=3
    )
    h_file.setLevel(logging.INFO)
    h_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    h_file.setFormatter(h_fmt)
    human_logger.addHandler(h_file)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(h_fmt)
    human_logger.addHandler(console)

    json_logger.setLevel(logging.INFO)
    p_file = logging.FileHandler(os.path.join(log_dir, "progress.jsonl"))
    p_file.setLevel(logging.INFO)
    p_fmt = logging.Formatter("%(message)s")
    p_file.setFormatter(p_fmt)
    json_logger.addHandler(p_file)


def ts_utc() -> str:
    """Return UTC ISO-8601 with Z suffix, timezone-aware."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_json(event: str, **kv):
    payload = {"ts": ts_utc(), "event": event}
    payload.update(kv)
    try:
        json_logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


# Crawl config — SAFE, VALID REGEXES
BLOCK_PATTERNS = [
    r".*\.(?:png|jpe?g|gif|webp|svg|ico)(?:\?.*)?$",
    r".*\.(?:woff2?|ttf|otf)(?:\?.*)?$",
    r".*/(?:gtm|googletagmanager)\.com/.*",
    r".*/google-analytics\.com/.*",
    r".*doubleclick\.net/.*",
    r".*/facebook\.com/tr/.*",
    r".*/snapchat\.com/.*",
    r".*/hotjar\.com/.*",
]
TEXT_CONTENT_TYPES = (
    "text/html",
    "application/json",
    "text/plain",
    "application/xhtml+xml",
    "application/xml",
)
COMMON_TLD_FIXES = {
    ".or": ".org",
    ".coo": ".coop",
    ".coom": ".com",
    ".con": ".com",
    ".og": ".org",
    ".cm": ".com",
}
BAD_HOSTS = {"facebook.com", "www.facebook.com", "wixsite.com", "linktr.ee"}


class DomainConfig:
    def __init__(
        self,
        base: str,
        allowed_paths: List[str],
        valid_email_domains: set,
        seeds: List[str],
        name: str,
    ):
        self.base = base
        self.allowed_paths = allowed_paths or ["/"]
        self.valid_email_domains = valid_email_domains or set()
        self.seeds = seeds or [base]
        self.name = name


def host_root(u: str) -> str:
    try:
        netloc = urlparse(u).netloc.lower()
        return netloc if HOST_STRICT else netloc.lstrip("www.")
    except Exception:
        return ""


def fix_tld(host: str) -> str:
    h = (host or "").strip().lower()
    for bad, good in COMMON_TLD_FIXES.items():
        if h.endswith(bad):
            return h[: -len(bad)] + good
    return h


def normalize_url(raw: str) -> str:
    u = (raw or "").strip().strip('"').strip("'")
    if not u:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-\.]*://", u):
        u = "https://" + u.lstrip("/")
    p = urlparse(u)
    host = fix_tld(p.netloc)
    if any(b in host for b in BAD_HOSTS):
        return ""
    return p._replace(netloc=host).geturl()


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    no_frag = parsed._replace(fragment="")
    s = no_frag.geturl()
    return s[:-1] if s.endswith("/") else s


def default_config_for(url: str) -> DomainConfig:
    host = urlparse(url).netloc.lower()
    label = host.split(".")[0].upper() if host else host
    return DomainConfig(
        base=f"{urlparse(url).scheme}://{host}/",
        allowed_paths=["/"],
        valid_email_domains=set(),
        seeds=[url],
        name=label,
    )


def jovia_config() -> DomainConfig:
    return DomainConfig(
        base="https://www.jovia.org/",
        allowed_paths=["/support", "/knowledge-base", "/learn", "/about", "/locations", "/"],
        valid_email_domains={"jovia.org", "joviafinancial.com"},
        seeds=["https://www.jovia.org/", "https://www.jovia.org/support"],
        name="JOVIA",
    )


def msfcu_config() -> DomainConfig:
    return DomainConfig(
        base="http://morrissheppardfcu.org/",
        allowed_paths=["/", "/contact/"],
        valid_email_domains={"mstfcu.org", "morrissheppardfcu.org"},
        seeds=["http://morrissheppardfcu.org/", "http://morrissheppardfcu.org/contact/"],
        name="MORRISSHEPPARDFCU",
    )


def select_config_for(url: str) -> DomainConfig:
    host = urlparse(url).netloc.lower()
    if "jovia.org" in host:
        return jovia_config()
    if "morrissheppardfcu.org" in host:
        return msfcu_config()
    return default_config_for(url)


PLACEHOLDER_EMAILS = {"user@example.com", "[email\u00A0protected]"}
GENERIC_LOCALPARTS = {"info", "service", "support", "customerservice", "hello", "contact"}
LEADERSHIP_TAG_HINTS = ("leadership", "json-ld", "pdf-window")

EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)

# First Last | First M. Last | First Middle Last | allow hyphenated last names
NAME_REGEX = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){0,2}(?:-[A-Z][a-z]+)?)\b"
)

ROLE_REGEX = re.compile(
    r"\b("
    r"Chief Executive Officer|CEO|COO|CFO|CMO|CTO|"
    r"Vice President|VP|Assistant Vice President|AVP|"
    r"Manager|Director|Supervisor|Officer|Loan Officer|Branch Manager|Member Services|"
    r"Operations|Lending|Marketing|HR|Compliance|Chief Lending Officer|CLO|SVP|EVP|"
    r"Chair|Board Chair|Treasurer|Secretary"
    r")\b",
    flags=re.IGNORECASE,
)


def decode_cfemail(cfhex: str) -> str:
    if not cfhex or len(cfhex) < 4:
        return ""
    try:
        r = int(cfhex[:2], 16)
        return "".join(chr(int(cfhex[i : i + 2], 16) ^ r) for i in range(2, len(cfhex), 2))
    except Exception:
        return ""


def deobfuscate_email(s: str) -> str:
    if not s:
        return ""
    t = html.unescape(s)
    t = re.sub(r"\s*\[at\]\s*|\s*\(at\)\s*", "@", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\[dot\]\s*|\s*\(dot\)\s*", ".", t, flags=re.IGNORECASE)
    return t


def email_domain(email: str) -> str:
    try:
        return email.split("@", 1)[1].lower()
    except Exception:
        return ""


def is_credit_union_domain(domain: str) -> bool:
    d = (domain or "").lower()
    return (
        d.endswith(".coop")
        or d.endswith(".org")
        or d.endswith(".cu")
        or "creditunion" in d
        or "fcu" in d
        or d.endswith("cu.com")
        or "cu." in d
    )


def keep_email(email: str, source_host: str, valid_domains: set) -> bool:
    if not email or email in PLACEHOLDER_EMAILS:
        return False
    d = email_domain(email)
    d_norm = d.lstrip("www.")
    s_norm = (source_host or "").lstrip("www.").lower()
    if d_norm == s_norm:
        return True
    if valid_domains and d in valid_domains:
        return True
    if is_credit_union_domain(d):
        return True
    return False


def add_email(email_or_href: str, source_url: str, tag: str, cfg: dict, keep_generic=False) -> None:
    if not email_or_href:
        return
    email = (email_or_href or "").strip()
    if email.startswith("mailto:"):
        email = email.split(":", 1)[1].split("?", 1)[0]
    email = deobfuscate_email(email)

    if EMAIL_REGEX.fullmatch(email):
        src_host = host_root(cfg["base"])
        local = email.split("@", 1)[0].lower()
        if (local in GENERIC_LOCALPARTS) and (not keep_generic) and (
            not any(h in tag for h in LEADERSHIP_TAG_HINTS)
        ):
            return
        if keep_email(email, src_host, cfg.get("valid_email_domains", set())):
            email_lc = email.lower()
            if email_lc not in found_emails:
                found_emails.add(email_lc)
                rec = {
                    "credit_union": cfg.get("credit_union", ""),
                    "email": email_lc,
                    "domain": email_domain(email_lc),
                    "source_url": source_url,
                    "found_via": tag,
                    "timestamp": ts_utc(),
                }
                email_records.append(rec)
                human_logger.info(f"[FOUND {tag}] {email_lc} (from {source_url})")
                log_json("email_found", **rec)


def confidence_for(tag: str, role: str = "") -> float:
    tag = (tag or "").lower()
    if "leadership-text" in tag:
        base = 0.90
    elif "leadership-mailto" in tag or "leadership-structured" in tag or "leadership-mailto-fallback" in tag:
        base = 0.85
    elif "pdf-window" in tag:
        base = 0.80
    elif "json-ld" in tag:
        base = 0.70
    elif ("support-mailto" in tag) or ("click-here" in tag) or ("cfemail" in tag):
        base = 0.70
    elif ("network" in tag) or ("html" in tag):
        base = 0.60
    else:
        base = 0.60

    try:
        if role and ROLE_REGEX.search(role or ""):
            base = max(base, base + 0.05)
    except Exception:
        pass
    return round(base, 2)


def add_leadership(role: str, name: str, email: str, source_url: str, tag: str, cfg: dict) -> None:
    role = (role or "").strip()
    name = (name or "").strip()
    email = (email or "").strip().lower()

    if not (role or name or email):
        return

    key = (role.lower(), name.lower(), email.lower(), source_url)
    if key in _leadership_keys:
        return
    _leadership_keys.add(key)

    conf = confidence_for(tag, role)
    rec = {
        "credit_union": cfg.get("credit_union", ""),
        "role": role,
        "name": name,
        "email": email,
        "source_url": source_url,
        "found_via": tag,
        "confidence": conf,
        "timestamp": ts_utc(),
    }
    leadership_records.append(rec)
    log_json("leadership_found", **rec)

    # Only add email if present (and let add_email handle validity)
    if email:
        add_email(email, source_url, f"{tag}-email", cfg, keep_generic=True)


async def progressive_reveal(page):
    try:
        details = page.locator("details")
        for i in range(await details.count()):
            try:
                await page.evaluate("(el)=>el.open=true", details.nth(i))
            except Exception:
                pass

        locs = [
            page.get_by_role(
                "button",
                name=re.compile(
                    r"(show\nexpand\nstaff\nteam\nleadership\ncontact\ndirectory\nview\ndetails)",
                    re.I,
                ),
            ),
            page.get_by_role("tab"),
        ]
        for loc in locs:
            cnt = await loc.count()
            for i in range(min(cnt, 20)):
                try:
                    el = loc.nth(i)
                    if await el.is_visible():
                        await el.scroll_into_view_if_needed()
                        await asyncio.sleep(0.2)
                        await el.click()
                        await asyncio.sleep(0.25)
                except Exception:
                    continue

        total = await page.evaluate(
 	   "Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"
	)

        step = max(300, int(total / 8))
        for y in range(0, total, step):
            try:
                await page.evaluate("window.scrollTo(0, arguments[0])", y)
                await asyncio.sleep(0.2)
            except Exception:
                break
    except Exception:
        pass


PEOPLE_ALLOW = [
    "about",
    "leadership",
    "board",
    "management",
    "executive",
    "team",
    "staff",
    "directory",
    "contact",
    "careers",
    "employment",
    "who-we-are",
    "locations",
    "branch",
    "community",
]
PEOPLE_DENY = [
    "checking",
    "savings",
    "loans",
    "mortgage",
    "auto",
    "credit-card",
    "rates",
    "calculator",
    "re-order",
    "personal-loans",
    "student-loans",
    "apply",
    "specials",
    "promo",
    "blog",
    "faq",
]
PEOPLE_DENY_REGEXES = [r"/accounts/.*", r"/loans/.*", r"/loan-.*", r".*/calculator.*", r".*/rates.*"]
DENY_RE_COMPILED = [re.compile(p, re.I) for p in PEOPLE_DENY_REGEXES]


def url_is_denied(url: str) -> bool:
    path = urlparse(url).path.lower()
    if any(rx.search(path) for rx in DENY_RE_COMPILED):
        return True
    u = url.lower()
    return any(k in u for k in PEOPLE_DENY)


def url_is_people_relevant(url: str) -> bool:
    u = url.lower()
    if any(k in u for k in PEOPLE_ALLOW):
        return True
    if any(k in u for k in PEOPLE_DENY):
        return False
    return True


async def parse_json_ld_for_people(page, url: str, cfg: dict):
    try:
        scripts = page.locator("script[type='application/ld+json']")
        n = await scripts.count()
        for i in range(n):
            raw = (await scripts.nth(i).inner_text() or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            def handle_person(obj):
                if not isinstance(obj, dict):
                    return
                email = obj.get("email") or obj.get("emailAddress") or ""
                job = obj.get("jobTitle") or obj.get("role") or ""
                name = obj.get("name") or ""
                if email or name or job:
                    add_leadership(job or "", name or "", email, url, "json-ld", cfg)

            if isinstance(data, dict):
                if (data.get("@type") == "Person") or (
                    "email" in data and ("jobTitle" in data or "role" in data)
                ):
                    handle_person(data)
                for key in ("employee", "employees", "member", "members", "staff", "person", "people"):
                    val = data.get(key)
                    if isinstance(val, list):
                        for obj in val:
                            handle_person(obj)
                    else:
                        handle_person(val)
            elif isinstance(data, list):
                for obj in data:
                    handle_person(obj)
    except Exception:
        pass


def _preprocess_image(img: "Image.Image", bw: bool = True) -> "Image.Image":
    if not PIL_AVAILABLE:
        return img
    out = img
    try:
        if bw:
            out = ImageOps.grayscale(out)
        out = ImageOps.autocontrast(out)
        out = out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
        w, h = out.size
        pixels = w * h
        if pixels > OCR_MAX_PIXELS:
            scale = (OCR_MAX_PIXELS / pixels) ** 0.5
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            out = out.resize(new_size, Image.LANCZOS)
    except Exception:
        return img
    return out


async def _download_pdf_bytes(url: str) -> bytes:
    data = b""
    try:
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=PDF_TIMEOUT_SEC) as resp:
                    if resp.status != 200:
                        return b""
                    content_length = int(resp.headers.get("Content-Length", 0))
                    if content_length and content_length > PDF_MAX_SIZE_MB * 1024 * 1024:
                        return b""
                    data = await resp.read()
        except Exception:
            import urllib.request

            with urllib.request.urlopen(url, timeout=PDF_TIMEOUT_SEC) as resp:
                hdr = resp.headers.get("Content-Length")
                cl = int(hdr) if hdr else 0
                data = resp.read()
                if cl and cl > PDF_MAX_SIZE_MB * 1024 * 1024:
                    return b""
        if len(data) > PDF_MAX_SIZE_MB * 1024 * 1024:
            return b""
    except Exception:
        return b""
    return data


def _py_pdf_texts(data: bytes, max_pages: int) -> List[str]:
    if PdfReader is None:
        return []
    try:
        reader = PdfReader(io.BytesIO(data))
        texts = []
        for page in reader.pages[:max_pages]:
            txt = page.extract_text() or ""
            if len(txt) > SAFE_TEXT_CHARS:
                txt = txt[:SAFE_TEXT_CHARS]
            texts.append(txt)
        return texts
    except Exception:
        return []


def _ocr_from_bytes(data: bytes, dpi: int, lang: str, oem: int, psm: int, max_pages: int) -> List[str]:
    if not (PDF2IMAGE_AVAILABLE and PYTESSERACT_AVAILABLE):
        return []
    try:
        pages: List["Image.Image"] = convert_from_bytes(
            data, dpi=dpi, first_page=1, last_page=max_pages
        )
    except Exception:
        return []
    texts = []
    cfg = f"--oem {oem} --psm {psm}"
    for idx, img in enumerate(pages, 1):
        img2 = _preprocess_image(img, bw=OCR_BW)
        try:
            txt = pytesseract.image_to_string(img2, lang=lang, config=cfg) or ""
        except Exception:
            txt = ""
        if len(txt) > SAFE_TEXT_CHARS:
            txt = txt[:SAFE_TEXT_CHARS]
        texts.append(txt)
        human_logger.info(f"[OCR] Page {idx}: {len(txt)} chars")
    return texts



from typing import Tuple

BAD_NAME_TOKENS = {
    "Email", "E-mail", "Mail", "St", "Street", "Hours", "Support",
    "Customer", "Service", "Info", "Contact", "Us", "Department",
}
GENERIC_LOCALPARTS = {"info", "support", "service", "customerservice", "hello", "contact"}

async def _infer_role_name_near_email(page, text: str, email: str) -> Tuple[str, str]:
    """
    Infer the role and name near a given email.
    1) Try parent element of an anchor containing the email.
    2) Fallback: scan a window of text surrounding the email in plain text.
    """
    try:
        # Try DOM anchor parent first
        el = page.locator(f"a[href*='{email}']").first
        parent_text = ""
        if await el.count():
            try:
                parent_text = await el.locator("xpath=..").inner_text()
            except Exception:
                parent_text = ""

        role_match = ROLE_REGEX.search(parent_text or "")
        name_match = NAME_REGEX.search(parent_text or "")

        if not (role_match or name_match):
            # Fallback: use a window of text surrounding the email
            idx = text.find(email)
            if idx >= 0:
                window_start = max(0, idx - 400)  # expand to 400 chars before
                window_end = min(len(text), idx + len(email) + 200)  # include 200 chars after
                window = text[window_start:window_end]

                # Split on line breaks or punctuation to isolate candidate phrases
                chunks = re.split(r"[\n\r,;•·|-]", window)
                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    if not role_match:
                        role_match = ROLE_REGEX.search(chunk) or role_match
                    if not name_match:
                        name_match = NAME_REGEX.search(chunk) or name_match
                    if role_match and name_match:
                        break  # stop early if both found

        role = role_match.group(0) if role_match else ""
        name = name_match.group(1) if name_match else ""

        # Filter generic/bad tokens
        if name in BAD_NAME_TOKENS:
            name = ""

        # If the role is missing and the mailbox is generic, avoid attributing to a person
        localpart = (email or "").split("@", 1)[0].lower()
        if (not role) and (localpart in GENERIC_LOCALPARTS):
            name = ""

        return role, name
    except Exception:
        return "", ""



async def _extract_people_from_pdf_texts(page, texts: List[str], source_url: str, cfg: dict):
    joined = "\n".join(texts or [])
    for m in EMAIL_REGEX.findall(joined):
        email = m
        role, name = await _infer_role_name_near_email(page, joined, email)
        add_leadership(role, name, email, source_url, "pdf-window", cfg)


async def _extract_emails_from_texts(page, texts: List[str], source_url: str, cfg: dict, tag: str = "pdf-text"):
    raw = "\n".join(texts)

    # Direct emails
    for m in EMAIL_REGEX.findall(raw):
        email = m
        role, name = await _infer_role_name_near_email(page, raw, email)
        add_leadership(role, name, email, source_url, f"{tag}-context", cfg)
        add_email(email, source_url, tag, cfg)

    # Deobfuscate and re-scan
    deob = deobfuscate_email(raw)
    for m in EMAIL_REGEX.findall(deob):
        email = m
        role, name = await _infer_role_name_near_email(page, deob, email)
        add_leadership(role, name, email, source_url, f"{tag}-context", cfg)
        add_email(email, source_url, tag, cfg)

    # mailto: links embedded in text
    for m in re.findall(
        r"mailto:([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", raw, flags=re.I
    ):
        email = m
        role, name = await _infer_role_name_near_email(page, raw, email)
        add_leadership(role, name, email, source_url, f"{tag}-mailto", cfg)
        add_email(email, source_url, "pdf-mailto", cfg)


async def parse_and_extract_pdf(page, pdf_url: str, host: str, cfg: dict):
    if PDF_COUNTS_BY_DOMAIN.get(host, 0) >= PDF_MAX_PER_DOMAIN:
        return
    data = await _download_pdf_bytes(pdf_url)
    if not data:
        return

    texts = _py_pdf_texts(data, PDF_MAX_PAGES)
    native_chars = sum(len(t) for t in texts)
    if (native_chars < 256) and PDF_OCR_ENABLED:
        if not (PDF2IMAGE_AVAILABLE and PYTESSERACT_AVAILABLE):
            human_logger.info(
                "[OCR] Skipped: install 'tesseract-ocr poppler-utils' and Python deps: 'pdf2image pytesseract pillow'"
            )
        else:
            human_logger.info(f"[OCR] Engaging OCR for {pdf_url} (native text={native_chars} chars)")
            ocr_texts = _ocr_from_bytes(data, OCR_DPI, OCR_LANG, OCR_OEM, OCR_PSM, PDF_MAX_PAGES)
            if ocr_texts:
                await _extract_emails_from_texts(page, ocr_texts, pdf_url, cfg, tag="pdf-ocr")
            texts = texts or ocr_texts

    if texts:
        await _extract_people_from_pdf_texts(page, texts, pdf_url, cfg)
        await _extract_emails_from_texts(page, texts, pdf_url, cfg, tag="pdf-text")

    PDF_COUNTS_BY_DOMAIN[host] = PDF_COUNTS_BY_DOMAIN.get(host, 0) + 1
    log_json("pdf_parsed", url=pdf_url, host=host)


async def extract_emails_from_page(page, base_url, cfg):
    try:
        content = await page.content()
        for m in EMAIL_REGEX.findall(content):
            add_email(m, base_url, "html", cfg)
        for c in re.findall(r'data-cfemail="([0-9a-fA-F]+)"', content):
            add_email(decode_cfemail(c), base_url, "cfemail", cfg)
    except Exception:
        pass


def same_domain(base, absolute_url):
    try:
        return urlparse(base).netloc == urlparse(absolute_url).netloc
    except Exception:
        return False


def allowed_path(url: str, allowed_paths: List[str]) -> bool:
    if not allowed_paths:
        return True
    try:
        path = urlparse(url).path or "/"
        if not path.endswith("/") and url.endswith("/"):
            path += "/"
        return any(path.startswith(p) for p in allowed_paths)
    except Exception:
        return True


async def process_page(
    page,
    url: str,
    depth: int,
    cfg: dict,
    dom_cfg: DomainConfig,
    depth_limit: int,
    max_links_per_page: int,
    max_fanout_per_page: int,
    time_limit_sec: int,
) -> List[str]:
    host = host_root(url)
    if host in DOMAIN_START_TS:
        elapsed = time.time() - DOMAIN_START_TS[host]
        if elapsed > time_limit_sec:
            log_json("domain_budget_exceeded", host=host, elapsed=elapsed)
            return []
    else:
        DOMAIN_START_TS[host] = time.time()

    async def route_block(route):
        req_url = route.request.url
        for pat in BLOCK_PATTERNS:
            try:
                if re.search(pat, req_url, flags=re.IGNORECASE):
                    return await route.abort()
            except Exception:
                pass
        return await route.continue_()

    await page.route("**/*", route_block)

    async def on_response(response):
        try:
            if not same_domain(dom_cfg.base, response.url):
                return
            ct = response.headers.get("content-type", "").lower()
            if not any(ct.startswith(t) for t in TEXT_CONTENT_TYPES):
                return
            text = await response.text()
            text_deob = deobfuscate_email(text)
            for m in EMAIL_REGEX.findall(text_deob):
                add_email(m, response.url, "network", cfg)
            for c in re.findall(r'data-cfemail="([0-9a-fA-F]+)"', text):
                add_email(decode_cfemail(c), response.url, "network-cfemail", cfg)
        except Exception:
            pass

    page.on("response", on_response)

    # Navigation
    try:
        human_logger.info(f"[NAV] depth={depth} {url}")
        log_json("page_start", url=url, depth=depth)
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(0.4)
        await progressive_reveal(page)
        await parse_json_ld_for_people(page, url, cfg)
    except Exception as e:
        human_logger.info(f"[LOAD FAIL] {url} — {e}")
        try:
            html_content = await page.content()
            for m in EMAIL_REGEX.findall(html_content):
                add_email(m, url, "html-after-fail", cfg)
        except Exception:
            pass
        try:
            page.off("response", on_response)
        except Exception:
            pass
        log_json("page_done", url=url, depth=depth, status="failed")
        return []

    # Emails from HTML
    await extract_emails_from_page(page, url, cfg)

    # Infer role/name near emails from HTML text
    try:
        html_content = await page.content()
        html_deob = deobfuscate_email(html_content or "")
        for m in EMAIL_REGEX.findall(html_deob):
            email = m
            role, name = await _infer_role_name_near_email(page, html_deob, email)
            add_leadership(role, name, email, url, "html-context", cfg)
    except Exception:
        pass

    # Domain-specific extractors (host-root keyed)
    hostkey = host_root(dom_cfg.base)  # e.g., 'morrissheppardfcu.org'
    DOMAIN_EXTRACTORS: Dict[str, Callable] = {
        "morrissheppardfcu.org": extract_msfcu_leadership,
        "jovia.org": extract_jovia_support,
    }
    extractor = DOMAIN_EXTRACTORS.get(hostkey)
    if extractor:
        try:
            log_json("extractor_invoked", hostkey=hostkey, url=url)
            await extractor(page, url, cfg)
        except Exception as e:
            human_logger.info(f"[EXTRACTOR FAIL] {hostkey} — {e}")

    # Discover internal links + schedule PDFs
    new_links_relevant: List[str] = []
    new_links_neutral: List[str] = []
    try:
        anchors = await page.query_selector_all("a")
        for a in anchors:
            href = await a.get_attribute("href")
            if not href:
                continue
            absolute = urljoin(url, href)
            if url_is_denied(absolute):
                continue

            # PDFs: parse but don't enqueue
            if absolute.lower().endswith(".pdf"):
                host_pdf = host_root(absolute)
                if PDF_COUNTS_BY_DOMAIN.get(host_pdf, 0) < PDF_MAX_PER_DOMAIN:
                    asyncio.create_task(parse_and_extract_pdf(page, absolute, host_pdf, cfg))
                continue

            # Normal link handling
            if same_domain(dom_cfg.base, absolute) and allowed_path(absolute, dom_cfg.allowed_paths):
                cu = canonical_url(absolute)
                if cu in visited_pages:
                    continue
                if url_is_people_relevant(absolute):
                    new_links_relevant.append(absolute)
                else:
                    new_links_neutral.append(absolute)
                if len(new_links_relevant) >= max_fanout_per_page:
                    break
    except Exception:
        pass

    try:
        page.off("response", on_response)
    except Exception:
        pass

    ordered = new_links_relevant + new_links_neutral
    log_json("page_done", url=url, depth=depth, status="ok", discovered=len(ordered))
    return list(dict.fromkeys(ordered))[:max_fanout_per_page]


async def extract_jovia_support(page, url: str, cfg: dict):
    try:
        mailto_loc = page.locator("a[href^='mailto:']")
        count = await mailto_loc.count()
        for i in range(count):
            href = (await mailto_loc.nth(i).get_attribute("href")) or ""
            email = href.split(":", 1)[1].split("?", 1)[0] if href.startswith("mailto:") else href
            parent = mailto_loc.nth(i).locator("xpath=..")
            context_text = ""
            try:
                context_text = await parent.inner_text()
            except Exception:
                pass

            if not context_text or len(context_text.strip()) < 10:
                try:
                    grand = parent.locator("xpath=..")
                    if await grand.count():
                        gp_text = await grand.first.inner_text()
                        if gp_text and len(gp_text.strip()) > len(context_text.strip()):
                            context_text = gp_text
                except Exception:
                    pass

            name_match = NAME_REGEX.search(context_text or "")
            role_match = ROLE_REGEX.search(context_text or "")
            add_leadership(
                role_match.group(0) if role_match else "",
                name_match.group(1) if name_match else "",
                email or "",
                url,
                "leadership-mailto",
                cfg,
            )
    except Exception as e:
        human_logger.warning(f"[MAILTO-FAIL] {url} — {e}")

    # Click-here / cfemail extraction
    try:
        click_here_loc = page.locator(
            "//a[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'click here')]"
        )
        c = await click_here_loc.count()
        for i in range(c):
            a = click_here_loc.nth(i)
            try:
                if not (await a.is_visible()):
                    continue
                href = (await a.get_attribute("href")) or ""
                onclick = (await a.get_attribute("onclick")) or ""
                if "mailto:" in href:
                    add_leadership("", "", href, url, "click-here", cfg)
                elif "/cdn-cgi/l/email-protection" in href:
                    span = a.locator("span.__cf_email__")
                    if await span.count():
                        cfhex = await span.nth(0).get_attribute("data-cfemail")
                        add_leadership("", "", decode_cfemail(cfhex or ""), url, "cfemail", cfg)
                if "mailto:" in (onclick or ""):
                    m = re.search(r"mailto:([^\)\"']+)", onclick)
                    if m:
                        add_leadership("", "", m.group(0), url, "onclick", cfg)
            except Exception:
                continue
    except Exception:
        pass


async def extract_msfcu_leadership(page, url: str, cfg: dict):
    """
    Structured leadership extraction for vertical Role · Name · Email layouts
    (e.g., "CEO Jennifer Price jennifer@mstfcu.org").
    """
    try:
        staff_head = page.locator(
            "//h2[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'meet our staff')]"
        )
        if await staff_head.count() == 0:
            staff_head = page.locator(
                "//*[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'meet our staff')]"
            )
        if await staff_head.count() == 0:
            staff_head = page.locator(
                "//*[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'meet our team')"
                " or contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'leadership')]"
            )

        container = staff_head.first if await staff_head.count() else page
        text = ""
        try:
            text = await container.inner_text()
        except Exception:
            text = ""

        # Role · Name · Email (allow separators dot/middle-dot/dash/colon/whitespace)
        pattern = re.compile(
            r"(?is)\b("
            r"CEO|Chief\s+Executive\s+Officer|President|Vice\s*President|VP|AVP|"
            r"Branch\s*Manager|Manager|Director|Supervisor|Officer|Loan\s*Officer|Member\s*Services"
            r")\b[\s·:\-]*"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})[\s·:\-]*"
            r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})"
        )

        for m in pattern.finditer(text or ""):
            role, name, email = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            add_leadership(role, name, email, url, "leadership-structured", cfg)

        # Fallback: associate mailto anchors with nearby role/name text
        mailto_loc = page.locator("a[href^='mailto:']")
        count = await mailto_loc.count()
        for i in range(count):
            href = await mailto_loc.nth(i).get_attribute("href")
            parent = mailto_loc.nth(i).locator("xpath=..")
            context_text = ""
            try:
                context_text = await parent.inner_text()
            except Exception:
                pass
            email = (href or "").replace("mailto:", "").split("?", 1)[0]
            name_match = NAME_REGEX.search(context_text or "")
            role_match = ROLE_REGEX.search(context_text or "")
            add_leadership(
                role_match.group(0) if role_match else "",
                name_match.group(1) if name_match else "",
                email or "",
                url,
                "leadership-mailto-fallback",
                cfg,
            )

    except Exception:
        pass


UNIFIED_FILE = os.path.join(BASE_DIR, "leadership_results.csv")


def _cu_label(name: str) -> str:
    try:
        return (name or "").strip().upper()
    except Exception:
        return ""


def build_unified_rows():
    rows = []

    # EMAIL rows
    for rec in email_records:
        ts = rec.get("timestamp")
        cu = _cu_label(rec.get("credit_union"))
        addr = rec.get("email")
        via = rec.get("found_via")
        src = rec.get("source_url")
        formatted = f"[EMAIL] {ts}  {cu}  {addr}  via={via}"
        rows.append(
            {
                "type": "EMAIL",
                "timestamp": ts,
                "credit_union": cu,
                "role": "",
                "name": "",
                "address": addr,
                "found_via": via,
                "source_url": src,
                "confidence": "",
                "formatted": formatted,
            }
        )

    # LEADERSHIP rows
    for rec in leadership_records:
        ts = rec.get("timestamp")
        cu = _cu_label(rec.get("credit_union"))
        role = rec.get("role") or ""
        name = rec.get("name") or ""
        email = rec.get("email") or ""
        via = rec.get("found_via")
        src = rec.get("source_url")
        conf = rec.get("confidence")
        display_addr = f"mailto:{email}" if email and not email.startswith("mailto:") else (email or "")
        formatted = f"[LEADERSHIP] {ts}  {cu}  · {role or name}  {display_addr or ''}  via={via}  conf={conf}"
        rows.append(
            {
                "type": "LEADERSHIP",
                "timestamp": ts,
                "credit_union": cu,
                "role": role,
                "name": name,
                "address": display_addr,
                "found_via": via,
                "source_url": src,
                "confidence": conf,
                "formatted": formatted,
            }
        )
    return rows


async def crawl_domain(
    start_url: str,
    credit_union_name: str,
    allowed_domains: set,
    concurrency: int,
    depth_limit: int,
    max_links_per_page: int,
    fanout_per_page: int,
    time_limit_sec: int,
    stealth: bool,
):
    dom_cfg = select_config_for(start_url)
    cfg = {
        "credit_union": credit_union_name or dom_cfg.name,
        "base": dom_cfg.base,
        "valid_email_domains": dom_cfg.valid_email_domains or allowed_domains,
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        user_agents = [
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
            ),
            (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            ),
        ]
        ua = random.choice(user_agents) if stealth else user_agents[0]
        viewport = {"width": random.randint(1100, 1440), "height": random.randint(700, 900)} if stealth else None

        context_args = {
            "user_agent": ua,
            "locale": "en-US",
        }
        if viewport:
            context_args["viewport"] = viewport
        context_args["color_scheme"] = "light"
        context_args["timezone_id"] = "America/Chicago"
        context_args["extra_http_headers"] = {"Accept-Language": "en-US,en;q=0.9"}

        context = await browser.new_context(**context_args)

        if stealth:
            await context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
                Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
                Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
                """
            )

        visited_pages.clear()
        PDF_COUNTS_BY_DOMAIN.clear()


        q: asyncio.Queue = asyncio.Queue()
        for s in dom_cfg.seeds:
            await q.put((s, 0))

        human_logger.info(f"[DOMAIN] {credit_union_name} — seeds={dom_cfg.seeds}")
        log_json("domain_start", name=credit_union_name, base=dom_cfg.base, stealth=stealth)

        async def worker():
            while True:
                try:
                    url, depth = await asyncio.wait_for(q.get(), timeout=0.75)
                except asyncio.TimeoutError:
                    if q.empty():
                        break
                    else:
                        continue

                cu = canonical_url(url)
                if cu in visited_pages or depth > depth_limit:
                    q.task_done()
                    continue

                visited_pages.add(cu)
                page = await context.new_page()
                page.set_default_timeout(30000)
                page.set_default_navigation_timeout(45000)

                try:
                    links = await process_page(
                        page,
                        url,
                        depth,
                        cfg,
                        dom_cfg,
                        depth_limit,
                        max_links_per_page,
                        fanout_per_page,
                        time_limit_sec,
                    )
                    for link in links:
                        await q.put((link, depth + 1))
                except Exception as e:
                    human_logger.info(f"[PAGE FAIL] {url} — {e}")
                finally:
                    try:
                        await page.close()
                    except Exception:
                        pass

                q.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*tasks)
        await context.close()
        await browser.close()
        log_json(
            "domain_done",
            name=credit_union_name,
            emails_found=len(found_emails),
            leadership_found=len(leadership_records),
        )


def save_state(checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        with open(os.path.join(checkpoint_dir, "visited_pages.json"), "w", encoding="utf-8") as f:
            json.dump(sorted(list(visited_pages)), f)
    except Exception:
        pass


def load_state(checkpoint_dir: str) -> None:
    try:
        fp = os.path.join(checkpoint_dir, "visited_pages.json")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                for u in json.load(f) or []:
                    visited_pages.add(canonical_url(u))
    except Exception:
        pass


def load_credit_union_csv(csv_path: str, sep: str = ",", encoding: str = "utf-8-sig") -> List[Tuple[str, str]]:
    df = pd.read_csv(csv_path, dtype=str, sep=sep, encoding=encoding).fillna("")
    original_cols = list(df.columns)

    def norm_key(c: str) -> str:
        return re.sub(r"\s+", "", (c or "").strip().lower())

    norm_map = {c: norm_key(c) for c in original_cols}
    url_candidates = [c for c, n in norm_map.items() if n in {"domain", "url", "website", "web", "homepage"}]
    name_candidates = [c for c, n in norm_map.items() if n in {"credit_union", "cuname", "name", "creditunion"}]

    if not url_candidates:

        def looks_like_url(s: str) -> bool:
            s = (s or "").strip()
            return bool(re.match(r"^[a-z0-9\.\-]+\.[a-z]{2,}(/.*)?$", s, re.I)) or bool(re.match(r"^https?://", s, re.I))

        score = {c: sum(1 for v in df[c].astype(str) if looks_like_url(v)) for c in original_cols}
        if score:
            best = max(score, key=score.get)
            if score.get(best, 0) > 0:
                url_candidates = [best]
    if df.shape[1] == 1 and not url_candidates:
        url_candidates = [original_cols[0]]

    url_col = url_candidates[0] if url_candidates else None
    name_col = name_candidates[0] if name_candidates else None

    def derive_name_from_url(u: str) -> str:
        if not u:
            return ""
        u2 = u if re.match(r"^[a-zA-Z][a-zA-Z0-9+\-\.]*://", u) else "https://" + u.lstrip("/")
        host = urlparse(u2).netloc.lower().lstrip("www.")
        base = host.split(".")[0]
        label = re.sub(r"[-_]", " ", base).strip()
        return label.upper() if label else host.upper()

    credit_unions: List[Tuple[str, str]] = []
    seen_hosts = set()

    for _, row in df.iterrows():
        raw_url = str(row[url_col]).strip() if url_col else ""
        if not raw_url:
            for c in original_cols:
                candidate = str(row[c]).strip()
                if candidate and (re.search(r"\.[a-z]{2,}", candidate, re.I) or re.match(r"^https?://", candidate, re.I)):
                    raw_url = candidate
                    break
        if not raw_url:
            continue

        url = normalize_url(raw_url)
        if not url:
            continue

        raw_name = str(row[name_col]).strip() if name_col else ""
        name = raw_name if raw_name else derive_name_from_url(url)
        h = host_root(url)
        if h in seen_hosts:
            continue
        seen_hosts.add(h)
        credit_unions.append((name, url))

    human_logger.info(
        f"[CSV] Parsed {len(credit_unions)} credit unions from {csv_path} (columns={original_cols}, url_col={url_col}, name_col={name_col})"
    )
    for preview in credit_unions[:3]:
        human_logger.info(f" [CSV sample] {preview}")
    return credit_unions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="CSV file with any columns; header-agnostic")
    parser.add_argument("--pdf", action="store_true", help="Enable PDF extraction")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR fallback for scanned PDFs")
    parser.add_argument("--ocr-lang", default="eng", help="Tesseract language code (default 'eng')")
    parser.add_argument("--ocr-dpi", type=int, default=200, help="Rasterization DPI (default 200)")
    parser.add_argument("--ocr-oem", type=int, default=1, help="Tesseract OEM (default 1)")
    parser.add_argument("--ocr-psm", type=int, default=6, help="Tesseract PSM (default 6)")
    parser.add_argument("--limit-domains", nargs="*", help="Only run for these domains (host roots)")
    parser.add_argument("--csv-sep", default=",", help="CSV separator (default ',')")
    parser.add_argument("--csv-encoding", default="utf-8-sig", help="CSV encoding (default 'utf-8-sig')")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Worker concurrency per domain")
    parser.add_argument("--depth-limit", type=int, default=DEFAULT_DEPTH_LIMIT, help="Max crawl depth")
    parser.add_argument("--max-links-per-page", type=int, default=DEFAULT_MAX_LINKS_PER_PAGE, help="Max links parsed per page")
    parser.add_argument("--fanout-per-page", type=int, default=DEFAULT_MAX_FANOUT_PER_PAGE, help="Max internal links enqueued per page")
    parser.add_argument("--time-limit-per-domain", type=int, default=DEFAULT_TIME_LIMIT_SEC, help="Time budget per domain (sec)")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, help="Directory for checkpoints")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Directory for logs (JSONL + rotating)")
    parser.add_argument("--stealth", action="store_true", help="Enable stealth mode (UA/viewport masking)")
    args = parser.parse_args()

    setup_logging(args.log_dir)
    human_logger.info("Starting mergedbattle (merged foundation + unified CSV + anti-freeze)")

    global PDF_ENABLED, PDF_OCR_ENABLED, OCR_LANG, OCR_DPI, OCR_OEM, OCR_PSM, STEALTH
    PDF_ENABLED = args.pdf
    PDF_OCR_ENABLED = args.ocr
    OCR_LANG = args.ocr_lang
    OCR_DPI = args.ocr_dpi
    OCR_OEM = args.ocr_oem
    OCR_PSM = args.ocr_psm
    STEALTH = bool(args.stealth)

    if PDF_OCR_ENABLED and not (PDF2IMAGE_AVAILABLE and PYTESSERACT_AVAILABLE):
        human_logger.warning(
            "[WARN] OCR requested but pdf2image/pytesseract not available. "
            "Install: 'tesseract-ocr poppler-utils' and Python deps: 'pdf2image pytesseract pillow'."
        )

    load_state(args.checkpoint_dir)
    credit_unions = load_credit_union_csv(args.csv_file, sep=args.csv_sep, encoding=args.csv_encoding)
    if args.limit_domains:
        credit_unions = [(n, d) for n, d in credit_unions if host_root(d) in set(args.limit_domains)]

    if not credit_unions:
        human_logger.warning("[WARN] No valid rows found in CSV.")
        sys.exit(1)

    loop = asyncio.get_event_loop()
    for name, domain_url in tqdm(credit_unions, desc="Credit Unions"):
        try:
            # External watchdog: hard timeout per domain
            loop.run_until_complete(
                asyncio.wait_for(
                    crawl_domain(
                        domain_url,
                        name,
                        {host_root(domain_url)},
                        args.concurrency,
                        args.depth_limit,
                        args.max_links_per_page,
                        args.fanout_per_page,
                        args.time_limit_per_domain,
                        STEALTH,
                    ),
                    timeout=args.time_limit_per_domain,
                )
            )
        except asyncio.TimeoutError:
            human_logger.warning(
                f"[TIMEOUT] Domain watchdog hit for {name} ({host_root(domain_url)}); moving on."
            )
            continue
        except Exception as e:
            human_logger.info(f"[RUN FAIL] {domain_url} — {e}")
            continue
        finally:
            save_state(args.checkpoint_dir)

    # Single unified CSV output
    try:
        rows = build_unified_rows()
        if rows:
            df = pd.DataFrame(rows)
            if os.path.exists(UNIFIED_FILE):
                df.to_csv(UNIFIED_FILE, mode="a", header=False, index=False)
            else:
                df.to_csv(UNIFIED_FILE, mode="w", header=True, index=False)
            human_logger.info(f"[DONE] Unified results saved to {UNIFIED_FILE} ({len(rows)})")
            log_json("results_unified_saved", file=UNIFIED_FILE, count=len(rows))
        else:
            human_logger.info("[INFO] No rows captured for unified output.")
    except Exception as e:
        human_logger.info(f"[SAVE FAIL] unified output — {e}")


if __name__ == "__main__":
    main()
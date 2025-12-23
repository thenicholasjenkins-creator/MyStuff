
# -*- coding: utf-8 -*-
import asyncio
import nest_asyncio
from playwright.async_api import async_playwright
import re
import random
import io
import json
import csv
from datetime import datetime
from urllib.parse import urlparse, urljoin
from tqdm.notebook import tqdm

# --- PDF import (optional) ---
try:
    from PyPDF2 import PdfReader  # preferred
    PDF_ENABLED = True
except ModuleNotFoundError:
    try:
        from pypdf import PdfReader  # fallback package name
        PDF_ENABLED = True
    except ModuleNotFoundError:
        PDF_ENABLED = False
        print("[WARN] PyPDF2/pypdf not installed; PDF parsing disabled. "
              "Run `!pip install PyPDF2` (or `!pip install pypdf`) and re-run the cell.")

nest_asyncio.apply()

# ---------------------------
# GLOBALS
# ---------------------------
found_emails = set()
visited_pages = set()

# Structured records: role, name, email, source, confidence, tag
people = []

# ---------------------------
# RUNTIME SETTINGS
# ---------------------------
MAX_DEPTH = 2
CONCURRENCY = 3
HEADLESS = True
STEALTH = True

# ---------------------------
# DOMAIN CONFIG
# ---------------------------
def domain_config_for(url: str):
    host = urlparse(url).netloc.lower()
    if "jovia.org" in host:
        return {
            "base": "https://www.jovia.org/",
            "allowed_paths": ["/", "/support", "/knowledge-base", "/learn", "/about", "/locations"],
            "valid_email_domains": {"jovia.org", "joviafinancial.com"},
            "start_urls": ["https://www.jovia.org/"],
            "name": "Jovia",
        }
    if "morrissheppardfcu.org" in host:
        return {
            "base": "http://morrissheppardfcu.org/",
            "allowed_paths": ["/", "/contact/"],
            "valid_email_domains": {"mstfcu.org", "morrissheppardfcu.org"},
            "start_urls": ["http://morrissheppardfcu.org/"],
            "name": "Morris Sheppard FCU",
        }
    return {
        "base": f"{urlparse(url).scheme}://{host}/",
        "allowed_paths": ["/"],
        "valid_email_domains": set(),   # accept any domain if empty
        "start_urls": [url],
        "name": host,
    }

# ---------------------------
# PERFORMANCE / BLOCKING
# ---------------------------
BLOCK_PATTERNS = [
    r".*\.(png|jpe?g|gif|webp|svg|ico)($|\?)",
    r".*\.(woff2?|ttf|otf)($|\?)",
    r".*\/(gtm|googletagmanager)\.com\/.*",
    r".*\/google-analytics\.com\/.*",
    r".*doubleclick\.net\/.*",
    r".*\/facebook\.com\/tr\/.*",
    r".*\/snapchat\.com\/.*",
    r".*\/hotjar\.com\/.*",
    # Don't block PDFs
]

TEXT_CONTENT_TYPES = (
    "text/html",
    "application/json",
    "text/plain",
    "application/xhtml+xml",
    "application/xml",
)

PLACEHOLDER_EMAILS = {"user@example.com", "[email protected]"}

# ---------------------------
# ROLE INFERENCE
# ---------------------------
ROLE_PATTERNS = [
    r"\bchief\s+executive\s+officer\b", r"\bceo\b",
    r"\bpresident\b", r"\bpresident\s*&\s*ceo\b",
    r"\bvice\s+president\b", r"\bvp\b", r"\bsvp\b", r"\bevp\b",
    r"\bmanager\b", r"\bbranch\s+manager\b",
    r"\bchief\s+financial\s+officer\b", r"\bcfo\b",
    r"\bchief\s+operating\s+officer\b", r"\bcoo\b",
    r"\bchief\s+marketing\s+officer\b", r"\bcmo\b",
    r"\bchief\s+information\s+officer\b", r"\bcio\b", r"\bcto\b",
    r"\bchair\b", r"\bchairman\b", r"\bchairperson\b", r"\bboard\b",
    r"\bdirector\b", r"\btreasurer\b", r"\bsecretary\b",
    r"\bcompliance\s+officer\b",
    r"\bloan\s*officer\b",
    r"\bmember\s*services\b",
]
ROLE_COMPILED = [re.compile(pat, re.I) for pat in ROLE_PATTERNS]

def infer_role_from_text(text: str) -> str:
    if not text:
        return ""
    matches = []
    for rc in ROLE_COMPILED:
        for m in rc.finditer(text):
            matches.append(m.group(0))
    if not matches:
        return ""
    matches.sort(key=len, reverse=True)  # prefer longer, more explicit titles
    return matches[0].strip()

def infer_name_from_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s+[A-Z][a-z]+)?)\b", text)
    return m.group(1).strip() if m else ""

def is_leadership_role(role: str) -> bool:
    return bool(role) and any(rc.search(role) for rc in ROLE_COMPILED)

# ---------------------------
# EMAIL HELPERS
# ---------------------------
def decode_cfemail(cfhex: str) -> str:
    if not cfhex or len(cfhex) < 4:
        return ""
    try:
        r = int(cfhex[:2], 16)
        return "".join(chr(int(cfhex[i:i+2], 16) ^ r) for i in range(2, len(cfhex), 2))
    except Exception:
        return ""

def extract_emails(html_or_text: str):
    if not html_or_text:
        return set()
    text = (html_or_text.replace("[at]", "@").replace("(at)", "@").replace(" at ", "@")
                         .replace("[dot]", ".").replace("(dot)", ".").replace(" dot ", "."))
    cf_matches = re.findall(r'data-cfemail="([0-9a-fA-F]+)"', text)
    decoded = [decode_cfemail(m) for m in cf_matches if m]
    base_matches = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return set([e for e in decoded + base_matches if e])

def email_domain(email: str) -> str:
    try:
        return email.split("@", 1)[1].lower()
    except Exception:
        return ""

def keep_email(email: str, valid_domains: set) -> bool:
    d = email_domain(email)
    if email in PLACEHOLDER_EMAILS:
        return False
    if valid_domains and d not in valid_domains:
        return False
    return True

def same_domain(base, absolute_url):
    try:
        return urlparse(base).netloc == urlparse(absolute_url).netloc
    except:
        return False

def allowed_path(url: str, allowed_paths: list) -> bool:
    if not allowed_paths:
        return True
    try:
        path = urlparse(url).path or "/"
        if not path.endswith("/") and url.endswith("/"):
            path += "/"
        return any(path.startswith(p) for p in allowed_paths)
    except:
        return True

def record_person(role: str, name: str, email: str, source: str, tag: str,
                  valid_domains: set, confidence: float = 0.6):
    email = (email or "").strip()
    if not email or not keep_email(email, valid_domains):
        return
    key = (role.strip().lower(), name.strip().lower(), email.lower(), source)
    for p in people:
        if (p["role"].strip().lower(), p["name"].strip().lower(),
            p["email"].lower(), p["source"]) == key:
            break
    else:
        people.append({
            "role": role.strip(),
            "name": name.strip(),
            "email": email,
            "source": source,
            "found_via": tag,
            "confidence": confidence
        })
    if email not in found_emails:
        found_emails.add(email)
        print(f"[FOUND {tag}] {email} ({role or 'role: ?'} — {name or 'name: ?'}; from {source})")

def add_email_only(email_or_href: str, source: str, tag: str, valid_domains: set):
    if not email_or_href:
        return
    email = email_or_href.strip()
    if email.startswith("mailto:"):
        email = email.split(":", 1)[1]
    email = email.split("?", 1)[0]
    if re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email) and keep_email(email, valid_domains):
        if email not in found_emails:
            found_emails.add(email)
            print(f"[FOUND {tag}] {email} (from {source})")

# ---------------------------
# PDF PARSING
# ---------------------------
def parse_pdf_emails(pdf_bytes: bytes, source_url: str, valid_domains: set):
    if not PDF_ENABLED:
        return
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                full_text.append(t)
        joined = "\n".join(full_text)
        for email in extract_emails(joined):
            if not keep_email(email, valid_domains):
                continue
            idx = joined.find(email)
            window = joined[max(0, idx-180):idx]  # look behind the email
            role = infer_role_from_text(window)
            name = infer_name_from_text(window)
            confidence = 0.8 if is_leadership_role(role) else 0.5
            record_person(role, name, email, source_url, "pdf", valid_domains, confidence)
    except Exception as e:
        print(f"[PDF PARSE FAIL] {source_url} — {e}")

async def fetch_and_parse_pdf(req_ctx, url: str, valid_domains: set):
    if not PDF_ENABLED:
        return
    try:
        resp = await req_ctx.get(url)
        if not resp.ok:
            return
        ct = (resp.headers.get("content-type") or "").lower()
        if "application/pdf" in ct or url.lower().endswith(".pdf"):
            pdf_bytes = await resp.body()
            parse_pdf_emails(pdf_bytes, url, valid_domains)
    except Exception as e:
        print(f"[PDF FETCH FAIL] {url} — {e}")

# ---------------------------
# JSON-LD PARSING (schema.org)
# ---------------------------
async def parse_json_ld_for_people(page, url: str, valid_domains: set):
    try:
        scripts = page.locator("script[type='application/ld+json']")
        n = await scripts.count()
        for i in range(n):
            raw = await scripts.nth(i).inner_text()
            if not raw.strip():
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
                if email:
                    record_person(job or "", name or "", email, url, "json-ld", valid_domains, confidence=0.7)

            if isinstance(data, dict):
                if (data.get("@type") == "Person") or ("email" in data and ("jobTitle" in data or "role" in data)):
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

# ---------------------------
# PROGRESSIVE REVEAL (scroll + accordions/tabs/details)
# ---------------------------
async def progressive_reveal(page):
    """Scroll the page and expand common UI patterns to reveal hidden emails/roles."""
    try:
        # 1) Expand <details> elements
        details = page.locator("details")
        dcount = await details.count()
        for i in range(dcount):
            try:
                await page.evaluate("(el) => el.open = true", details.nth(i))
            except:
                pass

        # 2) Click accessible accordions/toggles/tabs by text or roles
        candidates = [
            page.get_by_role("button", name=re.compile(r"(show|expand|more|staff|team|leadership|contact|directory|view|open|details)", re.I)),
            page.get_by_role("tab"),
            page.locator("//button[contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'show') or "
                         "contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'staff') or "
                         "contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'leadership') or "
                         "contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'team') or "
                         "contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'contact') or "
                         "contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'directory')]"),
        ]
        for locator in candidates:
            cnt = await locator.count()
            for i in range(min(cnt, 20)):  # cap to avoid excessive clicking
                try:
                    el = locator.nth(i)
                    if await el.is_visible():
                        await el.scroll_into_view_if_needed()
                        await asyncio.sleep(random.uniform(0.1, 0.3))
                        await el.click()
                        await asyncio.sleep(random.uniform(0.15, 0.4))
                except:
                    continue

        # 3) Progressive scrolling to trigger lazy-load
        total_height = await page.evaluate("document.body.scrollHeight || document.documentElement.scrollHeight")
        step = max(300, int(total_height / 8))
        for y in range(0, total_height, step):
            try:
                await page.evaluate("window.scrollTo(0, arguments[0])", y)
                await asyncio.sleep(random.uniform(0.15, 0.35))
            except:
                break
    except Exception:
        pass

# ---------------------------
# DOMAIN-SPECIFIC EXTRACTORS
# ---------------------------
async def extract_jovia_support(page, url: str, valid_domains: set):
    """Jovia Support: mailto, click-here, Cloudflare cfemail (explicit)."""
    try:
        # mailto anchors (with context)
        mailto_loc = page.locator("a[href^='mailto:']")
        count = await mailto_loc.count()
        for i in range(count):
            a = mailto_loc.nth(i)
            href = await a.get_attribute("href")
            parent = a.locator("xpath=..")
            text = ""
            try:
                text = await parent.inner_text()
            except:
                pass
            role = infer_role_from_text(text)
            name = infer_name_from_text(text)
            email = (href or "").replace("mailto:", "").split("?", 1)[0]
            confidence = 0.7 if is_leadership_role(role) else 0.5
            record_person(role, name, email, url, "support-mailto", valid_domains, confidence)

        # “click here”
        click_here_loc = page.locator(
            "//a[contains(translate(normalize-space(.),"
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'click here')]"
        )
        c = await click_here_loc.count()
        for i in range(c):
            a = click_here_loc.nth(i)
            try:
                if not (await a.is_visible()):
                    continue
                parent_txt = ""
                try:
                    parent_txt = await a.locator("xpath=..").inner_text()
                except:
                    pass
                # Pre-click
                href = (await a.get_attribute("href")) or ""
                onclick = (await a.get_attribute("onclick")) or ""
                if "mailto:" in href:
                    role = infer_role_from_text(parent_txt)
                    name = infer_name_from_text(parent_txt)
                    email = href.replace("mailto:", "").split("?", 1)[0]
                    record_person(role, name, email, url, "click-here", valid_domains, 0.7)
                elif "/cdn-cgi/l/email-protection" in href:
                    span = a.locator("span.__cf_email__")
                    if await span.count():
                        cfhex = await span.first.get_attribute("data-cfemail")
                        email = decode_cfemail(cfhex or "")
                        role = infer_role_from_text(parent_txt)
                        name = infer_name_from_text(parent_txt)
                        record_person(role, name, email, url, "cfemail", valid_domains, 0.7)

                if "mailto:" in (onclick or ""):
                    m = re.search(r"mailto:([^\"')]+)", onclick)
                    if m:
                        role = infer_role_from_text(parent_txt)
                        name = infer_name_from_text(parent_txt)
                        record_person(role, name, m.group(1), url, "onclick", valid_domains, 0.7)

                await a.scroll_into_view_if_needed()
                await asyncio.sleep(random.uniform(0.15, 0.45))
                await a.click()
                await asyncio.sleep(random.uniform(0.25, 0.6))
                href_after = (await a.get_attribute("href")) or ""
                onclick_after = (await a.get_attribute("onclick")) or ""
                parent_txt_after = ""
                try:
                    parent_txt_after = await a.locator("xpath=..").inner_text()
                except:
                    pass

                if "mailto:" in href_after:
                    role = infer_role_from_text(parent_txt_after or parent_txt)
                    name = infer_name_from_text(parent_txt_after or parent_txt)
                    email = href_after.replace("mailto:", "").split("?", 1)[0]
                    record_person(role, name, email, url, "click-here-after", valid_domains, 0.7)
                elif "/cdn-cgi/l/email-protection" in href_after:
                    span = a.locator("span.__cf_email__")
                    if await span.count():
                        cfhex = await span.first.get_attribute("data-cfemail")
                        email = decode_cfemail(cfhex or "")
                        role = infer_role_from_text(parent_txt_after or parent_txt)
                        name = infer_name_from_text(parent_txt_after or parent_txt)
                        record_person(role, name, email, url, "cfemail-click", valid_domains, 0.7)
                elif "mailto:" in (onclick_after or ""):
                    m = re.search(r"mailto:([^\"')]+)", onclick_after)
                    if m:
                        role = infer_role_from_text(parent_txt_after or parent_txt)
                        name = infer_name_from_text(parent_txt_after or parent_txt)
                        record_person(role, name, m.group(1), url, "onclick-post", valid_domains, 0.7)
            except:
                continue
    except:
        pass

async def extract_msfcu_leadership(page, url: str, valid_domains: set):
    """Morris Sheppard FCU Contact: vertically stacked Role · Name · Email."""
    try:
        staff_head = page.locator(
            "//h2[contains(translate(normalize-space(.),"
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'meet our staff')]"
        )
        if await staff_head.count() == 0:
            staff_head = page.locator(
                "//*[contains(translate(normalize-space(.),"
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'meet our staff')]"
            )

        if await staff_head.count():
            container = staff_head.first.locator("xpath=..")
            if await container.count():
                txt = await container.inner_text()
                pattern = re.compile(
                    r"(?is)\b(CEO|Loan\s*Officer|Member\s*Services)\b[\s:\-]*"
                    r"([A-Za-z][A-Za-z .'\-]+)[\s:\-]*"
                    r"([A-Za-z0-9._%+-]+@mstfcu\.org)\b"
                )
                for m in pattern.finditer(txt):
                    role, name, email = m.group(1), m.group(2), m.group(3)
                    record_person(role, name, email, url, "leadership-text", valid_domains, 0.9)

        # mailto links (associate role/name)
        mailto_loc = page.locator("a[href^='mailto:']")
        count = await mailto_loc.count()
        for i in range(count):
            a = mailto_loc.nth(i)
            href = await a.get_attribute("href")
            parent = a.locator("xpath=..")
            text = ""
            try:
                text = await parent.inner_text()
            except:
                pass
            role = infer_role_from_text(text)
            name = infer_name_from_text(text)
            email = (href or "").replace("mailto:", "").split("?", 1)[0]
            confidence = 0.85 if is_leadership_role(role) else 0.6
            record_person(role, name, email, url, "leadership-mailto", valid_domains, confidence)
    except:
        pass

# ---------------------------
# PAGE CRAWLER
# ---------------------------
async def crawl_page(context, req_ctx, url: str, depth: int, cfg: dict):
    if url in visited_pages or depth > MAX_DEPTH:
        return []
    visited_pages.add(url)

    page = await context.new_page()
    page.set_default_timeout(30000)
    page.set_default_navigation_timeout(45000)

    async def route_block(route):
        req_url = route.request.url
        if req_url.lower().endswith(".pdf"):
            return await route.continue_()
        for pat in BLOCK_PATTERNS:
            if re.match(pat, req_url, flags=re.IGNORECASE):
                return await route.abort()
        return await route.continue_()
    await page.route("**/*", route_block)

    async def on_response(response):
        try:
            if not same_domain(cfg["base"], response.url):
                return
            ct = (response.headers.get("content-type") or "").lower()

            if PDF_ENABLED and ("application/pdf" in ct or response.url.lower().endswith(".pdf")):
                pdf_bytes = await response.body()
                parse_pdf_emails(pdf_bytes, response.url, cfg["valid_email_domains"])
                return

            if any(ct.startswith(t) for t in TEXT_CONTENT_TYPES):
                text = await response.text()
                for email in extract_emails(text):
                    if keep_email(email, cfg["valid_email_domains"]):
                        idx = text.find(email)
                        window = text[max(0, idx-160):idx]
                        role = infer_role_from_text(window)
                        name = infer_name_from_text(window)
                        confidence = 0.7 if is_leadership_role(role) else 0.5
                        record_person(role, name, email, response.url, "network", cfg["valid_email_domains"], confidence)
        except:
            pass
    page.on("response", on_response)

    new_links = []
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(random.uniform(0.25, 0.6))

        # Progressive reveal BEFORE scanning
        await progressive_reveal(page)

        # JSON-LD staff hints
        await parse_json_ld_for_people(page, url, cfg["valid_email_domains"])

        # HTML scan (with page-level role window)
        html = await page.content()
        for email in extract_emails(html):
            if keep_email(email, cfg["valid_email_domains"]):
                idx = html.find(email)
                window = html[max(0, idx-160):idx]
                role = infer_role_from_text(window)
                name = infer_name_from_text(window)
                confidence = 0.7 if is_leadership_role(role) else 0.5
                record_person(role, name, email, url, "HTML", cfg["valid_email_domains"], confidence)

        # Domain-specific logic
        host = urlparse(cfg["base"]).netloc.lower()
        if "jovia.org" in host and "/support" in url.lower():
            await extract_jovia_support(page, url, cfg["valid_email_domains"])
        if "morrissheppardfcu.org" in host and "/contact" in url.lower():
            await extract_msfcu_leadership(page, url, cfg["valid_email_domains"])

        # Collect internal links + proactively fetch PDFs
        for a in await page.query_selector_all("a"):
            href = await a.get_attribute("href")
            if not href:
                continue
            absolute = urljoin(url, href)
            if PDF_ENABLED and absolute.lower().endswith(".pdf") and same_domain(cfg["base"], absolute):
                await fetch_and_parse_pdf(req_ctx, absolute, cfg["valid_email_domains"])
            if same_domain(cfg["base"], absolute) and allowed_path(absolute, cfg["allowed_paths"]) and absolute not in visited_pages:
                new_links.append(absolute)

    except Exception as e:
        print(f"[LOAD FAIL] {url} — {e}")
        try:
            html = await page.content()
            for email in extract_emails(html):
                if keep_email(email, cfg["valid_email_domains"]):
                    idx = html.find(email)
                    window = html[max(0, idx-160):idx]
                    role = infer_role_from_text(window)
                    name = infer_name_from_text(window)
                    confidence = 0.7 if is_leadership_role(role) else 0.5
                    record_person(role, name, email, url, "HTML-after-fail", cfg["valid_email_domains"], confidence)
        except:
            pass
    finally:
        try:
            page.off("response", on_response)
        except:
            pass
        await page.close()

    return list(dict.fromkeys(new_links))[:40]

# ---------------------------
# CONCURRENT RUNNER
# ---------------------------
def random_ua_viewport():
    uas = [
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
         "AppleWebKit/537.36 (KHTML, like Gecko) "
         "Chrome/120.0.0.0 Safari/537.36"),
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
         "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"),
        ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"),
    ]
    ua = random.choice(uas)
    vw = random.randint(1100, 1440)
    vh = random.randint(700, 900)
    return ua, {"width": vw, "height": vh}

async def run(start_urls):
    async with async_playwright() as p:
        ua, viewport = random_ua_viewport()
        browser = await p.chromium.launch(headless=HEADLESS)

        context_args = {
            "user_agent": ua,
            "locale": "en-US",
            "timezone_id": "America/Chicago",
            "viewport": viewport,
            "color_scheme": "light",
            "extra_http_headers": {"Accept-Language": "en-US,en;q=0.9"},
        }
        context = await browser.new_context(**context_args)

        # STEALTH
        if STEALTH:
            await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
            Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            """)

        # Request context for proactive PDF fetches
        req_ctx = await p.request.new_context(
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9", "Referer": start_urls[0]}
        )

        queues = []
        cfgs = []
        for seed in start_urls:
            cfg = domain_config_for(seed)
            cfgs.append(cfg)
            q = asyncio.Queue()
            for u in cfg["start_urls"]:
                await q.put((u, 0))
            queues.append((q, cfg))

        pbar = tqdm(total=0, desc="Crawling pages", unit="page")

        async def worker(q, cfg):
            while True:
                try:
                    url, depth = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if q.empty():
                        break
                    else:
                        continue
                links = await crawl_page(context, req_ctx, url, depth, cfg)
                for link in links:
                    await q.put((link, depth + 1))
                pbar.total = len(visited_pages) + sum(q2.qsize() for q2, _ in queues)
                pbar.update(1)
                q.task_done()

        tasks = []
        for q, cfg in queues:
            for _ in range(CONCURRENCY):
                tasks.append(asyncio.create_task(worker(q, cfg)))

        await asyncio.gather(*tasks)
        pbar.close()
        await req_ctx.dispose()
        await context.close()
        await browser.close()

# ---------------------------
# EXPORT HELPERS (CSV/JSON)
# ---------------------------
def export_results():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    people_csv = f"people_{ts}.csv"
    people_json = f"people_{ts}.json"
    emails_csv = f"emails_{ts}.csv"
    emails_json = f"emails_{ts}.json"

    # People → CSV
    with open(people_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["role", "name", "email", "source", "found_via", "confidence"])
        w.writeheader()
        for p in people:
            w.writerow(p)

    # People → JSON
    with open(people_json, "w", encoding="utf-8") as f:
        json.dump(people, f, ensure_ascii=False, indent=2)

    # Emails → CSV
    with open(emails_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["email"])
        for e in sorted(found_emails):
            w.writerow([e])

    # Emails → JSON
    with open(emails_json, "w", encoding="utf-8") as f:
        json.dump(sorted(list(found_emails)), f, ensure_ascii=False, indent=2)

    print("\n🗃️  Exported files:")
    print(f"  • {people_csv}")
    print(f"  • {people_json}")
    print(f"  • {emails_csv}")
    print(f"  • {emails_json}")

# ---------------------------
# RUN (both domains) + EXPORT
# ---------------------------
start_urls = [
    "https://www.jovia.org/",
    "http://morrissheppardfcu.org/",
]
await run(start_urls)

print("\n==============================")
print("🎉 DONE — PEOPLE FOUND (role · name · email):")
print("==============================")
leadership = [p for p in people if is_leadership_role(p["role"])]
others = [p for p in people if not is_leadership_role(p["role"])]

if leadership:
    print("\n👑 Leadership:")
    for p in leadership:
        print(f"- {p['role'] or '?'} · {p['name'] or '?'} · {p['email']}  "
              f"(source: {p['source']}; via: {p['found_via']}; conf: {p['confidence']:.2f})")

if others:
    print("\n👥 Others (non-leadership or unspecified):")
    for p in others:
        print(f"- {p['role'] or '?'} · {p['name'] or '?'} · {p['email']}  "
              f"(source: {p['source']}; via: {p['found_via']}; conf: {p['confidence']:.2f})")

print("\n===================")
print("📧 UNIQUE EMAILS (filtered):")
print("===================")
for e in sorted(found_emails):
    print(e)

# Write CSV/JSON
export_results()

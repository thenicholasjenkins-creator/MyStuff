
# =========================================================
# STEP 0: Install dependencies (Colab)
# =========================================================
!pip -q install playwright nest_asyncio tqdm spacy
!playwright install chromium
!python -m spacy download en_core_web_sm

# =========================================================
# STEP 1: Imports & setup
# =========================================================
import asyncio
import nest_asyncio
import pandas as pd
import json
import os
import re
import io
from urllib.parse import urlparse, urljoin
from tqdm.notebook import tqdm
from playwright.async_api import async_playwright
from google.colab import files
import spacy

nest_asyncio.apply()

# =========================================================
# STEP 2: Config
# =========================================================
DEBUG = True                   # set False later for production
USE_CSV = True
RESET_CHECKPOINTS = False      # keep state for resume
STOP_ON_SENIOR = True
RUN_LIMIT = 10                 # pilot: first 10 only; set None to run all
MAX_WORKERS = 3 if DEBUG else 5
MAX_LINKS_PER_PAGE = 10
MAX_PAGES_PER_DOMAIN = 50

DEFAULT_ALLOWED_PATHS = [
    "/contact", "/about", "/leadership", "/management",
    "/executive", "/board", "/team", "/staff",
    "/who-we-are", "/our-team", "/meet", "/people",
    "/directory", "/governance"
]

CHECKPOINT_DIR = "/content/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
VISITED_FILE    = os.path.join(CHECKPOINT_DIR, "visited_pages.json")
FRONTIER_FILE   = os.path.join(CHECKPOINT_DIR, "frontier.json")
STATE_FILE      = os.path.join(CHECKPOINT_DIR, "crawler_state.json")

# Single combined output file:
RESULTS_FILE    = "/content/leadershipemailresults.csv"

# In-memory stores
found_emails = set()
email_records = []       # dicts: credit_union, email, domain, source_url, found_via
leadership_entries = []  # dicts: credit_union, role, name, email, source_url, found_via
visited_pages = set()
DOMAIN_VISITS = {}
FOUND_SENIOR_BY_DOMAIN = set()
DOMAIN_SENIOR_COUNTS = {}

# =========================================================
# STEP 3: URL normalization / validation
# =========================================================
COMMON_TLD_FIXES = {
    ".or": ".org", ".coo": ".coop", ".coom": ".com",
    ".con": ".com", ".og": ".org", ".cm": ".com"
}
BAD_HOSTS = {"facebook.com", "www.facebook.com", "wixsite.com", "linktr.ee"}

def fix_tld(host: str) -> str:
    h = (host or "").strip().lower()
    for bad, good in COMMON_TLD_FIXES.items():
        if h.endswith(bad):
            return h[:-len(bad)] + good
    return h

def normalize_url(raw: str) -> str:
    u = (raw or "").strip().strip('"').strip("'")
    if not u:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", u):
        u = "https://" + u.lstrip("/")
    p = urlparse(u)
    host = fix_tld(p.netloc)
    if any(b in host for b in BAD_HOSTS):
        return ""
    return p._replace(netloc=host).geturl()

def is_valid_http_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False

def is_credit_union_domain(domain: str) -> bool:
    d = (domain or "").lower()
    return d.endswith(".coop") or "creditunion" in d or d.endswith(".org") or d.endswith(".cu")

def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    no_frag = parsed._replace(fragment="")
    s = no_frag.geturl()
    return s[:-1] if s.endswith("/") else s

# =========================================================
# STEP 4: Resume state helpers (frontier, visited, domain)
# =========================================================
FRONTIER_PENDING = set()

def save_state():
    try:
        with open(VISITED_FILE, "w") as f:
            json.dump(sorted(list(visited_pages)), f)
    except Exception:
        pass
    try:
        with open(FRONTIER_FILE, "w") as f:
            json.dump(sorted(list(FRONTIER_PENDING)), f)
    except Exception:
        pass
    try:
        blob = {
            "DOMAIN_VISITS": DOMAIN_VISITS,
            "FOUND_SENIOR_BY_DOMAIN": sorted(list(FOUND_SENIOR_BY_DOMAIN)),
            "DOMAIN_SENIOR_COUNTS": DOMAIN_SENIOR_COUNTS
        }
        with open(STATE_FILE, "w") as f:
            json.dump(blob, f)
    except Exception:
        pass

def load_state():
    if RESET_CHECKPOINTS:
        return
    # visited
    try:
        with open(VISITED_FILE, "r") as f:
            vv = json.load(f)
            for u in vv:
                visited_pages.add(canonical_url(u))
        print(f"[RESUME] visited pages loaded: {len(visited_pages)}")
    except Exception:
        pass
    # domain state
    try:
        with open(STATE_FILE, "r") as f:
            blob = json.load(f)
            DOMAIN_VISITS.update(blob.get("DOMAIN_VISITS", {}))
            for h in blob.get("FOUND_SENIOR_BY_DOMAIN", []):
                FOUND_SENIOR_BY_DOMAIN.add(h)
            DOMAIN_SENIOR_COUNTS.update(blob.get("DOMAIN_SENIOR_COUNTS", {}))
        print(f"[RESUME] domains stop={len(FOUND_SENIOR_BY_DOMAIN)} counters={len(DOMAIN_VISITS)}")
    except Exception:
        pass
    # frontier
    try:
        with open(FRONTIER_FILE, "r") as f:
            pts = json.load(f) or []
            for t in pts:
                try:
                    url, depth = t if isinstance(t, list) else (t[0], t[1])
                    FRONTIER_PENDING.add((url, depth))
                except Exception:
                    continue
        print(f"[RESUME] frontier pending={len(FRONTIER_PENDING)}")
    except Exception:
        pass
    # combined results (rebuild memory)
    try:
        if os.path.exists(RESULTS_FILE):
            df_all = pd.read_csv(RESULTS_FILE)
            for _, r in df_all.iterrows():
                cu = r.get("credit_union","")
                role = r.get("role","")
                name = r.get("name","")
                email = (r.get("email","") or "").strip()
                domain = r.get("domain","")
                src = r.get("source_url","")
                via = r.get("found_via","")
                # rebuild leadership entries
                leadership_entries.append({
                    "credit_union": cu, "role": role, "name": name,
                    "email": email, "source_url": src, "found_via": via
                })
                # rebuild email set & email_records
                if email:
                    found_emails.add(email)
                    email_records.append({
                        "credit_union": cu, "email": email, "domain": domain,
                        "source_url": src, "found_via": via
                    })
        print(f"[RESUME] combined rows loaded={len(leadership_entries)}")
    except Exception:
        pass

# =========================================================
# STEP 5: Email & leadership helpers
# =========================================================
PLACEHOLDER_EMAILS = {"user@example.com", "[email protected]"}
CATCH_ALL_LOCALPARTS = {"info", "contact", "support", "customerservice", "hello"}

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

SENIOR_TITLES = [
    "Chief Executive Officer", "CEO", "President",
    "Chief Financial Officer", "CFO",
    "Chief Operating Officer", "COO",
    "Chief Information Officer", "CIO",
    "Chief Technology Officer", "CTO",
    "Chief Risk Officer", "CRO",
    "Chief Lending Officer", "CLO",
    "Chief Marketing Officer", "CMO",
    "Executive Vice President", "EVP",
    "Senior Vice President", "SVP",
    "Vice President", "VP",
    "Board Chair", "Chair"
]

# Remove non-target roles & service labels
ROLE_DENY = {
    "bill payer", "online banking", "mobile deposit", "digital banking",
    "branch manager", "loan officer"  # explicitly excluded
}

def looks_like_service_role(role: str) -> bool:
    r = (role or "").lower()
    return any(s in r for s in ROLE_DENY)

def deobfuscate_email(s: str) -> str:
    if not s: return ""
    t = s
    t = re.sub(r"\s*\[at\]\s*|\s+at\s+", "@", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\[dot\]\s*|\s+dot\s+", ".", t, flags=re.IGNORECASE)
    t = t.replace(" (at) ", "@").replace(" (dot) ", ".")
    return t

def email_domain(email: str) -> str:
    try:
        return email.split("@", 1)[1].lower()
    except Exception:
        return ""

def is_catch_all(email: str) -> bool:
    try:
        local = email.split("@", 1)[0].lower()
        return local in CATCH_ALL_LOCALPARTS
    except Exception:
        return False

def keep_email(email: str, source_host: str, valid_domains: set) -> bool:
    if not email or email in PLACEHOLDER_EMAILS:
        return False
    d = email_domain(email)
    if d == (source_host or "").lower():
        return True
    if valid_domains and d in valid_domains:
        return True
    if is_credit_union_domain(d):
        return True
    return False

def email_matches_name(email: str, name: str) -> bool:
    try:
        local = email.split("@", 1)[0].lower()
        parts = [p.lower() for p in re.findall(r"[A-Za-z]+", name)]
        return any(p in local for p in parts if len(p) >= 3)
    except Exception:
        return False

def choose_best_email(candidates: list, name: str) -> str:
    if not candidates: return ""
    norm = []
    for e in candidates:
        if not e: continue
        e2 = deobfuscate_email(re.sub(r"\s+", "", e))
        if EMAIL_REGEX.fullmatch(e2) and e2 not in norm:
            norm.append(e2)
    if not norm: return ""
    for e in norm:
        if name and email_matches_name(e, name):
            return e
    for e in norm:
        if not is_catch_all(e):
            return e
    return norm[0]

def add_email(email_or_href: str, page_url: str, tag: str, cfg: dict):
    if not email_or_href: return
    email = (email_or_href or "").strip()
    if email.startswith("mailto:"):
        email = email.split(":", 1)[1].split("?", 1)[0]
    email = deobfuscate_email(email)
    if EMAIL_REGEX.fullmatch(email):
        src_host = urlparse(cfg["base"]).netloc.lower()
        if keep_email(email, src_host, cfg["valid_email_domains"]):
            if email not in found_emails:
                found_emails.add(email)
            email_records.append({
                "credit_union": cfg.get("credit_union",""),
                "email": email,
                "domain": email_domain(email),
                "source_url": page_url,
                "found_via": tag
            })
            if DEBUG: print(f"[FOUND {tag}] {email} (from {page_url})")

def add_leadership(role: str, name: str, email: str, page_url: str, tag: str, cfg: dict):
    role = (role or "").strip()
    name = (name or "").strip()
    email = (email or "").strip()

    key = (role.lower(), name.lower(), email.lower(), page_url.lower(), cfg.get("credit_union","").lower())
    for row in leadership_entries:
        if (row.get("role","").lower(),
            row.get("name","").lower(),
            row.get("email","").lower(),
            row.get("source_url","").lower(),
            row.get("credit_union","").lower()) == key:
            return

    leadership_entries.append({
        "credit_union": cfg.get("credit_union",""),
        "role": role, "name": name, "email": email,
        "source_url": page_url, "found_via": tag
    })
    if email:
        add_email(email, page_url, tag, cfg)

    if STOP_ON_SENIOR and any(t.lower() in role.lower() for t in SENIOR_TITLES) and email:
        FOUND_SENIOR_BY_DOMAIN.add(urlparse(page_url).netloc)

# =========================================================
# STEP 6: spaCy setup (ROLE proximity via regex + NER)
# NOTE: Branch Manager & Loan Officer REMOVED from ROLE_REGEX
# =========================================================
nlp = spacy.load("en_core_web_sm")

ROLE_REGEX = re.compile(
    r"\b("
    r"Chief\s+Executive\s+Officer|CEO|President|"
    r"Chief\s+Financial\s+Officer|CFO|"
    r"Chief\s+Operating\s+Officer|COO|"
    r"Chief\s+Information\s+Officer|CIO|"
    r"Chief\s+Technology\s+Officer|CTO|"
    r"Chief\s+Risk\s+Officer|CRO|"
    r"Chief\s+Lending\s+Officer|CLO|"
    r"Chief\s+Marketing\s+Officer|CMO|"
    r"Executive\s+Vice\s+President|EVP|"
    r"Senior\s+Vice\s+President|SVP|"
    r"Vice\s*President(?:\s+of\s+[A-Za-z &]+)?|VP(?:\s+of\s+[A-Za-z &]+)?|"
    r"Director(?:\s+of\s+[A-Za-z &]+)?|Manager|"
    r"Board\s+Chair|Chair|Treasurer|Secretary|Board\s+Member"
    r")\b",
    flags=re.IGNORECASE
)

NAME_REGEX = re.compile(r"\b([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+)+)\b")

def extract_leadership_from_text_spacy(html: str, page_url: str, cfg: dict):
    txt = re.sub(r"<[^>]+>", " ", html or "")
    if not txt.strip():
        return
    doc = nlp(txt)
    person_spans = [(ent.start_char, ent.end_char, ent.text) for ent in doc.ents if ent.label_ == "PERSON"]
    role_hits = [(m.start(), m.group(1)) for m in ROLE_REGEX.finditer(txt)]

    def nearest_person(pos:int, max_win:int=300):
        best = ("", 10**9)
        for s, e, name in person_spans:
            d = min(abs(s - pos), abs(e - pos))
            if d < best[1] and d <= max_win:
                best = (name, d)
        return best[0]

    for pos, role in role_hits:
        if looks_like_service_role(role):
            continue
        name = nearest_person(pos, max_win=300)
        window = txt[max(0, pos - 300): pos + 300]
        candidates = [m.group(0) for m in EMAIL_REGEX.finditer(window)]
        email = choose_best_email(candidates, name)
        add_leadership(role, name, email, page_url, "text-spacy", cfg)

# =========================================================
# SITE-SPECIFIC EXTRACTORS (Morris Sheppard FCU)
# =========================================================
async def extract_site_msfcu(page, page_url: str, cfg: dict):
    """
    Morris Sheppard FCU (morrissheppardfcu.org):
    Capture CEO/leadership with email even if split/spanned/obfuscated.
    """
    txt = ""
    try:
        txt = await page.evaluate("document.body.innerText")
    except Exception:
        try:
            txt = await page.inner_text("body")
        except Exception:
            txt = ""
    txt = deobfuscate_email(txt or "")

    trio_pat = re.compile(
        r"(?is)\b("
        r"Chief\s+Executive\s+Officer|CEO|President"
        r")\b\s*"
        r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+)+)\s*"
        r"([A-Za-z0-9._%+\-\s]+@\s*[A-Za-z0-9.\-]+\s*\.\s*[A-Za-z]{2,})"
    )

    matched_any = False
    for m in trio_pat.finditer(txt):
        role, name, raw_email = m.group(1), m.group(2), m.group(3)
        email = deobfuscate_email(re.sub(r"\s+", "", raw_email))
        if not EMAIL_REGEX.fullmatch(email):
            start = max(0, m.start() - 600); end = min(len(txt), m.end() + 600)
            window = txt[start:end]
            candidates = [mm.group(0) for mm in EMAIL_REGEX.finditer(window)]
            candidates.append(raw_email)
            email = choose_best_email(candidates, name)
        if DEBUG:
            print(f"[MSFCU STRICT] {role} · {name} · {email}")
        add_leadership(role, name, email, page_url, "msfcu-strict", cfg)
        matched_any = True

    if not matched_any and txt:
        doc = nlp(txt)
        person_spans = [(ent.start_char, ent.end_char, ent.text) for ent in doc.ents if ent.label_ == "PERSON"]
        role_hits = [(m.start(), m.group(1)) for m in ROLE_REGEX.finditer(txt)]

        def nearest_person(pos:int, max_win:int=400):
            best = ("", 10**9)
            for s, e, name in person_spans:
                d = min(abs(s - pos), abs(e - pos))
                if d < best[1] and d <= max_win:
                    best = (name, d)
            return best[0]

        for pos, role in role_hits:
            if looks_like_service_role(role):
                continue
            name = nearest_person(pos, max_win=400)
            window = txt[max(0, pos - 600): pos + 600]
            candidates = [mm.group(0) for mm in EMAIL_REGEX.finditer(window)]
            email = choose_best_email(candidates, name)
            if DEBUG:
                print(f"[MSFCU SPACY] role={role} name={name} email={email}")
            add_leadership(role, name, email, page_url, "msfcu-spacy", cfg)

# =========================================================
# STEP 7: Domain config & rate limiting
# =========================================================
def domain_config_for(url: str, cu_name: str = ""):
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}/"
    return {
        "base": base,
        "allowed_paths": DEFAULT_ALLOWED_PATHS[:],
        "valid_email_domains": {parsed.netloc.lower()},
        "start_urls": [url],
        "name": parsed.netloc.lower(),
        "credit_union": (cu_name or "").strip(),  # carry CU name
    }

async def rate_limit(base: str):
    await asyncio.sleep(0.3 if DEBUG else 0.6)

# =========================================================
# STEP 8: Crawl page (Playwright)
# =========================================================
def log_queue_state(q):
    print(f"[QUEUE] visited={len(visited_pages)} pending={q.qsize()}")

async def crawl_page(context, url: str, depth: int, cfg: dict):
    host = urlparse(cfg["base"]).netloc
    if STOP_ON_SENIOR and host in FOUND_SENIOR_BY_DOMAIN:
        if DEBUG: print(f"[DONE] Senior leadership captured for {host} — skipping {url}")
        return []

    if DEBUG: print(f"[NAVIGATE] depth={depth} → {url}")
    if url in visited_pages or depth > 4:
        return []

    DOMAIN_VISITS[host] = DOMAIN_VISITS.get(host, 0) + 1
    if DOMAIN_VISITS[host] > MAX_PAGES_PER_DOMAIN:
        if DEBUG: print(f"[CAP] {host} reached {MAX_PAGES_PER_DOMAIN} pages — skipping {url}")
        return []

    visited_pages.add(url)
    await rate_limit(cfg["base"])

    page = await context.new_page()
    page.set_default_timeout(30000)
    page.set_default_navigation_timeout(45000)

    BLOCK_PATTERNS = [
        r".*\.(png|jpe?g|gif|webp|svg|ico)(\?.*)?$",
        r".*\.(woff2?|ttf|otf)(\?.*)?$",
        r".*/(gtm|googletagmanager)\.com/.*",
        r".*/google-analytics\.com/.*",
        r".*doubleclick\.net/.*",
        r".*/facebook\.com/tr/.*",
        r".*/hotjar\.com/.*",
    ]
    async def route_block(route):
        try:
            req_url = route.request.url
            for pat in BLOCK_PATTERNS:
                if re.match(pat, req_url, flags=re.IGNORECASE):
                    return await route.abort()
            return await route.continue_()
        except Exception:
            return await route.continue_()
    await page.route("**/*", route_block)

    async def on_response(response):
        try:
            if urlparse(cfg["base"]).netloc != urlparse(response.url).netloc:
                return
            ct = (response.headers.get("content-type", "") or "").lower()
            if not any(ct.startswith(t) for t in ("text/html","application/json","text/plain","application/xhtml+xml","application/xml")):
                return
            text = await response.text()
            text = deobfuscate_email(text)
            for e in EMAIL_REGEX.findall(text or ""):
                add_email(e, response.url, "network", cfg)
        except Exception:
            pass
    page.on("response", on_response)

    new_links = []
    try:
        try:
            await page.goto(url, wait_until="domcontentloaded")
        except Exception:
            alt = url
            if url.startswith("https://"):
                alt = "http://" + url[len("https://"):]
            elif url.startswith("http://"):
                alt = "https://" + url[len("http://"):]
            if DEBUG: print(f"[RETRY] {url} → {alt}")
            await page.goto(alt, wait_until="domcontentloaded")
            url = alt

        await page.wait_for_selector("a[href], body", timeout=5000)
        html = await page.content()

        html_deob = deobfuscate_email(html)
        for e in EMAIL_REGEX.findall(html_deob or ""):
            add_email(e, url, "html", cfg)

        mailto_loc = page.locator("a[href^='mailto:']")
        count = await mailto_loc.count()
        for i in range(count):
            href = await mailto_loc.nth(i).get_attribute("href")
            raw_email = ""
            if href:
                raw_email = href.split(":", 1)[1].split("?", 1)[0]
            context_text = ""
            cur = mailto_loc.nth(i)
            for _ in range(3):
                try:
                    cur = cur.locator("xpath=..")
                    context_text += "\n" + (await cur.inner_text())
                except Exception:
                    break
            lines = [l.strip() for l in context_text.splitlines() if l.strip()]
            role, name = "", ""
            for j, line in enumerate(lines):
                rm = ROLE_REGEX.search(line)
                if rm:
                    role = rm.group(1)
                    if looks_like_service_role(role):
                        role = ""
                        break
                for delta in [0,1,-1,2,-2,3,-3,4,-4,5,-5]:
                    li = j + delta
                    if 0 <= li < len(lines):
                        nm = NAME_REGEX.search(lines[li])
                        if nm:
                            name = nm.group(1); break
            candidates = []
            for line in lines:
                ld = deobfuscate_email(line)
                candidates += [m.group(0) for m in EMAIL_REGEX.finditer(ld)]
            if raw_email:
                candidates.append(raw_email)
            email = choose_best_email(candidates, name)
            add_leadership(role, name, email, url, "mailto-context", cfg)

        extract_leadership_from_text_spacy(html, url, cfg)

        # Site-specific hook: MSFCU contact/staff pages
        host_lower = urlparse(cfg["base"]).netloc.lower()
        path_lower = (urlparse(url).path or "/").lower().rstrip("/")
        if "morrissheppardfcu.org" in host_lower and (
            path_lower.endswith("/contact") or "/contact" in path_lower or "/staff" in path_lower
        ):
            await extract_site_msfcu(page, url, cfg)

        anchors = await page.query_selector_all("a[href]")
        candidates = []
        for a in anchors:
            href = await a.get_attribute("href")
            if not href: continue
            abs_url = canonical_url(urljoin(url, href))
            if abs_url.lower().endswith(".pdf"):
                continue
            if urlparse(abs_url).netloc != urlparse(cfg["base"]).netloc:
                continue
            if abs_url in visited_pages:
                continue
            if not any((urlparse(abs_url).path or "/").startswith(p) for p in cfg["allowed_paths"]):
                continue
            candidates.append(abs_url)

        leadership_first = [u for u in candidates if any(k in u.lower() for k in [
            "/leadership","/management","/executive","/board","/team","/staff",
            "/about","/contact","/who-we-are","/our-team","/meet","/people","/directory","/governance"
        ])]
        others = [u for u in candidates if u not in leadership_first]
        quota = max(0, MAX_LINKS_PER_PAGE - len(leadership_first))
        new_links = leadership_first + others[:quota]

    except Exception as e:
        if DEBUG: print(f"[FAIL] {url} — {e}")
    finally:
        await page.close()

    # ===== Single combined CSV save =====
    # leadership_entries: credit_union, role, name, email, source_url, found_via
    # email_records:      credit_union, email, domain, source_url, found_via
    df_lead = pd.DataFrame(leadership_entries)
    df_em   = pd.DataFrame(email_records)

    # Ensure aligned columns for combined export
    if not df_lead.empty:
        if "domain" not in df_lead.columns:
            df_lead["domain"] = ""        # leadership rows may not have domain separately
    if not df_em.empty:
        for col in ("role","name"):
            if col not in df_em.columns:
                df_em[col] = ""           # email-only rows do not carry role/name

    # Merge & drop duplicates
    df_all = pd.concat([df_lead, df_em], ignore_index=True)
    if not df_all.empty:
        df_all = df_all[["credit_union","role","name","email","domain","source_url","found_via"]]
        df_all = df_all.drop_duplicates(subset=["credit_union","role","name","email","source_url","found_via"])
        df_all.to_csv(RESULTS_FILE, index=False)

    # Maintain visited pages checkpoint
    with open(VISITED_FILE, "w") as f:
        json.dump(sorted(list(visited_pages)), f)

    return new_links

# =========================================================
# STEP 9: Runner (queue + resume)
# =========================================================
async def run(start_items):
    if RESET_CHECKPOINTS:
        # wipe in-memory and on-disk state (except RESULTS_FILE so user can inspect)
        for p in (VISITED_FILE, FRONTIER_FILE, STATE_FILE):
            try:
                os.remove(p)
            except Exception:
                pass
        visited_pages.clear()
        FRONTIER_PENDING.clear()
        DOMAIN_VISITS.clear()
        FOUND_SENIOR_BY_DOMAIN.clear()
        DOMAIN_SENIOR_COUNTS.clear()
        leadership_entries.clear()
        email_records.clear()
        found_emails.clear()

    load_state()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = await browser.new_context(
            user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/122 Safari/537.36")
        )

        q = asyncio.Queue()

        if FRONTIER_PENDING:
            for (u, d) in sorted(list(FRONTIER_PENDING)):
                cfg = domain_config_for(u, cu_name="")  # unknown after resume; still safe
                await q.put((u, d, cfg))
        else:
            for item in start_items:
                cu_name = item.get("credit_union","")
                url     = item.get("url","")
                cfg     = domain_config_for(url, cu_name=cu_name)
                for u in cfg["start_urls"]:
                    if u not in visited_pages:
                        await q.put((u, 0, cfg))
                        add_to_frontier(u, 0)

        pbar = tqdm(total=len(visited_pages) + q.qsize(), desc="Crawling", unit="page")
        log_queue_state(q)

        async def worker():
            while True:
                try:
                    url, depth, cfg = await asyncio.wait_for(q.get(), timeout=0.75)
                except asyncio.TimeoutError:
                    if q.empty(): break
                    continue

                remove_from_frontier(url, depth)
                links = await crawl_page(context, url, depth, cfg)
                for l in links:
                    await q.put((l, depth + 1, cfg))
                    add_to_frontier(l, depth + 1)

                pbar.total = len(visited_pages) + q.qsize()
                pbar.update(1); pbar.refresh()
                log_queue_state(q)
                save_state()
                q.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(MAX_WORKERS)]
        await asyncio.gather(*tasks)

        await context.close(); await browser.close(); pbar.close()
        save_state()

# =========================================================
# STEP 10: CSV upload (A = credit union name, B = URL)
# =========================================================
print("📥 Upload your CSV (column A = credit union name, column B = URL; or a 'url' column).")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No CSV uploaded.")
csv_name = list(uploaded.keys())[0]

try:
    df = pd.read_csv(io.BytesIO(uploaded[csv_name]))
except Exception:
    df = pd.read_csv(io.BytesIO(uploaded[csv_name]), encoding="latin1")

pairs = []  # list of (credit_union_name, normalized_url)

if "url" in df.columns:
    url_col = df["url"].astype(str).tolist()
    name_col = (df.iloc[:, 0].astype(str).tolist() if df.shape[1] >= 2 else [""] * len(url_col))
    for cu_name, raw in zip(name_col, url_col):
        u = normalize_url(raw)
        if not u or not is_valid_http_url(u):
            continue
        host = urlparse(u).netloc
        if not is_credit_union_domain(host):
            continue
        pairs.append((cu_name.strip(), u))
else:
    name_col = df.iloc[:, 0].astype(str).tolist()
    url_col  = df.iloc[:, 1].astype(str).tolist()
    for cu_name, raw in zip(name_col, url_col):
        u = normalize_url(raw)
        if not u or not is_valid_http_url(u):
            continue
        host = urlparse(u).netloc
        if not is_credit_union_domain(host):
            continue
        pairs.append((cu_name.strip(), u))

seen = set()
pairs_unique = []
for cu_name, u in pairs:
    if u not in seen:
        seen.add(u)
        pairs_unique.append((cu_name, u))

pairs_seed = pairs_unique[:RUN_LIMIT] if RUN_LIMIT else pairs_unique

print(f"\n✅ Selected {len(pairs_seed)} credit unions (limit={RUN_LIMIT}):")
for i, (cu_name, u) in enumerate(pairs_seed, 1):
    print(f"{i}. {cu_name}  →  {u}")

# =========================================================
# STEP 11: Execute crawl
# =========================================================
await run([{"credit_union": cu_name, "url": u} for cu_name, u in pairs_seed])

# =========================================================
# STEP 12: Print results & point to single CSV
# =========================================================
print("\n===================")
print("🎉 DONE — EMAILS FOUND (deduped):")
print("===================")
personal_emails = sorted([e for e in found_emails if not is_catch_all(e)]) or sorted(list(found_emails))
for e in personal_emails:
    print(e)

if leadership_entries:
    print("\n===================")
    print("👥 LEADERSHIP — Role · Name · Email (source):")
    print("===================")
    priority = {"chief executive officer":0, "ceo":0, "president":1, "cfo":2, "coo":3, "vp":4}
    def role_priority(role):
        r = (role or "").lower()
        for k, v in priority.items():
            if k in r: return v
        return 9
    leadership_entries.sort(key=lambda r: role_priority(r.get("role","")))
    for row in leadership_entries[:100]:
        print(f"{row.get('role','')} · {row.get('name','')} · {row.get('email','')} (source: {row.get('source_url','')})")

print(f"\n✅ Combined CSV saved to: {RESULTS_FILE}")

"""
Normalization utilities for business_name and business_address fields.

Handles:
- URL domain extraction (e.g. maurewilliamscolombier.com -> maure williams colombier)
- Transliteration (Devanagari / Tamil / Kannada / Bengali / Accents -> ASCII)
- Multilingual legal suffix normalization (US, India, France)
- Address term expansions (US, India, France)
- Number and postal code extraction
"""
import re
import unicodedata
try:
    from unidecode import unidecode
except ImportError:
    def unidecode(value):
        return "".join(
            ch for ch in unicodedata.normalize("NFKD", str(value))
            if not unicodedata.combining(ch)
        )

# --- Legal-suffix / abbreviation normalization (business names) ---
NAME_ABBREV_MAP = {
    "pvt": "private",
    "pvtltd": "private limited",
    "ltd": "limited",
    "co": "company",
    "corp": "corporation",
    "inc": "incorporated",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "sarl": "sarl",
    "sas": "sas",
    "sa": "sa",
    "sci": "sci",
    "snc": "snc",
    "eurl": "eurl",
    "gie": "gie",
    "dba": "dba",
    "assoc": "associates",
    "assn": "association",
    "intl": "international",
    "mfg": "manufacturing",
    "engg": "engineering",
    "svcs": "services",
    "svc": "service",
    "grp": "group",
    "tech": "technologies",
    "ent": "enterprises",
    "enterprises": "enterprise",
    "solutions": "solution",
    "services": "service",
    "traders": "trader",
    "industries": "industry",
    "&": "and",
}

# Legal suffixes to optionally strip for core name matching
LEGAL_SUFFIXES = {
    "private", "limited", "company", "corporation", "incorporated",
    "llc", "llp", "plc", "sarl", "sas", "sa", "sci", "snc", "eurl", "gie",
    "inc", "corp", "ltd", "pvt", "co"
}

ADDRESS_ABBREV_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "apt": "apartment",
    "ste": "suite",
    "no": "number",
    "flr": "floor",
    "bldg": "building",
    "twp": "township",
    "hsg": "housing",
    "soc": "society",
    "co-op": "cooperative",
    "coop": "cooperative",
    "opp": "opposite",
    "nr": "near",
    "dist": "district",
    "col": "colony",
    "sec": "sector",
    "r": "rue",
    "rte": "route",
    "pl": "place",
    "all": "allee",
    "imp": "impasse",
    "ch": "chemin",
}

ADDRESS_FILLER_WORDS = {"near", "opposite", "behind", "beside", "next", "to", "opp", "nr"}
NULLISH_TOKENS = {"null", "none", "na", "n.a", "nil", "#", "--", "-", "nan", "undefined"}

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|in|org|net|co|fr|io|biz|info|us)\b)", re.IGNORECASE)
DOMAIN_CLEAN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+)\.(?:com|in|org|net|co|fr|io|biz|info|us).*$", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
POSTAL_CODE_RE = re.compile(r"\b\d{5,6}\b")
DIGITS_RE = re.compile(r"\b\d+\b")


def _transliterate(text: str) -> str:
    return unidecode(text)


def _clean_url_to_words(text: str) -> str:
    """If text contains or is a domain/URL, extract the core domain text."""
    if "." not in text and "/" not in text:
        return text
    def _replace_url(match):
        raw = match.group(0)
        m = DOMAIN_CLEAN_RE.match(raw)
        if m:
            domain = m.group(1).replace("-", " ")
            return f" {domain} "
        return " "
    return URL_RE.sub(_replace_url, text)


def _basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    if not text or text.lower() in NULLISH_TOKENS:
        return ""
    if "." in text or "/" in text or "@" in text:
        text = _clean_url_to_words(text)
    if not text.isascii():
        text = _transliterate(text)
    text = text.lower()
    if "&" in text:
        text = text.replace("&", " and ")
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _expand_tokens(tokens, abbrev_map):
    out = []
    for tok in tokens:
        if tok in NULLISH_TOKENS:
            continue
        out.append(abbrev_map.get(tok, tok))
    return out


def normalize_name(raw: str):
    """Return (normalized_string, token_list, core_name_str)."""
    cleaned = _basic_clean(raw)
    tokens = cleaned.split()
    tokens = _expand_tokens(tokens, NAME_ABBREV_MAP)
    norm_str = " ".join(tokens)
    core_tokens_list = [t for t in tokens if t not in LEGAL_SUFFIXES]
    core_name_str = " ".join(core_tokens_list) if core_tokens_list else norm_str
    return norm_str, tokens, core_name_str


def normalize_address(raw: str):
    """Return (normalized_string, token_list, postal_code_or_None, digits_set)."""
    postal = None
    raw_str = str(raw) if raw is not None else ""
    if raw_str and raw_str.lower() not in NULLISH_TOKENS:
        m = POSTAL_CODE_RE.findall(raw_str)
        if m:
            postal = m[-1]
    cleaned = _basic_clean(raw)
    tokens = cleaned.split()
    tokens = _expand_tokens(tokens, ADDRESS_ABBREV_MAP)
    tokens = [t for t in tokens if t not in ADDRESS_FILLER_WORDS]
    digits = set(DIGITS_RE.findall(cleaned)) if cleaned else set()
    return " ".join(tokens), tokens, postal, digits


def core_tokens(tokens, min_len=3):
    """Drop very short tokens (min_len) for use in blocking/Jaccard."""
    return {t for t in tokens if len(t) >= min_len}



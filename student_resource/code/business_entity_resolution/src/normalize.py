"""
Normalization utilities for business_name and business_address fields.

The dataset mixes Latin script with Devanagari / Tamil / Kannada / Bengali
transliterations of the same names, plus heavy abbreviation noise (Pvt/Private,
Ltd/Limited, Rd/Road...), stray URLs embedded in name fields, and landmark-style
address noise ("Near SBI ATM"). Everything here is deterministic, rule-based,
and uses only the record's own fields -- no external lookups.
"""
import re
import unicodedata
try:
    from unidecode import unidecode
except ImportError:  # keeps lightweight validation tools usable without extras
    def unidecode(value):
        return "".join(
            ch for ch in unicodedata.normalize("NFKD", str(value))
            if not unicodedata.combining(ch)
        )

# --- Legal-suffix / abbreviation normalization (business names) ---
# Applied as whole-word replacements after tokenization, so we don't clobber
# substrings inside longer words.
NAME_ABBREV_MAP = {
    "pvt": "private",
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
    "dba": "dba",
    "assoc": "associates",
    "assn": "association",
    "intl": "international",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "service",
    "grp": "group",
    "&": "and",
}

ADDRESS_ABBREV_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "apt": "apartment",
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
}

# Landmark / filler phrases that add noise without identifying the address.
ADDRESS_FILLER_WORDS = {"near", "opposite", "behind", "beside", "next", "to"}

# Tokens that are pure noise / missing-value markers seen in the data.
NULLISH_TOKENS = {"null", "none", "na", "n.a", "nil", "#", "--", "-"}

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:com|in|org|net|co)\b)", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
POSTAL_CODE_RE = re.compile(r"\b\d{5,6}\b")


def _transliterate(text: str) -> str:
    """Best-effort ASCII transliteration (handles Devanagari/Tamil/etc.).

    unidecode is a rule-based, offline transcription table -- not a lookup
    service -- so this stays within the "no external data" constraint.
    """
    return unidecode(text)


def _strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def _basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = _strip_urls(text)
    text = _transliterate(text)
    text = text.lower()
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
    """Return (normalized_string, token_list) for a business_name field."""
    cleaned = _basic_clean(raw)
    tokens = cleaned.split()
    tokens = _expand_tokens(tokens, NAME_ABBREV_MAP)
    return " ".join(tokens), tokens


def normalize_address(raw: str):
    """Return (normalized_string, token_list, postal_code_or_None)."""
    postal = None
    if raw:
        m = POSTAL_CODE_RE.findall(str(raw))
        if m:
            postal = m[-1]  # postal/PIN codes tend to trail the address
    cleaned = _basic_clean(raw)
    tokens = cleaned.split()
    tokens = _expand_tokens(tokens, ADDRESS_ABBREV_MAP)
    tokens = [t for t in tokens if t not in ADDRESS_FILLER_WORDS]
    return " ".join(tokens), tokens, postal


def core_tokens(tokens, min_len=3):
    """Drop very short tokens (min_len) for use in blocking/Jaccard, where
    1-2 char tokens are mostly noise (unit numbers, initials)."""
    return {t for t in tokens if len(t) >= min_len}

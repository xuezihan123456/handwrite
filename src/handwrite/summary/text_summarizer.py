"""Rule-based text summarizer for Chinese and English mixed content.

Extracts title, key sentences, bullet points, and keywords using regex
and heuristic rules -- no ML models required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SummaryResult:
    """Structured summary extracted from raw text."""

    title: str
    key_sentences: list[str]
    bullet_points: list[str]
    keywords: list[str]
    sections: list[SummarySection]


@dataclass(frozen=True)
class SummarySection:
    """A section within the summary with heading and items."""

    heading: str
    items: list[str]


# Common stop words for Chinese and English
_ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "if",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "about",
        "up",
        "down",
        "here",
        "there",
    }
)

_CHINESE_STOP_WORDS: frozenset[str] = frozenset(
    {
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "他",
        "她",
        "它",
        "们",
        "那",
        "些",
        "什么",
        "怎么",
        "如何",
        "为什么",
        "可以",
        "可能",
        "已经",
        "因为",
        "所以",
        "但是",
        "而且",
        "或者",
        "如果",
        "虽然",
        "不过",
        "还是",
        "以及",
        "然后",
        "因此",
        "其实",
        "只是",
        "这个",
        "那个",
        "这些",
        "那些",
        "之",
        "与",
        "及",
        "等",
        "被",
        "把",
        "对",
        "从",
        "向",
        "让",
        "给",
        "又",
        "再",
        "才",
        "已",
        "将",
        "并",
        "而",
        "但",
        "或",
        "所",
        "以",
        "为",
        "中",
        "里",
        "下",
        "来",
        "去",
        "过",
        "地",
        "得",
        "能",
        "会",
        "该",
        "应该",
        "需要",
    }
)

# Patterns that indicate important sentences
_IMPORTANT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^[\s]*[#]+\s+"),  # Markdown headings
    re.compile(r"^[一二三四五六七八九十]+[、.．]"),  # Chinese numbered items
    re.compile(r"^[（(]\s*[一二三四五六七八九十]+\s*[)）]"),  # Chinese parenthesized numbers
    re.compile(r"^\d+[.、)）]\s*"),  # Numbered items
    re.compile(r"^[\s]*[-*•]\s+"),  # Bullet points
    re.compile(r"^[A-Z][A-Z\s]+:"),  # ALL-CAPS labels like "NOTE:", "TODO:"
    re.compile(r"^(总之|综上|总结|结论|关键|核心|重点|要点|注意|摘要|概要)[：:,.]?"),
    re.compile(r"^(In conclusion|Summary|Key|Important|Note|Result|Therefore|Thus)[,:]"),
]

# Section heading patterns
_HEADING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^[\s]*#{1,3}\s+(.+)"),  # Markdown headings
    re.compile(r"^(.{2,20})[：:]\s*$"),  # Chinese label followed by colon on its own line
    re.compile(r"^([A-Z][\w\s]{2,30})[:]\s*$"),  # English heading with colon
    re.compile(
        r"^[一二三四五六七八九十]+[、.．]\s*(.{2,40})$"
    ),  # Chinese numbered heading
    re.compile(r"^\d+[.、]\s*(.{2,60})$"),  # Numbered heading
]

# Sentence splitting patterns
_SENTENCE_SPLIT_ZH: re.Pattern[str] = re.compile(r"[。！？!?\n]+")
_SENTENCE_SPLIT_EN: re.Pattern[str] = re.compile(r"[.!?]+\s+|\n+")

# Keyword extraction: match Chinese words (2-4 chars) and English words (3+ chars)
_WORD_PATTERN_ZH: re.Pattern[str] = re.compile(r"[\u4e00-\u9fff]{2,4}")
_WORD_PATTERN_EN: re.Pattern[str] = re.compile(r"[a-zA-Z]{3,}")


def extract_summary(
    text: str,
    *,
    max_key_sentences: int = 8,
    max_bullet_points: int = 12,
    max_keywords: int = 15,
) -> SummaryResult:
    """Extract a structured summary from raw text.

    Args:
        text: Input text (Chinese/English mixed).
        max_key_sentences: Maximum number of key sentences to extract.
        max_bullet_points: Maximum number of bullet points.
        max_keywords: Maximum number of keywords.

    Returns:
        SummaryResult with title, key sentences, bullet points, keywords, sections.
    """
    if not text or not text.strip():
        return SummaryResult(
            title="",
            key_sentences=[],
            bullet_points=[],
            keywords=[],
            sections=[],
        )

    normalized = _normalize_text(text)
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]

    title = _extract_title(lines, normalized)
    sections = _extract_sections(lines)
    key_sentences = _extract_key_sentences(normalized, max_key_sentences)
    bullet_points = _extract_bullet_points(lines, max_bullet_points)
    keywords = _extract_keywords(normalized, max_keywords)

    return SummaryResult(
        title=title,
        key_sentences=key_sentences,
        bullet_points=bullet_points,
        keywords=keywords,
        sections=sections,
    )


def _normalize_text(text: str) -> str:
    """Normalize whitespace while preserving structure."""
    # Collapse runs of spaces/tabs (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_title(lines: list[str], full_text: str) -> str:
    """Extract a title from the first meaningful line or markdown heading."""
    for line in lines[:10]:
        # Markdown heading
        match = re.match(r"^#{1,3}\s+(.+)", line)
        if match:
            return match.group(1).strip()

    # First non-trivial line that looks like a title
    for line in lines[:5]:
        cleaned = line.strip()
        if len(cleaned) >= 2 and len(cleaned) <= 80:
            # Skip lines that look like list items
            if re.match(r"^[-*•]\s", cleaned):
                continue
            if re.match(r"^\d+[.、]", cleaned):
                continue
            return cleaned

    # Fallback: first N characters
    if full_text:
        first_line = full_text.split("\n")[0].strip()
        return first_line[:60] if first_line else "摘要"

    return "摘要"


def _extract_sections(lines: list[str]) -> list[SummarySection]:
    """Extract sections with headings and their items."""
    sections: list[SummarySection] = []
    current_heading: str | None = None
    current_items: list[str] = []

    for line in lines:
        heading = _match_heading(line)
        if heading is not None:
            # Save previous section
            if current_heading is not None and current_items:
                sections.append(SummarySection(heading=current_heading, items=current_items))
            current_heading = heading
            current_items = []
            continue

        # Collect items under current heading
        if current_heading is not None:
            item = _clean_list_item(line)
            if item:
                current_items.append(item)

    # Save last section
    if current_heading is not None and current_items:
        sections.append(SummarySection(heading=current_heading, items=current_items))

    return sections


def _match_heading(line: str) -> str | None:
    """Check if a line looks like a section heading."""
    for pattern in _HEADING_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def _clean_list_item(line: str) -> str | None:
    """Extract the text content from a list-item line."""
    # Remove bullet markers
    match = re.match(r"^[\s]*[-*•]\s+(.+)", line)
    if match:
        return match.group(1).strip()

    # Remove numbered markers
    match = re.match(r"^[\s]*\d+[.、)）]\s*(.+)", line)
    if match:
        return match.group(1).strip()

    # Chinese numbered items
    match = re.match(r"^[一二三四五六七八九十]+[、.．]\s*(.+)", line)
    if match:
        return match.group(1).strip()

    return None


def _extract_key_sentences(text: str, max_count: int) -> list[str]:
    """Extract the most important sentences using heuristic scoring."""
    # Split on sentence boundaries
    raw_sentences = _SENTENCE_SPLIT_ZH.split(text)
    sentences: list[str] = []
    for raw in raw_sentences:
        parts = _SENTENCE_SPLIT_EN.split(raw)
        for part in parts:
            cleaned = part.strip()
            if len(cleaned) >= 4:
                sentences.append(cleaned)

    if not sentences:
        return []

    # Score each sentence
    scored: list[tuple[float, str, int]] = []
    for idx, sentence in enumerate(sentences):
        score = _score_sentence(sentence, idx, len(sentences))
        scored.append((score, sentence, idx))

    # Sort by score descending, then by original position
    scored.sort(key=lambda x: (-x[0], x[2]))

    # Take top N while preserving original order
    selected = scored[:max_count]
    selected.sort(key=lambda x: x[2])

    return [item[1] for item in selected]


def _score_sentence(sentence: str, position: int, total: int) -> float:
    """Score a sentence for importance."""
    score = 0.0

    # Position bonus: first and last sentences tend to be important
    if position == 0:
        score += 3.0
    elif position == 1:
        score += 1.5
    elif position == total - 1:
        score += 2.0

    # Length bonus: medium-length sentences are often key
    length = len(sentence)
    if 15 <= length <= 100:
        score += 1.0
    elif length < 5:
        score -= 2.0

    # Pattern matching bonus
    for pattern in _IMPORTANT_PATTERNS:
        if pattern.search(sentence):
            score += 3.0
            break

    # Keyword density bonus
    keywords_here = _extract_keywords(sentence, max_keywords=5)
    score += len(keywords_here) * 0.3

    # Chinese sentence importance markers
    importance_markers = [
        "重要",
        "关键",
        "核心",
        "主要",
        "基本",
        "根本",
        "本质",
        "重点",
        "要点",
        "总结",
        "结论",
        "因此",
        "所以",
        "总之",
        "综上",
        "表明",
        "证明",
        "显示",
        "指出",
        "认为",
        "发现",
    ]
    for marker in importance_markers:
        if marker in sentence:
            score += 1.0
            break

    return score


def _extract_bullet_points(lines: list[str], max_count: int) -> list[str]:
    """Extract bullet points and list items from the text."""
    bullets: list[str] = []

    for line in lines:
        if len(bullets) >= max_count:
            break

        item = _clean_list_item(line)
        if item and len(item) >= 2:
            bullets.append(item)

    return bullets


def _extract_keywords(text: str, max_keywords: int) -> list[str]:
    """Extract keywords by frequency and position, filtering stop words."""
    # Extract Chinese words
    zh_words = _WORD_PATTERN_ZH.findall(text)
    # Extract English words (lowercase)
    en_words = [w.lower() for w in _WORD_PATTERN_EN.findall(text)]

    # Count frequencies (excluding stop words)
    freq: dict[str, int] = {}
    for word in zh_words:
        if word not in _CHINESE_STOP_WORDS and len(word) >= 2:
            freq[word] = freq.get(word, 0) + 1

    for word in en_words:
        if word not in _ENGLISH_STOP_WORDS and len(word) >= 3:
            freq[word] = freq.get(word, 0) + 1

    # Sort by frequency descending
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])

    # Return top keywords
    return [word for word, _ in sorted_words[:max_keywords]]

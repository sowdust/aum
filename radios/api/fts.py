"""
SQLite FTS5 query helpers.

Sanitizes user input to prevent FTS injection, builds MATCH expressions,
and returns (id, rank) tuples for the calling view to filter with.
"""
import re

from django.db import connection


# Characters with special meaning in FTS5 query syntax
_FTS5_SPECIAL = re.compile(r'["\'\*\+\-\(\)\{\}\[\]\^~:\\/<>!@#$%&=|;,.]')


def sanitize_fts_query(raw_query):
    """
    Strip FTS5 special characters and build a safe MATCH expression.
    Auto-appends * to the last token for prefix matching.
    Returns None if the query is empty after sanitization.
    """
    cleaned = _FTS5_SPECIAL.sub(" ", raw_query)
    tokens = cleaned.split()
    if not tokens:
        return None
    # Quote each token to prevent any FTS5 interpretation
    # Append * to last token for prefix matching
    quoted = [f'"{t}"' for t in tokens[:-1]]
    quoted.append(f'"{tokens[-1]}"*')
    return " ".join(quoted)


def search_transcription_fts(raw_query, limit=1000):
    """
    Search radios_transcription_fts for matching segments.
    Returns list of (segment_id, rank) tuples ordered by relevance.
    """
    match_expr = sanitize_fts_query(raw_query)
    if not match_expr:
        return []

    sql = """
        SELECT segment_id, rank
        FROM radios_transcription_fts
        WHERE radios_transcription_fts MATCH %s
        ORDER BY rank
        LIMIT %s
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, [match_expr, limit])
        return cursor.fetchall()


def search_summary_fts(raw_query, summary_type=None, limit=1000):
    """
    Search radios_summary_fts for matching summaries.
    Returns list of (summary_id, summary_type, rank) tuples ordered by relevance.
    """
    match_expr = sanitize_fts_query(raw_query)
    if not match_expr:
        return []

    if summary_type and summary_type in ("chunk", "daily"):
        sql = """
            SELECT summary_id, summary_type, rank
            FROM radios_summary_fts
            WHERE radios_summary_fts MATCH %s AND summary_type = %s
            ORDER BY rank
            LIMIT %s
        """
        params = [match_expr, summary_type, limit]
    else:
        sql = """
            SELECT summary_id, summary_type, rank
            FROM radios_summary_fts
            WHERE radios_summary_fts MATCH %s
            ORDER BY rank
            LIMIT %s
        """
        params = [match_expr, limit]

    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchall()

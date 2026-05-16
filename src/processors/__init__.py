"""File-type-specific processors. Each exports `process(path) -> dict` with keys:

    {
        "kind":    "pdf"|"image"|"audio"|"video"|"text"|"office"|"archive"|"web",
        "source":  original path or URL,
        "filename": display name,
        "text":    extracted searchable text,
        "pages":   optional [{num, text}, ...] for paged sources,
        ...       extra type-specific metadata
    }
"""

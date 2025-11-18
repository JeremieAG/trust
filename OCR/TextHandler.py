import re

def clean_lines(lines) -> list[str]:
    """
    Clean the lines by removing empty lines and lines with no alphanumeric characters.
    We also combine lines that only have 1 word with the previous line.

    Args:
        lines (list[str]): List of lines to clean

    Returns:
        list[str]: Cleaned list of lines
    """
    cleaned = []
    for raw in lines:
        line = raw.strip()
        if not line or all(not c.isalnum() for c in line):
            continue
        words = re.sub(r"[^a-zA-Z0-9 ]", "", line).split()
        
        if len(words) > 1:
            cleaned.append(line)
            continue
        
        if not cleaned:
            continue
        
        cleaned[-1] += " " + line
    return cleaned

def create_batch(
    texts: list[str],
    first_n_lines: int = 3,
    last_n_lines: int = 2,
    batch_size: int = 50,
    overlap: int = 2,
    custom_start: int = 0,
    custom_end: int = None,
) -> list[str]:
    """
    Create smaller batches from a list of texts pages, preserving the first n and last n lines of each page.
    While ensuring some overlap between batches for better context.

    Args:
        texts (list[str]): List of text pages
        first_n_lines (int, optional): Number of lines to keep from the start of each page. Defaults to 3.
        last_n_lines (int, optional): Number of lines to keep from the end of each page. Defaults to 2.
        batch_size (int, optional): Number of pages per batch. Defaults to 50.
        overlap (int, optional): Number of overlapping pages between batches. Defaults to 2.
        custom_start (int, optional): Custom start page index. Defaults to 0.
        custom_end (int, optional): Custom end page index. Defaults to None.

    Returns:
        list[str]: List of text batches
    """
    batch_start = custom_start
    batches = []

    while batch_start < (len(texts) if custom_end is None else custom_end):
        batch_end = min(batch_start + batch_size, len(texts))

        batch_text = ""
        for i in range(batch_start, batch_end):
            text = texts[i]
            if text is None:
                continue
            lines = clean_lines([line for line in text.splitlines() if line.strip() != ""])
            first_lines = "\n".join(lines[:first_n_lines])
            last_lines = (
                "\n".join(lines[-last_n_lines:]) if len(lines) >= last_n_lines else ""
            )

            if last_lines and last_lines != first_lines:
                batch_text += f"Page {i+1}:\n{first_lines}\n...\n{last_lines}\n\n"
            else:
                batch_text += f"Page {i+1}:\n{first_lines}\n\n"

        if not batch_text.strip():
            batch_start += batch_size - overlap
            continue

        batches.append(batch_text.strip())
        batch_start += batch_size - overlap

    return batches

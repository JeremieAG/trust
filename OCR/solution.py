import argparse
import concurrent.futures
from difflib import SequenceMatcher
from time import time
import json
import re
import unicodedata

from API import APIClient, MODELS
from PDFProcessor import PDFProcessor
from PromptGenerator import PromptGenerator
from TextHandler import create_batch, clean_lines

CATEGORY_MAP = {
    "1": "Articles & Amendments / Statuts et Amendements",
    "2": "By Laws / Règlements",
    "3": "Unanimous Shareholder Agreement / Convention Unanime d'Actionnaires",
    "4": "Minutes & Resolutions / Procès-verbaux et Résolutions",
    "5": "Directors Register / Registre des Administrateurs",
    "6": "Officers Register / Registre des Dirigeants",
    "7": "Shareholder Register / Registre des Actionnaires",
    "8": "Securities Register / Registre des Valeurs Mobilières",
    "9": "Share Certificates / Certificats d'Actions",
    "10": "Ultimate Beneficial Owner Register / Registre des Particuliers Ayant un Contrôle Important",
}


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process PDF files with AutoComply APIClient"
    )
    parser.add_argument("pdf_file", help="Path to the PDF file to process")

    parser.add_argument(
        "--api-url",
        default="https://ai-models.autocomply.ca",
        help="APIClient base URL (default: https://ai-models.autocomply.ca)",
    )
    parser.add_argument(
        "--api-key",
        default="sk-ac-7f8e9d2c4b1a6e5f3d8c7b9a2e4f6d1c",
        help="APIClient key for authentication",
    )
    parser.add_argument(
        "--output", "-o", default="result.json", help="Output file to save results"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PDF to image conversion (default: 150)",
    )
    parser.add_argument(
        "--model",
        choices=["gpt-4o", "gemini-2.5-flash", "claude-sonnet-4.5"],
        default="gemini-2.5-flash",
        help="AI model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="The size of the individual batch sent to the API",
    )

    return parser.parse_args()


def run_batch_api(
    api: APIClient, model: MODELS, prompt: str, batchs: list[str]
) -> list[dict[str]]:
    """
    Run the APIClient on a batch of text pages.

    Args:
        api (APIClient): APIClient instance
        model (MODELS): AI model to use
        prompt (str): Processing prompt
        batchs (list[str]): List of text pages in the batch

    Returns:
        List[dict]: APIClient responses for each batch ({"batch": batch, "response": response})
    """
    results = []
    if not batchs:
        return results

    # 8 Pour ne pas surcharger l'endpoint de l'API
    max_workers = min(8, len(batchs))

    def _worker(i: int, batch: str):
        try:
            query = f"{prompt}\n\n{batch}"
            print(f"Processing batch {i+1}/{len(batchs)}")
            resp = api.ask(query, model=model)
            return {"batch": i, "response": resp}
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            return {"batch": i, "response": ""}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_worker, i, b): i for i, b in enumerate(batchs)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            res = future.result()
            results.append(res)

    results.sort(key=lambda x: x["batch"])
    return results


def parse_response(responses: list[dict]) -> list[tuple[int, int, float]]:
    """
    Parse the APIClient response into a list of tuples.

    Args:
        responses (list[dict]): APIClient responses list of dictionaries

    Returns:
        list[tuple[int, int, float]]: List of tuples containing (Page Number, Category Number, Confidence Score)
    """
    results = []
    for response in responses:
        if not response["response"]:
            print(f"Empty response for batch {response['batch']}")
            continue
        lines = response["response"].strip().splitlines()
        for line in lines:
            if line.strip() == "":
                continue
            if line.startswith("Page Number"):
                continue
            if line.startswith("`"):
                continue

            parts = line.strip().split(",")
            if len(parts) != 3:
                print(f"Invalid line format: {line}")
                continue

            try:
                page_number = int(parts[0].strip())
                category_number = int(parts[1].strip())
                confidence_score = float(parts[2].strip())
                results.append((page_number, category_number, confidence_score))
            except ValueError as e:
                print(f"Error parsing line: {line} - {e}")
                continue
    results.sort(key=lambda x: x[0])
    return results


def smooth_results(
    results: list[tuple[int, int, float]],
    confidence_threshold: int = 85,
    confidence_reduction: float = 0.9,
) -> list[tuple[int, int, float]]:
    """
    Smooth the results to ensure continuity and logical flow.
    For example, if a section is small and surrounded by another section, it may be reclassified if the confidence is low.

    Args:
        results (list[tuple[int, int, float]]): List of tuples containing (Page Number, Category Number, Confidence Score)
        confidence_threshold (int, optional): Confidence score threshold to consider reclassification. Defaults to 85.
        confidence_reduction (float, optional): Factor to reduce confidence score upon reclassification. Defaults
    Returns:
        list[tuple[int, int, float]]: Smoothed list of tuples
    """
    if not results:
        return results

    smoothed = []
    window_size = 3
    for i, (page_num, category, confidence) in enumerate(results):
        start = max(0, i - window_size // 2)
        end = min(len(results), i + window_size // 2 + 1)
        window = results[start:end]

        category_counts = {}
        for _, cat, _ in window:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        if (
            category_counts.get(category, 0) == 1
            and confidence < confidence_threshold
            and len(window) >= 2
        ):
            most_common_cat = max(category_counts.items(), key=lambda x: x[1])[0]
            if most_common_cat != category:
                smoothed.append(
                    (page_num, most_common_cat, confidence * confidence_reduction)
                )
                continue

        smoothed.append((page_num, category, confidence))

    return smoothed


def clean_category_map(category_map: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Clean the category map by removing duplicate categories (keeping biggest and highest confidence).

    Args:
        category_map (list[dict[str, str]]): List of category dictionaries
    Returns:
        list[dict[str, str]]: Cleaned list of category dictionaries
    """
    seen_categories = {}
    final_categories = []

    for category in category_map:
        name = category["name"]
        if name not in seen_categories:
            seen_categories[name] = category
            final_categories.append(category)
            continue

        existing = seen_categories[name]
        current_size = category["pageCount"]
        existing_size = existing["pageCount"]

        if current_size < existing_size:
            continue

        if (
            current_size == existing_size
            and category["avgConfidence"] <= existing["avgConfidence"]
        ):
            continue

        idx = final_categories.index(existing)
        final_categories[idx] = category
        seen_categories[name] = category

    final_categories.sort(key=lambda x: x["startPage"])
    return final_categories


def build_category_map(results: list[tuple[int, int, float]]) -> list[dict[str, str]]:
    """
    Build a category map from the results.

    Args:
        results (list[tuple[int, int, float]]): List of tuples containing (Page Number, Category Number, Confidence Score)
    Returns:
        List[dict[str, str]]: List of dictionaries containing : {"name": category_name, "startPage": start_page, "endPage": end_page, "avgConfidence": avg_confidence, "pageCount": page_count}
    """
    category_map = []
    if not results:
        return category_map

    current_category = results[0][1]
    start_page = results[0][0]
    total_confidence = 0
    page_count = 0

    for page_num, category, confidence in results:
        category_name = CATEGORY_MAP.get(str(current_category), "Unknown")
        if category_name == "Unknown":
            print(f"Unknown category {current_category} at page {page_num}")
            continue

        if category == current_category:
            total_confidence += confidence
            page_count += 1
            continue

        avg_confidence = total_confidence / page_count if page_count > 0 else 0
        category_map.append(
            {
                "name": category_name,
                "startPage": start_page,
                "endPage": page_num - 1,
                "avgConfidence": round(avg_confidence, 2),
                "pageCount": page_count,
            }
        )

        current_category = category
        start_page = page_num
        total_confidence = confidence
        page_count = 1

    avg_confidence = total_confidence / page_count if page_count > 0 else 0
    category_name = CATEGORY_MAP.get(str(current_category), "Unknown")
    if category_name != "Unknown":
        category_map.append(
            {
                "name": category_name,
                "startPage": start_page,
                "endPage": page_num,
                "avgConfidence": round(avg_confidence, 2),
                "pageCount": page_count,
            }
        )

    return clean_category_map(category_map)


def has_discontinuity(results: list[tuple[int, int, float]]) -> list[dict[str, str]]:
    """
    Check if there is any discontinuity in the results.

    Args:
        results (list[tuple[int, int, float]]): List of tuples containing (Page Number, Category Number, Confidence Score)

    Returns:
        List[dict[str, str]]: List of discontinuities found:
        ({"type": "overlap"/"gap", "index": index, "current": current_category, "next": next_category, gap_start": gap_start, "gap_end": gap_end})
    """
    discontinuities = []
    for i in range(len(results) - 1):
        current_end = results[i]["endPage"]
        next_start = results[i + 1]["startPage"]

        if next_start <= current_end:
            # Overlap or wrong order
            discontinuities.append(
                {
                    "type": "overlap",
                    "index": i,
                    "current": results[i],
                    "next": results[i + 1],
                }
            )
        elif next_start > current_end + 1:
            # Gap
            discontinuities.append(
                {
                    "type": "gap",
                    "index": i,
                    "current": results[i],
                    "next": results[i + 1],
                    "gap_start": current_end + 1,
                    "gap_end": next_start - 1,
                }
            )
    return discontinuities


def handle_discontinuity(
    api: APIClient,
    model: MODELS,
    prompt_gen: PromptGenerator,
    category_map: list[dict[str, str]],
    texts: list[str],
    max_iterations: int = 3,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Handle the discontinuity by reprocessing the affected pages.

    Args:
        api (APIClient): APIClient instance
        model (MODELS): AI model to use
        prompt_gen (PromptGenerator): Prompt generator instance
        category_map (List[dict[str, str]]): List of category dictionaries
        texts (list[str]): List of text pages
        max_iterations (int, optional): Maximum number of reprocessing iterations. Defaults to 3.

    Returns:
        tuple[list[dict[str, str]], list[dict[str, str]]]: Updated category map and list of remaining discontinuities
    """
    page_classifications = {}
    for section in category_map:
        category_num = None
        for num, name in CATEGORY_MAP.items():
            if name == section["name"]:
                category_num = int(num)
                break
        if category_num is None:
            continue

        confidence = section.get("avgConfidence", 90.0)
        for page in range(section["startPage"], section["endPage"] + 1):
            page_classifications[page] = (category_num, confidence)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        discontinuity = has_discontinuity(category_map)
        if not discontinuity:
            break

        print(
            f"\nIteration {iteration}: Found {len(discontinuity)} discontinuities, re-prompting API..."
        )

        for disc in discontinuity:
            if disc["type"] == "overlap":
                start_page = min(
                    disc["current"]["startPage"], disc["next"]["startPage"]
                )
                end_page = max(disc["current"]["endPage"], disc["next"]["endPage"])
                print(f"Reprocessing overlap from page {start_page} to {end_page}")
            else:  # gap
                start_page = disc["gap_start"]
                end_page = disc["gap_end"]
                print(f"Reprocessing gap from page {start_page} to {end_page}")

            context_before = ""
            context_after = ""

            if disc["index"] > 0:
                prev_section = (
                    category_map[disc["index"] - 1]
                    if disc["index"] > 0
                    else category_map[0]
                )
                context_before = f"The section before this is '{prev_section['name']}' (pages {prev_section['startPage']}-{prev_section['endPage']})."

            if disc["index"] + 2 < len(category_map):
                after_section = category_map[disc["index"] + 2]
                context_after = f"The section after this is '{after_section['name']}' (pages {after_section['startPage']}-{after_section['endPage']})."

            retry_text = ""
            for page_idx in range(start_page - 1, min(end_page, len(texts))):
                if page_idx < 0 or page_idx >= len(texts):
                    continue
                text = texts[page_idx]
                if text is None:
                    continue
                lines = clean_lines(
                    [line for line in text.splitlines() if line.strip() != ""]
                )
                first_three_lines = "\n".join(lines[:3])
                last_two_lines = "\n".join(lines[-2:]) if len(lines) > 3 else ""

                if last_two_lines and last_two_lines != first_three_lines:
                    retry_text += f"Page {page_idx + 1}:\n{first_three_lines}\n...\n{last_two_lines}\n\n"
                else:
                    retry_text += f"Page {page_idx + 1}:\n{first_three_lines}\n\n"

            if not retry_text.strip():
                continue

            retry_prompt = prompt_gen.get_discontinuity_prompt(
                context_before, context_after
            )
            query = f"{retry_prompt}\n\n{retry_text}"
            print(f"Re-prompting for pages {start_page}-{end_page}...")

            retry_result = api.ask(query, model=model)

            if retry_result:
                retry_classifications = []
                for line in retry_result.strip().splitlines():
                    if (
                        line.strip()
                        and not line.startswith("Page Number")
                        and not line.startswith("`")
                    ):
                        parts = line.strip().split(",")
                        if len(parts) == 3:
                            try:
                                page_num = int(parts[0].strip())
                                category = int(parts[1].strip())
                                confidence = float(parts[2].strip())
                                retry_classifications.append(
                                    (page_num, category, confidence)
                                )
                            except ValueError:
                                continue

                for page_num, category, confidence in retry_classifications:
                    page_classifications[page_num] = (category, confidence)

        all_classifications = [
            (page, cat, conf)
            for page, (cat, conf) in sorted(page_classifications.items())
        ]
        smoothed_results = smooth_results(all_classifications)
        category_map = build_category_map(smoothed_results)

        category_map = clean_category_map(category_map)

    return category_map, discontinuity


def quick_category_title_match(
    category_map: list[dict[str, str]], texts: list[str]
) -> list[dict[str, str]]:
    """
    Pass through the pages and if the first few lines of a page match a category title,
    reclassify that page to the matched category except if it is in the middle of another category section.
    We ignore categories that are already been classified.

    Args:
        category_map (list[dict[str, str]]): List of category dictionaries
        texts (list[str]): List of text pages
    Returns:
        list[dict[str, str]]: Updated list of category dictionaries
    """

    def normalize_text(text: str) -> str:
        """Normalize text for better matching: lowercase, remove accents, extra spaces, punctuation"""
        text = unicodedata.normalize("NFKD", text)
        text = "".join([c for c in text if not unicodedata.combining(c)])
        text = re.sub(r"[^\w\s]", " ", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        return text

    title_to_category = {}
    for key, name in CATEGORY_MAP.items():
        parts = name.split(" / ")
        category_num = int(key)

        for part in parts:
            normalized = normalize_text(part)
            title_to_category[normalized] = category_num

            words = normalized.split()
            if len(words) < 2:
                continue

            title_to_category[" ".join(words[:2])] = category_num
            if len(words) < 3:
                continue

            title_to_category[" ".join(words[:3])] = category_num

    missing_categories = set(int(num) for num in CATEGORY_MAP.keys())
    for section in category_map:
        category_num = None
        for num, name in CATEGORY_MAP.items():
            if name == section["name"]:
                category_num = int(num)
                break
        if category_num in missing_categories:
            missing_categories.remove(category_num)

    for page_idx, text in enumerate(texts):
        if text is None:
            continue
        lines = [line for line in text.splitlines() if line.strip() != ""]
        if not lines:
            continue

        lines_to_check = lines[: min(2, len(lines))]

        best_match = None
        highest_ratio = 0.0
        matched_line_text = ""

        for line in lines_to_check:
            normalized_line = normalize_text(line)
            if not normalized_line:
                continue

            for title_text, category_num in title_to_category.items():
                if title_text in normalized_line:
                    ratio = 1.0
                elif normalized_line in title_text and len(normalized_line) >= 5:
                    ratio = 0.95
                else:
                    ratio = SequenceMatcher(None, normalized_line, title_text).ratio()

                if normalized_line.startswith(title_text):
                    ratio = min(1.0, ratio + 0.1)

                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = (title_text, category_num)
                    matched_line_text = line.strip()

        if best_match is None or highest_ratio < 0.75:
            continue

        _, matched_category = best_match
        confidence = round(85.0 + (highest_ratio * 15.0), 2)

        if matched_category not in missing_categories:
            continue

        in_section = False
        current_section = None
        section_idx = None
        for idx, section in enumerate(category_map):
            if section["startPage"] <= page_idx + 1 <= section["endPage"]:
                in_section = True
                current_section = section
                section_idx = idx
                break

        if in_section:
            if highest_ratio >= 0.9:

                if current_section["startPage"] < page_idx + 1:
                    current_section["endPage"] = page_idx
                    current_section["pageCount"] = (
                        page_idx - current_section["startPage"] + 1
                    )
                else:
                    category_map.pop(section_idx)

                print(
                    f"Quick match: Creating new section for category {matched_category} ({CATEGORY_MAP[str(matched_category)]}) starting at page {page_idx + 1} (confidence: {confidence}%, matched: '{matched_line_text}')"
                )
                category_map.append(
                    {
                        "name": CATEGORY_MAP[str(matched_category)],
                        "startPage": page_idx + 1,
                        "endPage": page_idx + 1,
                        "avgConfidence": confidence,
                        "pageCount": 1,
                    }
                )
                missing_categories.remove(matched_category)
            continue

        print(
            f"Quick match: Reclassifying page {page_idx + 1} to category {matched_category} ({CATEGORY_MAP[str(matched_category)]}) based on title match (confidence: {confidence}%, matched: '{matched_line_text}')"
        )
        category_map.append(
            {
                "name": CATEGORY_MAP[str(matched_category)],
                "startPage": page_idx + 1,
                "endPage": page_idx + 1,
                "avgConfidence": confidence,
                "pageCount": 1,
            }
        )
        missing_categories.remove(matched_category)

    category_map = clean_category_map(category_map)
    return category_map


def write_output(category_map: list[dict[str, str]], output_file: str):
    """
    Write the category map to the specified output file.

    Args:
        category_map (list[dict[str, str]]): List of category dictionaries
        output_file (str): Path to the output file
    """
    sections = []
    for category in category_map:
        sections.append(
            {
                "name": category.get("name").split(" / ")[0].strip(),
                "startPage": category.get("startPage"),
                "endPage": category.get("endPage"),
            }
        )

    data = {"sections": sections}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process(texts: list[str], api: APIClient, args: argparse.Namespace, hasRetried: bool = False) -> list[dict[str, str]]:
    """
    Process the texts through the AI model to generate the category map.

    Args:
        texts (list[str]): List of text pages
        api (APIClient): APIClient instance
        args (argparse.Namespace): Parsed command line arguments
        hasRetried (bool): Flag indicating if the process has been retried

    Returns:
        list[dict[str, str]]: List of category dictionaries
    """
    print("Generating prompts...")

    prompt_gen = PromptGenerator(CATEGORY_MAP)
    main_prompt = prompt_gen.get_main_prompt()

    print("Running main prompt through API...")

    batchs = create_batch(
        texts, batch_size=args.batch_size, overlap=2, first_n_lines=3, last_n_lines=2
    )
    model = MODELS.find_by_name(args.model.upper().replace("-", "_").replace(".", "_"))
    if model is None:
        print(f"Invalid model name: {args.model}")
        return
    response = run_batch_api(api, model, main_prompt, batchs)

    print("Parsing and smoothing results...")

    parsed_results = parse_response(response)
    smoothed_results = smooth_results(parsed_results)
    category_map = build_category_map(smoothed_results)

    print("Handling discontinuities...")

    category_map, discontinuity = handle_discontinuity(
        api, model, prompt_gen, category_map, texts
    )

    if discontinuity and not hasRetried:
        print(
            f"There are still {len(discontinuity)} discontinuities after reprocessing."
        )
        print("Restarting the whole process to fix remaining discontinuities...")
        return process(texts, api, args, hasRetried=True)

    print("Applying quick category title matching...")
    category_map = quick_category_title_match(category_map, texts)

    return category_map


def main():
    args = parse_args()

    print("Starting PDF processing...")
    start_time = time()

    procesor = PDFProcessor(dpi=args.dpi)
    texts = procesor.ocr_pdf(args.pdf_file)

    print("Processing texts through AI model...")
    api = APIClient(api_url=args.api_url, api_key=args.api_key)
    category_map = process(texts, api, args)

    print("Finalizing category map...")

    for category in category_map:
        page_count = category.pop("pageCount")
        category.pop("avgConfidence")
        category["pageCount"] = page_count

    print("Writing output...")
    if args.output:
        write_output(category_map, args.output)
        print(f"Output written to {args.output}")
    else:
        for category in category_map:
            print(category)

    end_time = time()
    call_count = api.call_count()
    print(
        f"Processing completed in {end_time - start_time:.2f} seconds with {call_count} API calls."
    )


if __name__ == "__main__":
    main()

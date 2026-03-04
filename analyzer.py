"""
Headline Sentiment Analyzer (Zero-Shot Classification)
Scrapes top 20 headlines from TechCrunch and classifies sentiment
using Facebook's BART-large-mnli model via Hugging Face Transformers.

Requirements:
    pip install transformers torch beautifulsoup4 requests

Note: First run will download ~1.6GB model. Subsequent runs use cache.
      CPU inference takes ~2-4 sec per headline. GPU cuts that to <0.5s.
"""

import time
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from datetime import datetime


# Candidate labels the model evaluates each headline against.
# You can customize these to fit any use case (e.g., add "Exciting", "Alarming", etc.)
SENTIMENT_LABELS = ["Positive", "Negative", "Neutral"]


def scrape_techcrunch_headlines(count=20):
    """Scrape headlines from TechCrunch homepage."""
    url = "https://techcrunch.com/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []

    # Primary: TechCrunch post-title classes
    for tag in soup.find_all(["h2", "h3"], class_=lambda c: c and "post-title" in c if c else False):
        link = tag.find("a")
        if link and link.get_text(strip=True):
            headlines.append(link.get_text(strip=True))
        if len(headlines) >= count:
            break

    # Fallback: any h2/h3 with meaningful text
    if len(headlines) < count:
        for tag in soup.find_all(["h2", "h3"]):
            link = tag.find("a")
            text = link.get_text(strip=True) if link else tag.get_text(strip=True)
            if text and len(text) > 15 and text not in headlines:
                headlines.append(text)
            if len(headlines) >= count:
                break

    return headlines[:count]


def load_classifier():
    """Load the zero-shot classification pipeline."""
    print("Loading BART-large-mnli model (first run downloads ~1.6GB)...")
    start = time.time()

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,  # CPU. Change to 0 for GPU (requires CUDA).
    )

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s\n")
    return classifier


def analyze_headlines(classifier, headlines):
    """
    Classify each headline against the candidate labels.
    Returns list of (headline, top_label, confidence, all_scores) tuples.
    """
    results = []

    for i, headline in enumerate(headlines, 1):
        output = classifier(headline, candidate_labels=SENTIMENT_LABELS)

        top_label = output["labels"][0]
        top_score = output["scores"][0]

        # Build a dict of all label scores for the detail view
        all_scores = {
            label: round(score, 3)
            for label, score in zip(output["labels"], output["scores"])
        }

        results.append((headline, top_label, top_score, all_scores))
        print(f"  [{i}/{len(headlines)}] {headline[:60]}...")

    return results


def print_table(results):
    """Print results as a formatted table with confidence scores."""
    col_num = 4
    col_headline = 52
    col_label = 10
    col_conf = 12
    total_width = col_num + col_headline + col_label + col_conf + 13

    divider = (
        f"+{'':-<{col_num + 2}}"
        f"+{'':-<{col_headline + 2}}"
        f"+{'':-<{col_label + 2}}"
        f"+{'':-<{col_conf + 2}}+"
    )

    print(f"\n{'HEADLINE SENTIMENT ANALYSIS (Zero-Shot)':^{total_width}}")
    print(f"{'Model: facebook/bart-large-mnli':^{total_width}}")
    print(f"{'Source: TechCrunch | ' + datetime.now().strftime('%B %d, %Y %I:%M %p'):^{total_width}}")
    print(divider)
    print(
        f"| {'#':>{col_num}} "
        f"| {'Headline':<{col_headline}} "
        f"| {'Sentiment':<{col_label}} "
        f"| {'Confidence':>{col_conf}} |"
    )
    print(divider)

    for i, (headline, label, confidence, _) in enumerate(results, 1):
        truncated = (headline[:col_headline - 3] + "...") if len(headline) > col_headline else headline
        conf_str = f"{confidence:.1%}"
        print(
            f"| {i:>{col_num}} "
            f"| {truncated:<{col_headline}} "
            f"| {label:<{col_label}} "
            f"| {conf_str:>{col_conf}} |"
        )

    print(divider)

    # Summary
    pos = [r for r in results if r[1] == "Positive"]
    neg = [r for r in results if r[1] == "Negative"]
    neu = [r for r in results if r[1] == "Neutral"]

    avg_conf = sum(r[2] for r in results) / len(results) if results else 0

    print(f"\n  Distribution: {len(pos)} Positive | {len(neg)} Negative | {len(neu)} Neutral")
    print(f"  Avg confidence: {avg_conf:.1%}")

    if pos:
        top_pos = max(pos, key=lambda r: r[2])
        print(f"  Most positive:  \"{top_pos[0][:55]}\" ({top_pos[2]:.1%})")
    if neg:
        top_neg = max(neg, key=lambda r: r[2])
        print(f"  Most negative:  \"{top_neg[0][:55]}\" ({top_neg[2]:.1%})")

    # Detailed breakdown for headlines where the model was uncertain
    uncertain = [r for r in results if r[2] < 0.50]
    if uncertain:
        print(f"\n  Low-confidence classifications ({len(uncertain)} headlines < 50%):")
        for headline, label, conf, scores in uncertain:
            print(f"    \"{headline[:50]}...\"")
            print(f"      Scores: {scores}")

    print()


def main():
    print("\n" + "=" * 60)
    print("  Headline Sentiment Analyzer")
    print("  Zero-Shot Classification with BART-large-mnli")
    print("=" * 60 + "\n")

    # Step 1: Scrape
    print("Scraping TechCrunch headlines...")
    try:
        headlines = scrape_techcrunch_headlines(20)
    except requests.RequestException as e:
        print(f"\nFailed to fetch headlines: {e}")
        print("Check your internet connection or try again later.")
        return

    if not headlines:
        print("No headlines found. Site structure may have changed.")
        return

    print(f"Found {len(headlines)} headlines.\n")

    # Step 2: Load model
    classifier = load_classifier()

    # Step 3: Classify
    print("Classifying headlines:")
    start = time.time()
    results = analyze_headlines(classifier, headlines)
    elapsed = time.time() - start
    print(f"\nClassification complete in {elapsed:.1f}s ({elapsed / len(results):.1f}s per headline)")

    # Step 4: Display
    print_table(results)


if __name__ == "__main__":
    main()

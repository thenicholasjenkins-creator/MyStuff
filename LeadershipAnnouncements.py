
# --- Install dependencies ---
!pip install feedparser pandas spacy
!python -m spacy download en_core_web_sm

# --- Import libraries ---
import feedparser
import pandas as pd
import os
import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# RSS feeds for credit union news + Google News search
rss_feeds = [
    "https://www.cuinsight.com/feed",
    "https://nacuso.org/feed",
    "https://servicecu.org/feed",
    "https://blog.safecu.org/rss.xml",
    "https://cutoday.info/feed",
    # Google News RSS for leadership changes in credit unions
    "https://news.google.com/rss/search?q=credit+union+CEO+appointed+OR+leadership+change&hl=en-US&gl=US&ceid=US:en"
]

# Leadership-related keywords
keywords = [
    "appointed", "named", "ceo", "chief executive", "president",
    "executive change", "promotion", "joins as ceo", "leadership team"
]

# File to store previous data
previous_data_file = "leadership_changes_rss.csv"

def contains_leadership_change(text):
    """Check if text mentions leadership change using keywords + NER."""
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    keyword_match = any(re.search(rf"\b{kw}\b", text, re.IGNORECASE) for kw in keywords)

    # Relaxed: include if keyword matches even if NER fails
    if keyword_match:
        return True, persons, orgs
    return False, persons, orgs

def fetch_leadership_changes():
    changes = []
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = entry.get("title", "")
                description = entry.get("description", "")
                link = entry.get("link", "")
                combined_text = f"{title} {description} {link}"
                is_change, persons, orgs = contains_leadership_change(combined_text)
                if is_change:
                    changes.append({
                        "Title": title.strip(),
                        "Link": link.strip(),
                        "Persons": ", ".join(persons) if persons else "N/A",
                        "Organizations": ", ".join(orgs) if orgs else "N/A"
                    })
        except Exception as e:
            print(f"Error fetching {feed_url}: {e}")
    return changes

# Fetch current changes
current_changes = fetch_leadership_changes()
current_df = pd.DataFrame(current_changes)

# Load previous data if exists
if os.path.exists(previous_data_file):
    previous_df = pd.read_csv(previous_data_file)
else:
    previous_df = pd.DataFrame(columns=["Title", "Link", "Persons", "Organizations"])

# Detect new changes
new_changes = []
for _, row in current_df.iterrows():
    if not ((previous_df["Title"] == row["Title"]) & (previous_df["Link"] == row["Link"])).any():
        new_changes.append(row.to_dict())

# Save current data
current_df.to_csv(previous_data_file, index=False)

# Output detected new changes
if new_changes:
    print("New Leadership Changes Detected:")
    for change in new_changes:
        print(f"- {change['Title']} ({change['Link']}) | Persons: {change['Persons']} | Orgs: {change['Organizations']}")
else:
    print("No new leadership changes detected.")

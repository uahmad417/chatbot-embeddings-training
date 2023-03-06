import sys

page_start_delimiters = ["The Great Gatsby", "Free eBooks at Planet eBook.com"]

chapter_number = 0
pages = []
current_page = ""
page_number = 0

with open(sys.argv[1]) as f:
    text = f.readlines()
for line in text:
    line = line.strip()
    # Check if the line matches any of the chapter delimiters
    if "chapter" in line.lower():
        # If so, increment the chapter number
        chapter_number += 1
        continue
    # Check if the line matches any of the start delimiters
    if any(start in line for start in page_start_delimiters):
        # If so, append the current page to the list of pages and start a new page
        if current_page:
            page_number += 1
            pages.append({
                "chapter": chapter_number,
                "page": current_page,
                "page_number": page_number
            })
            current_page = ""
        continue
    # Append the current line to the current page
    current_page += line + "\n"

# Append the final page if there is any remaining text
if current_page:
    page_number += 1
    pages.append({
        "chapter": chapter_number,
        "page": current_page,
        "page_number": page_number
    })

# Print the resulting pages
for i, page in enumerate(pages[:5]):
    print(f"Chapter {page['chapter']}, Page {page['page_number']}:\n{page['page'][0:100]}...\n...{page['page'][-100:]}")
    print("-" * 80)

# dump the pages array into a json file
import json
with open('pages.json', 'w') as f:
    json.dump(pages, f)
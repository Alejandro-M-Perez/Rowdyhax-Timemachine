# Web Scraper for llm RAG

# To do List:
# 1. Grab "See also" pages from "See also section"
# 2. Add "See also" pages to the input text
# 3. Add Text to llm RAG

import requests
from bs4 import BeautifulSoup as bs
import wikipediaapi

# Prompt user for Name of person to search
name = input("Enter the name of the person you would like to search: ")

# URL to scrape
wiki_wiki = wikipediaapi.Wikipedia('RowdyHack_2024 (Alejandromperez714@gmail.com)', 'en')

page = wiki_wiki.page(name)
if not page.exists():
    print("Target does not exist")
else:
    page_text = page.text

    # Extract lines from "See also" section to the next blank line
    see_also_index = page_text.find("See also")
    if see_also_index != -1:
        see_also_text = page_text[see_also_index:]
        end_index = see_also_text.find("\n\n")
        if end_index != -1:
            see_also_text = see_also_text[:end_index]

        print("See also section:")
        print(see_also_text)

        # Remove text after "See also" section
        page_text = page_text[:see_also_index]

    #print("Page - Title: %s" % page.title)
    #print("Page - Summary: %s" % page.summary[0:600])
    print(page_text)





    
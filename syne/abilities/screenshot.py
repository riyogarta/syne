#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ABILITY DEFINITION
# name: screenshot
# description: "Capture a screenshot of a website."
# parameters:
#   - name: url
#     type: string
#     description: "The URL of the website to capture (e.g., https://example.com)."
#     required: true
#   - name: full_page
#     type: boolean
#     description: "Capture the full scrollable page."
#     required: false
#     default: false
# ---

import argparse
import sys
from playwright.sync_api import sync_playwright, Error

def take_screenshot(url, output_path, full_page=False):
    """
    Takes a screenshot of a URL using Playwright.
    """
    try:
        with sync_playwright() as p:
            # Using chromium as it's the most common
            browser = p.chromium.launch()
            page = browser.new_page()
            # Increase timeout to 60 seconds for slower pages
            page.goto(url, wait_until='networkidle', timeout=60000)
            
            page.screenshot(
                path=output_path,
                full_page=full_page,
                type='png' # Force PNG for simplicity
            )
            browser.close()
            return output_path
    except Error as e:
        # Provide a more specific error message if possible
        print(f"Playwright Error: Could not take screenshot of {url}. Reason: {e}", file=sys.stderr)
        # Check for a common error when running headless
        if "browser has been closed" in str(e) or "Target page, context or browser has been closed" in str(e):
             print("\nHint: This might be due to missing system dependencies for the headless browser. Please check the logs.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    # This part is executed when the ability is called by Syne's runner
    parser = argparse.ArgumentParser(description="Take a screenshot of a website.")
    parser.add_argument("--url", required=True, help="The URL to screenshot.")
    parser.add_argument("--output", required=True, help="Path to save the screenshot.")
    parser.add_argument("--full_page", action="store_true", help="Capture the full page.")
    
    args = parser.parse_args()

    # Ensure URL has a scheme
    if not args.url.startswith(('http://', 'https://')):
        url = 'http://' + args.url
    else:
        url = args.url

    result_path = take_screenshot(
        url=url,
        output_path=args.output,
        full_page=args.full_page
    )

    if result_path:
        # Syne expects the output path on stdout for success
        print(result_path)
        sys.exit(0)
    else:
        # Errors were printed to stderr, so just exit with a failure code
        sys.exit(1)

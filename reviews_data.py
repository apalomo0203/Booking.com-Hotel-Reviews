import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import time

async def scrape_reviews():

    hotels = pd.read_csv("booking_hotels_chi.csv")

    reviews_list = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        page = await browser.new_page()

        # Loop through all hotels
        for idx, row in hotels.iterrows():
            hotel_name = row["Name"]
            hotel_url  = row["URL"]


            # Go to hotel page
            await page.goto(hotel_url, timeout=60000)
            await asyncio.sleep(3)

            # Scroll 
            await page.mouse.wheel(0, 2000)
            await asyncio.sleep(1.5)

            # Read all reviews button
            review_button = page.locator("button[data-testid='fr-read-all-reviews']")

            #In case one of the hotels does not have a reall all reviews button
            if await review_button.count() > 0:
                await review_button.first.scroll_into_view_if_needed()
                await asyncio.sleep(0.5)
                await review_button.first.click()
                await asyncio.sleep(3)
            else:
                print("No button")
                continue

            # Scroll inside to load more reviews
            for _ in range(5):
                await page.mouse.wheel(0, 4000)
                await asyncio.sleep(1)

            # Select all review cards
            cards = await page.query_selector_all("div[data-testid='review-card']")

            for card in cards:
                parts = []   # We will add title + positive + negative

                #Title
                title_el = await card.query_selector("h4[data-testid='review-title']")
                if title_el is not None:
                    title_text = (await title_el.inner_text()).strip()
                    if title_text:
                        parts.append(title_text)

                #Positive Text
                pos_el = await card.query_selector("div[data-testid='review-positive-text']")
                if pos_el is not None:
                    pos_text = (await pos_el.inner_text()).strip()
                    if pos_text:
                        parts.append(pos_text)

                #Negative Text
                neg_el = await card.query_selector("div[data-testid='review-negative-text']")
                if neg_el is not None:
                    neg_text = (await neg_el.inner_text()).strip()
                    if neg_text:
                        parts.append(neg_text)

                # Combine the three pieces of reviews into one single text block
                full_text = " ".join(parts) if parts else None

                reviews_list.append({
                    "Hotel Name": hotel_name,
                    "Review": full_text
                })

        # Close browser
        await browser.close()

        # Save all reviews to CSV
        df = pd.DataFrame(reviews_list)
        df.to_csv("booking_hotel_reviews_chi.csv", index=False)


asyncio.run(scrape_reviews())
import pandas as pd
from playwright.sync_api import sync_playwright, Playwright
import re
import asyncio
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as pw:

        browser = await pw.chromium.launch(headless=False)
        page = await browser.new_page()

        await page.goto("https://www.booking.com/searchresults.html?ss=Chicago&checkin=2025-12-12&checkout=2025-12-13&lang=en-us")

        await page.wait_for_selector('div[data-testid="property-card"]') #Booking loads hotels dynamically

        hotels = page.get_by_test_id('property-card')
        hotel_count = await hotels.count()
        print(hotel_count)



        hotel_list = []

        #Loop through each hotel
        for i in range(hotel_count):
            hotel = hotels.nth(i)
#NAME
            name = None
            if await hotel.get_by_test_id('title').is_visible():
            #if await hotel.locator('div[data-testid="title"]').is_visible():
                name = await hotel.get_by_test_id('title').inner_text()
                #name = await hotel.locator('div[data-testid="title"]')


#Rating
            rating = None

            if await hotel.get_by_test_id('review-score').is_visible():
                review_text = await hotel.get_by_test_id('review-score').inner_text()
            
                lines = review_text.split('\n')

                if len(lines) >= 1: 
                    rating = lines[0].strip()
                if len(lines) >= 2:
                    count_reviews = re.sub(r'\D', '', lines[1].strip())
#PRICE
            price = None
            price_element = hotel.get_by_test_id('price-and-discounted-price').first 
            
            if await price_element.is_visible():
                price_text = await price_element.inner_text()
                price = re.sub(r'[^0-9]', '', price_text)

           

#DATAFRAME
            hotel_df = pd.DataFrame({
                "Name": [name],
                "Rating": [rating],
                "Number of Reviews": [count_reviews],
                "Price per Night": [price]
            })
            hotel_list.append(hotel_df)
#COMBINE ALL 
        hotel_results = pd.concat(hotel_list, ignore_index=True)

#SAVE TO CSV
        hotel_results.to_csv("booking_hotels_chi.csv", index=False)
        await browser.close()



asyncio.run(main())



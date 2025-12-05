import pandas as pd
import re
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


reviews = pd.read_csv(
    "C:/Users/Alejandra Palomo/Unstructured Data/Final Project/booking_hotel_reviews_chi.csv",
    encoding="latin1"
)

hotels = pd.read_csv(
    "C:/Users/Alejandra Palomo/Unstructured Data/Final Project/booking_hotels_chi.csv",
    encoding="latin1"
)

stopwords = {"the","and","is","to","it","in","that","was","for","of","with",
    "this","but","on","we","they","be","as","are","at","you","i","so",
    "had","have","were","my","our","me","from","or","by","an","not","all","there"}

sample = reviews["Review"].loc[0]
print(sample)

#Convert to string 
sample_string = str(sample)
sample_string 

#Remove NA from the reviews
sample_no_na = re.sub(r"\bNA\b", "", sample_string)
print(sample_no_na)

#Remove html tags 
sample_no_html = re.sub(r"<.*?>", "", sample_no_na)
print(sample_no_html)

#Remove stopwords
no_stpwrds = sample_no_html.split()
words_no_stop = [w for w in no_stpwrds if w not in stopwords]
sample_clean_final = " ".join(words_no_stop)
print(sample_clean_final)


def clean_review(text):
    text = str(text).strip()
    try:
        text = text.encode("latin1").decode("utf-8")
    except:
        pass

    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\b(NA|N/A|Na|na|null)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"ðŸ\w*", "", text)
    text = text.encode("ascii", "ignore").decode()

    text = re.sub(r"\s+", " ", text).strip()
    

    words = text.split()
    words_no_stop = [w for w in words if w not in stopwords]

    return " ".join(words_no_stop)

reviews["cleaned_review"] = reviews["Review"].apply(clean_review)

# Remove original messy review column
reviews = reviews.drop(columns=["Review"])

#Sentiment Analysis 
vader = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return vader.polarity_scores(str(text))['compound']

reviews["sentiment"] = reviews["cleaned_review"].apply(get_sentiment)



#Merge and save to csv
merged = reviews.merge(
    hotels,
    left_on="Hotel Name",   # column from reviews
    right_on="Name",        # column from hotels
    how="left"
)

merged = merged.drop(columns=["Name"])
merged.to_csv("merged_hotel_reviews_chi.csv", index=False)




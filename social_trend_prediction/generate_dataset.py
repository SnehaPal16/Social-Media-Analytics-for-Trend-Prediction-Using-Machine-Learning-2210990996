"""
generate_dataset.py
Generates a synthetic tweets_sample.csv that mirrors the dataset described
in the research paper (150 k records, 31% trending / 69% non-trending).
Run once: python generate_dataset.py
"""

import random
import csv
import datetime
import os

random.seed(42)

TRENDING_HASHTAGS = [
    "#GlobalClimateSummit", "#AIRevolution", "#ElectionDay2024",
    "#WorldCup2024", "#CovidUpdate", "#TechLayoffs", "#MarsLanding",
    "#CryptoNews", "#NASADiscovery", "#ClimateChange",
]
NON_TRENDING_HASHTAGS = [
    "#MondayMotivation", "#ThrowbackThursday", "#FoodPhotography",
    "#PetLife", "#Gardening", "#BookReview", "#DailyWalk",
    "#CoffeeLover", "#SundayVibes", "#TravelDiaries",
]
TWEET_TEMPLATES_TRENDING = [
    "Breaking: Major developments in {topic} – everyone needs to see this! {hashtag}",
    "This is huge! {topic} is changing everything. {hashtag} {hashtag2}",
    "Scientists confirm {topic}. The implications are enormous. {hashtag}",
    "URGENT: {topic} update just dropped. Stay informed. {hashtag} {hashtag2}",
    "Opinion: {topic} will define the next decade. Here's why. {hashtag}",
    "Just heard about {topic} – cannot believe this is happening. {hashtag}",
    "Everyone talking about {topic} right now. Don't miss out! {hashtag} {hashtag2}",
    "LIVE updates on {topic} coming in. Follow for more. {hashtag}",
]
TWEET_TEMPLATES_NON_TRENDING = [
    "Good morning! Starting the day with coffee and {hashtag} vibes.",
    "Just finished a great book. Highly recommend it! {hashtag}",
    "Beautiful sunset today. Nature is amazing. {hashtag}",
    "Tried a new recipe today – turned out great! {hashtag}",
    "Working from home again. At least the coffee is good. {hashtag}",
    "Weekend plans: relax and enjoy {hashtag} moments.",
    "Throwback to that amazing trip last year. {hashtag}",
    "Nothing beats a quiet Sunday morning. {hashtag}",
]
TOPICS = [
    "the Climate Summit", "AI legislation", "the election results",
    "the new space mission", "pandemic policy", "tech sector changes",
    "the Mars discovery", "crypto regulations",
]

def make_tweet(trending: bool, ts: datetime.datetime) -> dict:
    if trending:
        ht1 = random.choice(TRENDING_HASHTAGS)
        ht2 = random.choice(TRENDING_HASHTAGS)
        topic = random.choice(TOPICS)
        tmpl = random.choice(TWEET_TEMPLATES_TRENDING)
        text = tmpl.format(topic=topic, hashtag=ht1, hashtag2=ht2)
        hashtags = f"{ht1} {ht2}"
        likes = random.randint(150, 5000)
        retweets = random.randint(80, 3000)
        hashtag_count = random.randint(2, 5)
        hashtag_24h_freq = random.randint(500, 8000)
        hashtag_velocity = round(random.uniform(0.4, 1.0), 4)
        sentiment = round(random.uniform(0.2, 0.9), 4)
    else:
        ht = random.choice(NON_TRENDING_HASHTAGS)
        tmpl = random.choice(TWEET_TEMPLATES_NON_TRENDING)
        text = tmpl.format(hashtag=ht)
        hashtags = ht
        likes = random.randint(0, 150)
        retweets = random.randint(0, 50)
        hashtag_count = random.randint(1, 3)
        hashtag_24h_freq = random.randint(10, 500)
        hashtag_velocity = round(random.uniform(0.0, 0.3), 4)
        sentiment = round(random.uniform(-0.4, 0.4), 4)

    rt_like_ratio = round(retweets / likes, 4) if likes > 0 else 0.0
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "hashtags": hashtags,
        "like_count": likes,
        "retweet_count": retweets,
        "hashtag_count": hashtag_count,
        "hashtag_24h_freq": hashtag_24h_freq,
        "hashtag_velocity": hashtag_velocity,
        "sentiment_polarity": sentiment,
        "retweet_like_ratio": rt_like_ratio,
        "trending": int(trending),
    }

def generate(n: int = 150_000, out: str = "data/tweets_sample.csv"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    fields = [
        "timestamp", "text", "hashtags", "like_count", "retweet_count",
        "hashtag_count", "hashtag_24h_freq", "hashtag_velocity",
        "sentiment_polarity", "retweet_like_ratio", "trending",
    ]
    # 31% trending, 69% non-trending (matches paper)
    n_trending = int(n * 0.31)
    n_non = n - n_trending
    rows = []
    for i in range(n_trending):
        ts = start + datetime.timedelta(seconds=random.randint(0, 180*24*3600))
        rows.append(make_tweet(True, ts))
    for i in range(n_non):
        ts = start + datetime.timedelta(seconds=random.randint(0, 180*24*3600))
        rows.append(make_tweet(False, ts))
    random.shuffle(rows)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅  Generated {n:,} tweets → {out}")
    print(f"   Trending: {n_trending:,} ({100*n_trending/n:.0f}%)")
    print(f"   Non-trending: {n_non:,} ({100*n_non/n:.0f}%)")

if __name__ == "__main__":
    generate()

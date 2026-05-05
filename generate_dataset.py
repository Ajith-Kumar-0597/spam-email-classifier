"""
generate_dataset.py
====================
Generates a realistic spam vs ham email dataset.
Run this ONCE before train.py if you don't have the Kaggle dataset.

To use the real SpamAssassin dataset instead:
  kaggle datasets download -d beatoa/spamassassin-public-corpus --unzip -p data/
  Then update DATA_PATH in train.py accordingly.
"""

import pandas as pd
import random

random.seed(42)

SPAM_TEMPLATES = [
    "CONGRATULATIONS! You have been selected as the winner of ${amount} lottery. Click here to claim now!",
    "FREE offer! Get a ${product} absolutely free. Limited time offer. Act now!",
    "URGENT: Your account has been compromised. Verify your details immediately at {url}",
    "Make ${amount} per day working from home! No experience needed. Join thousands of happy earners.",
    "You have won a ${product}! To claim your prize, send your bank details to {email}",
    "Dear friend, I am a Nigerian prince and I need your help to transfer ${amount} million dollars.",
    "WEIGHT LOSS SECRET doctors don't want you to know! Lose {number} lbs in {number} days guaranteed!",
    "Buy cheap medications online! No prescription needed. Viagra, Cialis from ${amount}.",
    "YOUR LOAN IS APPROVED! Get ${amount} in your account within 24 hours. No credit check!",
    "HOT SINGLES in your area want to meet you! Click here to see profiles.",
    "INVESTMENT OPPORTUNITY: Double your money in {number} days. Guaranteed returns!",
    "Claim your FREE iPhone {number}! You are our {number}th visitor. Click to claim!",
    "ALERT: Suspicious login detected on your PayPal account. Verify now: {url}",
    "Earn ${amount} just by clicking ads! Work from home, be your own boss.",
    "Limited offer: 90% OFF all products! Shop now at {url} before offer expires!",
    "You have been pre-approved for a credit card with ${amount} limit. No fees ever!",
    "BITCOIN opportunity! Invest ${amount} and get ${amount} back in {number} hours.",
    "Meet local {city} singles tonight! Register free at {url}",
    "Your computer has {number} viruses! Download our FREE antivirus now!",
    "Refinance your mortgage! Save ${amount} per month. Apply in {number} minutes.",
]

HAM_TEMPLATES = [
    "Hi {name}, just wanted to follow up on the project status. Can we schedule a call this week?",
    "Please find attached the meeting minutes from yesterday's team sync. Let me know if you have questions.",
    "Hi team, reminder that the sprint review is scheduled for {day} at {time}. Please come prepared.",
    "Hey {name}, happy birthday! Hope you have a wonderful day.",
    "Your order #{number} has been shipped. Expected delivery: {day}. Track your order at our website.",
    "Hi {name}, thanks for your application. We would like to invite you for an interview on {day}.",
    "Reminder: Your dentist appointment is scheduled for {day} at {time}. Please confirm.",",
    "Please review the attached document and share your feedback by {day}.",
    "Hi {name}, I wanted to share this interesting article about {topic} that I thought you might enjoy.",
    "The quarterly report is ready. Key highlights: revenue up {number}%, costs down {number}%.",
    "Team lunch is planned for {day}. We will be going to {city} restaurant at {time}.",
    "Hi {name}, can you please send me the updated version of the presentation before {day}?",
    "Your subscription to {product} has been renewed successfully. Next billing date: {day}.",
    "Meeting agenda for tomorrow: 1) Project updates 2) Budget review 3) Q&A. See you at {time}.",
    "Thanks for attending the workshop today. Here are the resources shared during the session.",
    "Hi {name}, just checking in. How is the {topic} project coming along?",
    "Your password was changed successfully. If you did not make this change, contact support.",
    "Friendly reminder: the {topic} report is due on {day}. Please submit on time.",
    "Hi {name}, I have reviewed your code. Left some comments on the pull request. Good work overall!",
    "The server maintenance is scheduled for {day} at {time}. Expected downtime: {number} minutes.",
]

def fill_template(template):
    replacements = {
        "{amount}"  : str(random.randint(100, 9999)),
        "{number}"  : str(random.randint(1, 99)),
        "{url}"     : random.choice(["http://bit.ly/win123", "http://click-here.net", "http://verify-now.com"]),
        "{email}"   : random.choice(["prince@nigeria.com", "claims@lottery.net"]),
        "{product}" : random.choice(["iPhone", "MacBook", "Samsung TV", "iPad", "Rolex"]),
        "{name}"    : random.choice(["John", "Sarah", "Michael", "Emma", "David"]),
        "{day}"     : random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        "{time}"    : random.choice(["10:00 AM", "2:00 PM", "3:30 PM", "11:00 AM"]),
        "{city}"    : random.choice(["New York", "London", "Mumbai", "Chennai", "Bangalore"]),
        "{topic}"   : random.choice(["machine learning", "data science", "AI", "Python", "cloud computing"]),
    }
    for key, val in replacements.items():
        template = template.replace(key, val)
    return template


rows = []

for _ in range(1500):
    rows.append({
        "text" : fill_template(random.choice(SPAM_TEMPLATES)),
        "label": "spam"
    })

for _ in range(2500):
    rows.append({
        "text" : fill_template(random.choice(HAM_TEMPLATES)),
        "label": "ham"
    })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/emails.csv", index=False)

spam_count = (df["label"] == "spam").sum()
ham_count  = (df["label"] == "ham").sum()
print(f"[✓] Dataset saved → data/emails.csv")
print(f"    Total emails : {len(df):,}")
print(f"    Spam         : {spam_count:,} ({spam_count/len(df)*100:.1f}%)")
print(f"    Ham          : {ham_count:,}  ({ham_count/len(df)*100:.1f}%)")
print(df.sample(3))

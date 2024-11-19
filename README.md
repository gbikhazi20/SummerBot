# SummerBot

Hello!

This is a small project for [Summer Health](https://www.summerhealth.com/topics).

I finetuned a small Google Flan model on some Q and A data I scraped from [summerhealth.com/topics](summerhealth.com/topics)

The idea: If Summer Health pediatricians are stretched thin, users could be temporarily forwarded to a chatbot with domain knowledge so they can get answers instantly.

The results are... not that great :D (At least when compared to state-of-the-art models/ chatbots). When I went model shopping, my options were pretty limited: I had to pick something I could train locally on my ThinkPad.
I think this has the potential to be much better given more compute and a bigger model!

If you want to try this out yourself, clone this repo and run the following commands:

    1. pip install -r requirements.txt
    2. python scrape.py
    3. python finetune.py
    4. python chat.py

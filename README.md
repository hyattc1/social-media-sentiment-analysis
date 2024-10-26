# Cybersecurity Attacker Ideology Analysis
## Project Overview
This project features a series of eight visual displays from Facebook and Twitter data containing thousands of messages sent by US Senators and politicians. A focus on sentiment and ideological trends are examined through algorithms built for scraping and cleaning data, targeting hashtags, calculating sentiment scores, and finding geographic indicators. The project’s primary objectives are to:
- Scrape and process data from social media platforms.
- Analyze and categorize content based on sentiment and geographic indicators.
- Visualize ideological patterns and messaging trends in U.S. politics.
## Key Features
- Data Collection: Algorithms developed with pandas and the nltk SentimentIntensityAnalyzer can clean data from multiple sources.
- Sentiment Analysis: Natural language processing methods classify messages and assess sentiment using both nltk and custom sentiment scoring.
- Hashtag and Geographic Tagging: Identification of prominent hashtags and extraction of geographic information embedded in social media messages.
- Visualization: Comprehensive visualizations generated with matplotlib highlight patterns and trends in political messaging. Charts portray total messages per state, total *facebook* messages per state, top attack messages per state, top policy messages per state, top national messages per state, most frequent buzzwords used per region of the US, political alignment by region, and hashtag sentiment score vs frequency used.
## Tools and Libraries
- Python: Core language used for data manipulation, analysis, and visualization.
- Pandas: Data manipulation library for handling and cleaning datasets.
- NLTK (Natural Language Toolkit): Used for language processing, tokenization, and sentiment analysis.
- Matplotlib: Visualization library for generating comprehensive visual insights.
## Dataset Information
The data used in this project comes from the Political Social Media Posts dataset on Kaggle, contributed by Crowdflower as part of their Data for Everyone Library. It contains 5,000 messages from social media accounts of U.S. politicians, including U.S. Senators. Each message is tagged with:
- Audience: either national or constituency.
- Bias: either neutral or partisan.
- Message purpose: includes categories like attack, information, mobilization, and more.
- Confidence measures for each tag, and other metadata such as embed codes, message IDs, and text.

https://www.kaggle.com/datasets/crowdflower/political-social-media-posts/data

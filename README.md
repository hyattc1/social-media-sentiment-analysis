#Cybersecurity Attacker Ideology Analysis
This project features a series of visual displays derived from social media data, with a focus on sentiment and ideological trends across messages shared by U.S. Senators and politicians. It leverages data from Facebook and Twitter, containing thousands of messages for deep analysis of themes and attitudes using advanced data processing techniques.

##Project Overview
The repository includes a set of eight data visualizations from messages sourced on Facebook and Twitter, specifically targeting sentiment and thematic indicators from the political arena. The project’s primary objectives are to:
- Scrape and preprocess data from social media platforms.
- Analyze and categorize content based on sentiment and geographic indicators.
- Visualize ideological patterns and messaging trends in U.S. politics.
##Key Features
- Data Collection: Algorithms developed with pandas and nltk extract and clean data from multiple sources.
- Sentiment Analysis: Natural language processing methods classify messages and assess sentiment using both nltk and custom sentiment scoring.
- Hashtag and Geographic Tagging: Identification of prominent hashtags and extraction of geographic information embedded in social media messages.
- Visualization: Comprehensive visualizations generated with matplotlib highlight patterns and trends in political messaging.
##Tools and Libraries
- Python: Core language used for data manipulation, analysis, and visualization.
- Pandas: Data manipulation library for handling and cleaning datasets.
- NLTK (Natural Language Toolkit): Used for language processing, tokenization, and sentiment analysis.
- Matplotlib: Visualization library for generating comprehensive visual insights.
##Dataset Information
The data used in this project comes from the Political Social Media Posts dataset on Kaggle, contributed by Crowdflower as part of their Data for Everyone Library. https://www.kaggle.com/datasets/crowdflower/political-social-media-posts/data
This dataset contains 5,000 messages from social media accounts of U.S. politicians, including U.S. Senators. Each message is tagged with:
- Audience: either national or constituency.
- Bias: either neutral or partisan.
- Message purpose: includes categories like attack, information, mobilization, and more.
- Confidence measures for each tag, and other metadata such as embed codes, message IDs, and text.

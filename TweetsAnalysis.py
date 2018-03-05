#Script to retrieve and analyze Twitter data for Tableau visualization
#Author: Alisher Mansurov


""" Get required packages """
import csv
import tweepy
import datetime
import xlsxwriter
import networkx
import twython
import pandas as pd
import datetime
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from collections import defaultdict
from string import punctuation
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


""" Connect to Twitter API """
ACCESS_TOKEN = "****************"
ACCESS_TOKEN_SECRET = "****************"
CONSUMER_KEY = "****************"
CONSUMER_SECRET = "****************"
STOPWORDS = set(stopwords.words('english'))

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth, wait_on_rate_limit=True)

""" Twitter usernames of interest """
usernames = [
            #
            '@Financial_MD', '@CMA_Docs', '@Joule_CMA', '@CIHI_ICIS', '@cpso_ca', '@MSF_canada', '@MMJPRca', '@CAPHCTweets', '@CPHA_ACSP', #Organizations
            '@ResidentDoctors', '@OMSAofficial', '@ResidentDocsBC', '@para_ab', '@ResidentDocsSK', #Canadian Resident Doctors
            '@MacHealthSci', '@MUNMed', '@uoftmedicine', '@DalMedSchool', '@uOttawaMed', '@SchulichMedDent', '@UBCmedicine', '@UAlberta_FoMD',
            '@UCalgaryMed', '@USaskMedDean', '@USherbrooke', '@UMontreal', '@McGillMed', '@facmedUL', '@UM_RadyFHS', #Canadian Medical Schools
            '@DoctorsOfBC', '@SMA_docs', '@Albertadoctors', '@_nlma', '@Doctors_NS', '@OnCall4ON', '@OntariosDoctors', '@nb_docs', '@MSPEI_Docs', '@amquebec', #Provincial and Territorial Medical Associations
            '@CFMSFEMC', '@AMMICanada', '@CASUpdate', '@CAEP_Docs', '@CanGastroAssn', '@canmacmn', '@CAPACP', '@CAPSsurgeons', '@CAPM_R', '@caro_acro_ca', '@CARadiologists', '@CritCareSociety', '@CdnDermatology', '@CanGeriSoc', '@CNS_Update', '@SocCdnNSx', '@CANeyeMDs', '@CdnOrthoAssoc', '@CanPaedSociety', '@CPA_APC', '@CSACI_ca', '@CSIMSCMI', '@CSPCP_SCMSP', '@CanUrolAssoc', '@FAMS_SE', '@CAOT_ACE', #MD Investor Affiliated Societies
            '@CASEMACMSE', '@CAPE_Doctors', '@CAPD_ca', '@CMPAmembers', '@FMRAC_ca', '@FMWCanada', '@SRPCanada'] # MD Investor Associated Societies


""" Time period from tweets """
startDate = datetime.datetime.now() - datetime.timedelta(days=7, hours=0, minutes=0, seconds=0)
endDate = datetime.datetime.now()

tweets = []

pd.options.display.max_colwidth = 200
""" Get tweets for the above specied period (past week) """
for user in usernames:
    tmpTweets = api.user_timeline(user)
    for tweet in tmpTweets:
        if tweet.created_at < endDate and tweet.created_at > startDate:
            tweets.append(tweet)

    while (tmpTweets[-1].created_at > startDate):
        tmpTweets = api.user_timeline(user, max_id=tmpTweets[-1].id)
        for tweet in tmpTweets:
            if tweet.created_at < endDate and tweet.created_at > startDate:
                tweets.append(tweet)


def all_tweets(tweets):
""" Write tweet details in a Pandas dataframe """
    id_list = [tweet.id for tweet in tweets]
    tweets_dataframe = pd.DataFrame(id_list, columns=['tweet_id'])
    tweets_dataframe['screen_name'] = [tweet.user.screen_name for tweet in tweets]
    tweets_dataframe['tweet_text'] = [tweet.text for tweet in tweets]
    tweets_dataframe['tweet_date'] = [tweet.created_at for tweet in tweets]
    tweets_dataframe['tweet_lang'] = [tweet.lang for tweet in tweets]
    tweets_dataframe['tweet_source'] = [tweet.source for tweet in tweets]
    tweets_dataframe['retweet'] = [tweet.retweet for tweet in tweets]
    tweets_dataframe['retweets_count'] = [tweet.retweet_count for tweet in tweets]
    tweets_dataframe['coordinates'] = [tweet.coordinates for tweet in tweets]
    tweets_dataframe['geo'] = [tweet.geo for tweet in tweets]
    tweets_dataframe['entities'] = [tweet.entities for tweet in tweets]

    #tweets_dataframe['favorites_count'] = [tweet.favorite_count for tweet in tweets]
    tweets_dataframe['favorites_count'] = [tweet.favorite_count for tweet in tweets]
    tweets_dataframe['time_zone'] = [tweet.user.time_zone for tweet in tweets]
    tweets_dataframe['data_pull_date'] = [endDate for tweet in tweets]
    tweets_dataframe['followers_count'] = [tweet.user.followers_count for tweet in tweets]
    tweets_dataframe['friends_count'] = [tweet.user.friends_count for tweet in tweets]
    tweets_dataframe['geo_enabled'] = [tweet.user.geo_enabled for tweet in tweets]
    tweets_dataframe['time_zone'] = [tweet.user.time_zone for tweet in tweets]
    tweets_dataframe['location'] = [tweet.user.location for tweet in tweets]

    return tweets_dataframe


tweets_df = all_tweets(tweets)


def vader_sentiment(tweets):
""" Get Vader Sentiment scores """
    vs=[]
    analyzer = SentimentIntensityAnalyzer()
    for tweet in tweets:
        a = analyzer.polarity_scores(tweet)['compound']
        vs.append(a)
    return vs

tweets = tweets_df['tweet_text']
tweets_df['vader_sentiment'] = vader_sentiment(tweets)



def textblob_sentiment(tweets):
""" Get Textblob Sentiment scores (en) """
    tb = []
    tb_fr = []
    for tweet in tweets:
        blob_tweet = TextBlob(tweet)
        tb.append(blob_tweet.sentiment.polarity)
    return tb

def textblob_sentiment_fr(tweets):
""" Get Textblob Sentiment scores (fr) """
    tb_fr = []
    for tweet in tweets:
        blob_tweet_fr = TextBlob(tweet, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        tb_fr.append(blob_tweet_fr.sentiment[0])
    return tb_fr

tweets_df['textblob_sentiment'] = textblob_sentiment(tweets)
tweets_df['textblob_sentiment_fr'] = textblob_sentiment_fr(tweets)


"""Define stopwords"""
STOPWORDS = set(stopwords.words('english'))


def get_wordnet_pos(tag):
    """ Get wordnet part of speech from a tag. Return None if
    nothing matches """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return None


def lemmatize_sentence(text):
    """ Lemmatize text while removing stopwords from the outcome """
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word, tag in pos_tag(word_tokenize(text)):
        if word in punctuation or word.lower() in STOPWORDS:
            continue
        pos = get_wordnet_pos(tag)
        if pos:
            lemmas.append(lemmatizer.lemmatize(word, pos=pos))
    return lemmas


def analyze(tweets):
	""" Calculate tweet scores based on frequencies """
    # print("All tweets: ", tweets)
    all_words = defaultdict(lambda: 0)

    tweet_lemmas = []
    for tweet in tweets:
        lemmas = lemmatize_sentence(tweet)
        for lemma in lemmas:
            all_words[lemma] += 1
        tweet_lemmas.append(lemmas)

    tweet_scores = []
    for lemmas in tweet_lemmas:
        score = 0
        for lemma in lemmas:
            score += all_words[lemma]
            # print("Lemma score: %s = %d" % (lemma, all_words[lemma]))
        tweet_scores.append(score)
        # print("Tweet scores: ", tweet_scores)

    return tweet_scores
    scored_tweets = list(zip(tweets, tweet_scores))


tweets = tweets_df['tweet_text']

tweets_df['tweet_score'] = analyze(tweets)

tweets_df.to_csv('****************', index=False, encoding='utf-8')


def get_hashtags(text):
""" Get hashtags only """
    hashtags = [tag.replace(':', '').replace('.', '').replace(',', '').replace('_', '') for tag in text.split() if
                tag.startswith("#")]
    return hashtags


def get_ats(text):
""" Get @s only """
    ats = [tag.replace(':', '').replace('.', '').replace('  ', '').replace(',', '').replace('_', '') for tag in text.split() if
           tag.startswith("@")]
    return ats


def analyze_hashtags(tweets):
""" Calculate hashtag scores based on frequencies """
    tweet_hashtags = []
    all_tags = defaultdict(lambda: 0)
    for tweet in tweets:
        tags = get_hashtags(tweet)
        for tag in tags:
            all_tags[tag] += 1
            tweet_hashtags.append(tag)

    tag_scores = []
    score = 0
    for tag in tweet_hashtags:
        score += all_tags[tag]
        tag_scores.append(score)

    return all_tags


def analyze_ats(tweets):
""" Calculate @ scores based on frequencies """
    tweet_ats = []
    all_at_tags = defaultdict(lambda: 0)
    for tweet in tweets:
        tags = get_ats(tweet)
        for tag in tags:
            all_at_tags[tag] += 1
            tweet_ats.append(tag)

    tag_scores = []
    score = 0
    for tag in tweet_ats:
        score += all_at_tags[tag]
        tag_scores.append(score)

    return all_at_tags


# tweets = tweets_df['tweet_text']
tweets = tweets_df['tweet_text']

h_dict = analyze_hashtags(tweets)
a_dict = analyze_ats(tweets)

pd.options.display.max_colwidth = 100

h_keys = list(h_dict.keys())
h_values = list(h_dict.values())
a_keys = list(a_dict.keys())
a_values = list(a_dict.values())

hashtags_df = pd.DataFrame()
ats_df = pd.DataFrame()

hashtags_df['hashtag'] = h_keys
hashtags_df['hashtag_occurence'] = h_values
ats_df['at'] = a_keys
ats_df['at_occurence'] = a_values



s_name = []
t_text = []
hashtag = []
sentiment = []
""" Assign sentiment to hashtags """
for index, row in hashtags_df.iterrows():
    for index, roww in tweets_df.iterrows():
        if row['hashtag'] in roww['tweet_text']:
            s_name.append(roww['screen_name'])
            t_text.append(roww['tweet_text'])
            hashtag.append(row['hashtag'])
            sentiment.append(roww['vader_sentiment'])

h_df = pd.DataFrame()
h_df['screen_name'] = s_name
h_df['t_text'] = t_text
h_df['hashtag'] = hashtag
h_df['sentiment'] = sentiment

s_name = []
t_text = []
hashtag = []
sentiment = []


""" Assign sentiment to ats """
for index, row in ats_df.iterrows():
    for index, roww in tweets_df.iterrows():
        if row['at'] in roww['tweet_text']:
            s_name.append(roww['screen_name'])
            t_text.append(roww['tweet_text'])
            hashtag.append(row['at'])
            sentiment.append(roww['vader_sentiment'])

a_df = pd.DataFrame()
a_df['screen_name'] = s_name
a_df['t_text'] = t_text
a_df['at'] = hashtag
a_df['sentiment'] = sentiment


h_df.to_csv('****************')
a_df.to_csv('****************')






""" TRYING A LITTLE DIFFERENT APPROACH (FOR GROUPING, SCORING AND CONSOLIDATION)"""

groups = {
    'CMA Affiliate Societies': {
        'usernames': [ '@CAEP_Docs', '@CANeyeMDs', '@CanGastroAssn', '@CanGeriSoc',
                      '@CanPaedSociety', '@CanUrolAssoc', '@CAOT_ACE', '@CPA_APC', '@CAPHCTweets', '@CAPM_R',
                      '@CARadiologists', '@caro_acro_ca', '@CASEMACMSE',  '@CASUpdate', '@CdnDermatology',
                      '@CdnOrthoAssoc', '@CFMSFEMC', '@CNS_Update', '@CritCareSociety', '@CSIMSCMI',   '@CSPCP_SCMSP',
                      '@FAMS_SE', '@MSF_canada']
    },
    'CMA Associate Societies': {
        'usernames': ['@CAPE_Doctors', '@CIHI_ICIS', '@CMPAmembers', '@FMRAC_ca', '@FMWCanada', '@SRPCanada', '@MMJPRca', '@CAPD_ca']
    },
    'CMA Companies': {
        'usernames': ['@Financial_MD', '@CMA_Docs', '@Joule_CMA']
    },
    'Medical Schools': {
        'usernames': [ '@cpso_ca', '@MacHealthSci', '@MUNMed', '@uoftmedicine', '@DalMedSchool', '@uOttawaMed',
                       '@SchulichMedDent', '@UBCmedicine', '@UAlberta_FoMD', '@UCalgaryMed', '@USaskMedDean',
                       '@USherbrooke', '@UMontreal', '@McGillMed', '@facmedUL', '@UM_RadyFHS',
                       '@ResidentDoctors', '@OMSAofficial', '@ResidentDocsBC', '@para_ab', '@ResidentDocsSK']
    },
    'PTMAs': {
        'usernames': ['@DoctorsOfBC', '@SMA_docs', '@Albertadoctors', '@_nlma', '@Doctors_NS', '@OnCall4ON', '@OntariosDoctors', '@nb_docs', '@MSPEI_Docs', '@amquebec']
    }
}

startDate = datetime.datetime.now() - datetime.timedelta(days=7, hours=0, minutes=0, seconds=0)
endDate = datetime.datetime.now()


def get_tweets(usernames):
    tweets = []
    pd.options.display.max_colwidth = 200
    for user in usernames:
        tmpTweets = api.user_timeline(user)
        for tweet in tmpTweets:
            if tweet.created_at < endDate and tweet.created_at > startDate:
                tweets.append(tweet)

        while (tmpTweets[-1].created_at > startDate):
            tmpTweets = api.user_timeline(user, max_id=tmpTweets[-1].id)
            for tweet in tmpTweets:
                if tweet.created_at < endDate and tweet.created_at > startDate:
                    tweets.append(tweet)
    return tweets


def get_normalized_tweets(tweets):
    return {
        'id_list': [str(tweet.id) for tweet in tweets],
        'screen_name': [tweet.user.screen_name for tweet in tweets],
        'tweet_text': [tweet.text for tweet in tweets],
        'tweet_date': [tweet.created_at for tweet in tweets],
        'tweet_lang': [tweet.lang for tweet in tweets],
        'tweet_source': [tweet.source for tweet in tweets],
        'retweet': [tweet.retweet for tweet in tweets],
        'retweets_count': [tweet.retweet_count for tweet in tweets],
        'coordinates': [tweet.coordinates for tweet in tweets],
        'geo': [tweet.geo for tweet in tweets],
        'entities': [tweet.entities for tweet in tweets],
        'favorites_count': [tweet.favorite_count for tweet in tweets],
        'data_pull_date': [endDate for tweet in tweets],
        'followers_count': [tweet.user.followers_count for tweet in tweets],
        'friends_count': [tweet.user.friends_count for tweet in tweets],
        'geo_enabled': [tweet.user.geo_enabled for tweet in tweets],
        'time_zone': [tweet.user.time_zone for tweet in tweets],
        'location': [tweet.user.location for tweet in tweets]
    }


def vader_sentiment(tweets):
    """ Get Vader Sentiment scores """
    vs = []
    analyzer = SentimentIntensityAnalyzer()
    for tweet in tweets:
        a = analyzer.polarity_scores(tweet)['compound']
        vs.append(a)
    return vs


def textblob_sentiment(tweets):
    """ Get Textblob Sentiment scores (en/fr) """
    tb = []
    tb_fr = []
    for tweet in tweets:
        blob_tweet = TextBlob(tweet)
        tb.append(blob_tweet.sentiment.polarity)
        blob_tweet_fr = TextBlob(
            tweet, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        tb_fr.append(blob_tweet_fr.sentiment[0])
    return tb, tb_fr


def get_wordnet_pos(tag):
    """ Get wordnet part of speech from a tag. Return None if
    nothing matches. """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return None


def lemmatize_sentence(text):
    """ Lemmatize text while removing stopwords from the outcome """
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word, tag in pos_tag(word_tokenize(text)):
        if word in punctuation or word.lower() in STOPWORDS:
            continue
        pos = get_wordnet_pos(tag)
        if pos:
            lemmas.append(lemmatizer.lemmatize(word, pos=pos))
    return lemmas


def analyze(tweets):
    # print("All tweets: ", tweets)
    all_words = defaultdict(lambda: 0)

    tweet_lemmas = []
    for tweet in tweets:
        lemmas = lemmatize_sentence(tweet)
        for lemma in lemmas:
            all_words[lemma] += 1
        tweet_lemmas.append(lemmas)

    tweet_scores = []
    for lemmas in tweet_lemmas:
        score = 0
        for lemma in lemmas:
            score += all_words[lemma]
            # print("Lemma score: %s = %d" % (lemma, all_words[lemma]))
        tweet_scores.append(score)
        # print("Tweet scores: ", tweet_scores)

    return tweet_scores


def get_part(text, tag):
    return [
        token.replace(':', '').replace('.', '').replace(',', '').
        replace('_', '')
        for token in text.split() if token.startswith(tag)
    ]


def analyze_parts(tweets, tag):
    tweet_parts = []
    all_tags = defaultdict(lambda: 0)
    for tweet in tweets:
        parts = get_part(tweet, tag)
        for part in parts:
            all_tags[part] += 1
            tweet_parts.append(part)

    tag_scores = []
    score = 0
    for part in tweet_parts:
        score += all_tags[part]
        tag_scores.append(score)

    return all_tags


def get_group_data():
    for group_name, data in groups.items():
        tweets = get_tweets(data['usernames'])
        normalized_tweets = get_normalized_tweets(tweets)
        data['normalized_tweets'] = normalized_tweets

        tweet_texts = normalized_tweets['tweet_text']
        data['vader_sentiment'] = vader_sentiment(tweet_texts)
        tb, tb_fr = textblob_sentiment(tweet_texts)
        data['textblob_sentiment'] = tb
        data['textblob_sentiment_fr'] = tb_fr
        data['tweet_scores'] = analyze(tweet_texts)
        data['hashtags'] = analyze_parts(tweet_texts, '#')
        data['ats'] = analyze_parts(tweet_texts, '@')


def save_ats():
    at_counts = defaultdict(lambda: 0)
    for group_name, data in groups.items():
        for at, count in data['ats'].items():
            at_counts[at] += count

    with open('****************', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            'group',
            'at',
            'group_at_count',
            'total_at_count'
        ])
        for group_name, data in groups.items():
            for at, count in data['ats'].items():
                writer.writerow([
                    group_name,
                    at,
                    count,
                    at_counts[at]
                ])


def save_hashtags():
    hashtag_counts = defaultdict(lambda: 0)
    for group_name, data in groups.items():
        for hashtag, count in data['hashtags'].items():
            hashtag_counts[hashtag] += count

    with open('****************', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            'group',
            'hashtag',
            'group_hashtag_count',
            'total_hashtag_count'
        ])
        for group_name, data in groups.items():
            for hashtag, count in data['hashtags'].items():
                writer.writerow([
                    group_name,
                    hashtag,
                    count,
                    hashtag_counts[hashtag]
                ])

					
get_group_data()
save_hashtags()
save_ats()






# Optional (not used)			
# def save_tweets():
#     with open('tweets.csv', 'w', encoding='utf-8', newline='') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         writer.writerow([
#             'group',
#             'id_list',
#             'screen_name',
#             'tweet_text',
#             'tweet_date',
#             'tweet_lang',
#             'tweet_source',
#             'retweet',
#             'retweets_count',
#             'coordinates',
#             'geo',
#             'entities',
#             'favorites_count',
#             'data_pull_date',
#             'followers_count',
#             'friends_count',
#             'geo_enabled',
#             'time_zone',
#             'location',
#             'vader_sentiment',
#             'textblob_sentiment',
#             'textblob_sentiment_fr',
#             'tweet_score'
#         ])
#         for group_name, data in groups.items():
#             normalized_tweets = data['normalized_tweets']
#             for i, id in enumerate(normalized_tweets['id_list']):
#                 writer.writerow([
#                     group_name,
#                     id,
#                     normalized_tweets['screen_name'][i],
#                     normalized_tweets['tweet_text'][i],
#                     normalized_tweets['tweet_date'][i],
#                     normalized_tweets['tweet_lang'][i],
#                     normalized_tweets['tweet_source'][i],
#                     normalized_tweets['retweet'][i],
#                     normalized_tweets['retweets_count'][i],
#                     normalized_tweets['coordinates'][i],
#                     normalized_tweets['geo'][i],
#                     normalized_tweets['entities'][i],
#                     normalized_tweets['favorites_count'][i],
#                     normalized_tweets['data_pull_date'][i],
#                     normalized_tweets['followers_count'][i],
#                     normalized_tweets['friends_count'][i],
#                     normalized_tweets['geo_enabled'][i],
#                     normalized_tweets['time_zone'][i],
#                     normalized_tweets['location'][i],
#                     data['vader_sentiment'][i],
#                     data['textblob_sentiment'][i],
#                     data['textblob_sentiment_fr'][i],
#                     data['tweet_scores'][i]
#                 ])



# save_tweets()

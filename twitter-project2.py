import json
import twitter, tweepy
from operator import itemgetter

api_key = 'nv14VGDu5qZokarI4uUJTuiuJ'
api_secret_key = 'xZF3LJmC3PdzKP4Gm5e5MyR9MrJcZ7R72sz1mTtfDtINWn4Qxl'
access_token = '1498856690908733442-gZdhtqBok2PAbRnRRV3IDbAMJagv63'
access_token_secret = 'SIdbbReIdG2k62tsSzRAO8bN22EmDI9UDGtcdjmwc7bF6'
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAImjcwEAAAAA4x8nTOt23cuR86pCSfutdLEAFt4%3DI2XWC71LVsMxwx4THuaaKRTr8Ld3C3FrEWuTH1SSPrl6lAJFgo'


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
kor_stop_words = stopwords.words('english') + ['.', ',', '--', '\'s', '?', '!', ')', '(', ':', '\'', '\'re', '"', '-', '}', '{', u'—', 'rt', 'http', 'https', 't', 'co', '@', '#']


import pandas as pd
from collections import Counter
from konlpy.tag import Okt

auth = twitter.oauth2.OAuth2(bearer_token=BEARER_TOKEN)
twitter_api = twitter.Twitter(auth=auth)
# for keyword in ["경기도지사", "국민의힘", "국힘", "김동연", "김은혜", "민주당", "지방선거"]:
# for keyword in ["서울시장", "오세훈", "송영길", "부산시장", "계양", "이재명", "윤형선"]:
for keyword in ["정치", "선거", "지지율", "윤석열", "김건희", "대통령", "청와대", "국회", "장관", "한동훈", "총리", "한덕수", "안철수", "이준석", "박지현", "윤호중", "계파", "당권", "사퇴"]:
    print(f"query: {keyword}")
    q = keyword
    # start_time = "2022-05-20T12:00:00Z"
    # end_time = "2022-05-21T12:00:00Z"
    count = 100

    # # cancelled idea
    # client = tweepy.Client(bearer_token=BEARER_TOKEN, consumer_key=api_key, consumer_secret=api_secret_key, access_token=access_token, access_token_secret=access_token_secret)
    # for _ in range(2):
    #     search_results = client.search_recent_tweets(query=q, tweet_fields=['context_annotations', 'created_at'], start_time=start_time, end_time=end_time, max_results=count)
    #     for tweet in search_results.data:
    #         print(tweet.created_at)
    #         print(tweet.text)
    #         # if len(tweet.context_annotations) > 0:
    #         #     print(tweet.context_annotations)

    for start_time in ["2022-05-26", "2022-05-27", "2022-05-28", "2022-05-29", "2022-05-30", "2022-05-31", "2022-06-01", "2022-06-02"]:
        print(f"Tweet time: {start_time}")
        search_results = twitter_api.search.tweets(q=q, count=count, result_type='recent', until=start_time)
        statuses = search_results['statuses']
        for _ in range(5):
            print("Length of statuses", len(statuses))
            query = {}
            try:
                next_results = search_results['search_metadata']['next_results']
                parameters = next_results[1:].split('&')
                for s in parameters:
                    key_value = s.split('=')
                    key = key_value[0]
                    value = key_value[1]
                    query[key] = value
            except KeyError:
                print(search_results['search_metadata'])
                break
            # kwargs = dict([kv.split('=') for kv in next_results[1:].split('&')])
            query['q'] = q
            # print(query)
            search_results = twitter_api.search.tweets(**query)
            statuses += search_results['statuses']

        status_texts = [status['text'] for status in statuses]
        # # screen_names = [status['user']['screen_name'] for status in statuses]
        # # hashtags = [hashtag['text'] for status in statuses for hashtag in status['entities']['hashtags']]
        # # created_ats = [status['created_at'] for status in statuses]
        # # words = [w for t in status_texts for w in t.split()]

        # # popular_retweets = [(status['retweet_count'], status['retweeted_status']['user']['screen_name'], status['text']) for status in statuses if 'retweeted_status' in status]
        # # popular_retweets = list(set(popular_retweets))
        # # top_retweets = sorted(popular_retweets, key=itemgetter(0), reverse=True)

        # # user_created_ats = [status['user']['created_at'] for status in statuses]
        # # user_statuses = [status['user']['statuses_count'] for status in statuses]
        # # _user = [(status['user']['name'], status['user']['followers_count']) for status in statuses]
        # # user_followers = sorted(list(set(_user)), key=itemgetter(1), reverse=True)

        # # _hashtags = [(hash_name, hashtags.count(hash_name)) for hash_name in set(hashtags)]
        # # top_hashtags = sorted(_hashtags, key=itemgetter(1), reverse=True)

        # print()
        # print(status_texts[:10])
        # # print()
        # # print(screen_names[:10])
        # # print()
        # # print(hashtags[:10])
        # # print("Created at")
        # # print(created_ats[:10])
        # # print()
        # # print(words[:5])
        # # print()
        # # print(top_retweets[:10])
        # # print("User created at")
        # # print(user_created_ats[:10])
        # # print("User tweets")
        # # print(user_statuses[:10])
        # # print("User followers")
        # # print(user_followers[:10])
        # # print("Top hashtags")
        # # print(top_hashtags[:10])
        # # print("Top retweets")
        # # for retweet in top_retweets[:10]:
        # #     print(retweet)


        okt = Okt()
        tweet_words = []
        for tweet in status_texts:
            pos_tokens = okt.pos(tweet)
            noun_adj_list = []
            for word, tag in pos_tokens:
                if tag in ['Noun', 'Adjective'] and word not in kor_stop_words:
                    noun_adj_list.append(word)
                    tweet_words.append([tweet.replace("\n", " ").replace(",", " "), word])
            # print(noun_adj_list)
            # # counts = Counter(noun_adj_list)
            # # tags = counts.most_common(300)
            # # print(tags)

        data_frame = pd.DataFrame(tweet_words)
        # file_path = "../election_data/" + keyword + "_" + start_time + ".csv"
        file_path = "../politics_data/" + keyword + "_" + start_time + ".csv"
        data_frame.to_csv(file_path, header=False, index=False)

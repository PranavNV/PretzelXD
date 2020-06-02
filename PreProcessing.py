import pandas as pd
import re
from pandas import DataFrame
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

#slangs are present in 'slangs' file
ser = pd.read_table('slangs', sep = "  -  ", header = None)

def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_mentioned(tweet):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def ekphrasis_pre(tweet):
    return ' '.join(text_processor.pre_process_doc(tweet))


def stopwords_rem(tweets):
    querywords = tweets.split()
    resultwords  = [word for word in querywords if word.lower() not in stopw]
    result = ' '.join(resultwords)
    return result

def stopwords_rem1(tweets):
    querywords = tweets.split()
    resultwords  = [word for word in querywords if word.lower() not in stop]
    result = ' '.join(resultwords)
    return result

def remove_digits(tweets):
    res = ''.join([i for i in tweets if not i.isdigit()]) 
    return res


#slang function
def slang_remove(tweet):
  tweet = tweet.split(" ")
  loc = 0
  for _str in tweet:
    for j in range(0,5385):
      _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
      if _str == ser[0][j]:
        tweet[loc] = ser[1][j]
    loc = loc+1
  return ' '.join(tweet)


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

stopw = ['<hashtag>','</hashtag>','@','<','>','<allcaps>','</allcaps>','<user>','...','..','.','-','+','#','!','/','<emphasis>','<elongated>','<repeated>', ',',"'","'"]
stop = ['<hashtag>','</hashtag>', '[',']','[]', ',',"'","'"]


with open("us_train.text", encoding="utf8") as doc:
    train_text = doc.read().splitlines()

with open("us_trial.text", encoding="utf8") as doc:
    train_text = doc.read().splitlines()

df_train = pd.DataFrame(train_text)
df_trial = pd.DataFrame(train_text)

frames = [df_train, df_trial]
df = pd.concat(frames)
df.columns = ['text']



df['hashtags'] = df.text.apply(find_hashtags)
hashtags_list_df = df.loc[
                       df.hashtags.apply(
                           lambda hashtags_list: hashtags_list !=['sdfsdfds']
                       ),['hashtags']]

flattened_hashtags_df = pd.DataFrame(
    [hashtag for hashtags_list in hashtags_list_df.hashtags
    for hashtag in hashtags_list],
    columns=['hashtag'])


popular_hashtags = flattened_hashtags_df.groupby('hashtag').size()\
                                        .reset_index(name='counts')\
                                        .sort_values('counts', ascending=False)\
                                        .reset_index(drop=True)

# take hashtags which appear at least this amount of times
min_appearance = 5
# find popular hashtags - make into python set for efficiency
popular_hashtags_set = set(popular_hashtags[
                           popular_hashtags.counts>=min_appearance
                           ]['hashtag'])

# make a new column with only the popular hashtags


df['popular_hashtags'] = hashtags_list_df.hashtags.apply(
            lambda hashtag_list: [hashtag for hashtag in hashtag_list
                                  if hashtag in popular_hashtags_set])


df['text'] = df.text.apply(slang_remove)
df['text'] = df.text.apply(ekphrasis_pre)
df['text'] = df.text.apply(stopwords_rem)


df['popular_hashtags'] = df.popular_hashtags.apply(
            lambda hashtag_list: ekphrasis_pre(str([hashtag for hashtag in hashtag_list])))

df['popular_hashtags'] = df['popular_hashtags'].astype('str')
#df['popular_hashtags']= df['popular_hashtags'].apply(slang_remove)
df['popular_hashtags']= df['popular_hashtags'].apply(stopwords_rem1)
df['popular_hashtags'] = df['popular_hashtags'].popular_hashtags.apply(remove_digits)

df.to_csv(r'out.csv')

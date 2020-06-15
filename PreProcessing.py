import pandas as pd
import re
from pandas import DataFrame
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()

#slangs are present in 'slangs' file
ser = pd.read_table('slangs', sep = "  -  ", header = None)

# define punctuation
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

def remove_punct(tweet):
    '''This function will remove punctuations'''
	no_punct = ""
	for char in tweet:
  		 if char not in punctuations:
      			 no_punct = no_punct + char
	return no_punct

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

def cleaner(tweet):
    tweet = tweet[2:]
    return tweet

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

stopw = ['<hashtag>','<url>','<time>','<date>','<number>','</hashtag>','@','<','>','<allcaps>','</allcaps>','<user>','...','..','.','-','+','#','!','/','<emphasis>','<elongated>','<repeated>', ',',"'","'",'â€¦']
stop = ['<hashtag>','</hashtag>','<url>','<time>', '[',']','[]', ',',"'","'",'<number>','<date>','â€¦']

with open("tweets_combolabels.labels", encoding="utf8") as doc:
    labels = doc.read().splitlines()

labels = pd.DataFrame(labels)

with open("tweets_combo.text", encoding="utf8") as doc:
    data = doc.read().splitlines()

df = pd.DataFrame(data)
df.columns = ['text']
df['labels'] = labels

#remove the noise character
#df['text'] = df.text.progress_apply(cleaner)

#remove retweets
df = df[~df.text.str.startswith('RT')]
df = df[~df.text.str.startswith('rt')]


df['hashtags'] = df.text.progress_apply(find_hashtags)
hashtags_list_df = df.loc[
                       df.hashtags.progress_apply(
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
min_appearance = 1
# find popular hashtags - make into python set for efficiency
popular_hashtags_set = set(popular_hashtags[
                           popular_hashtags.counts>=min_appearance
                           ]['hashtag'])

# make a new column with only the popular hashtags


df['popular_hashtags'] = hashtags_list_df.hashtags.progress_apply(
            lambda hashtag_list: [hashtag for hashtag in hashtag_list
                                  if hashtag in popular_hashtags_set])



df['popular_hashtags'] = df.popular_hashtags.progress_apply(
            lambda hashtag_list: ekphrasis_pre(str([hashtag for hashtag in hashtag_list])))

df['popular_hashtags'] = df['popular_hashtags'].astype('str')
#df['popular_hashtags']= df['popular_hashtags'].progress_apply(slang_remove)
df['popular_hashtags']= df['popular_hashtags'].progress_apply(stopwords_rem1)
df['popular_hashtags'] = df['popular_hashtags'].progress_apply(remove_digits)
df['popular_hashtags'] = df['popular_hashtags'].progress_apply(remove_punct)



df['text'] = df.text.progress_apply(slang_remove)
df['text'] = df.text.progress_apply(ekphrasis_pre)
df['text'] = df.text.progress_apply(stopwords_rem)
df['text'] = df.text.progress_apply(remove_digits)
df['text'] = df.text.progress_apply(remove_punct)

df.to_csv(r'out.csv')

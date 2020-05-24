import pandas as pd

#slangs are present in "slangs" file
ser = pd.read_table('slangs', sep = "  -  ", header = None)
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

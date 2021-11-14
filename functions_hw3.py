## FUNCTIONS FOR QUESTION 1##

def getText(parent):
    return ''.join(parent.find_all(text=True, recursive=False)).strip()

def article_parser(article_path, soup):
        anime_title = soup.title.text.strip()
        anime_type = ""
        try:
            anime_type = getText(soup.find('span', text="Type:").parent.a)
        except:
            pass
        anime_num_episodes = ""
        try:
            anime_num_episodes = getText(soup.find('span', text="Episodes:").parent)
        except:
            pass
        anime_aired = ""
        try:
            anime_aired = getText(soup.find('span', text="Aired:").parent).split(" to ")
        except:
            pass
        anime_status = ""
        try:
            anime_status = getText(soup.find('span', text="Status:").parent)
        except:
            pass
        anime_score = ""
        try:
            anime_score = getText(soup.find('span', text="Score:").parent.find_all('span', {'class': 'score-label'})[0])
        except:
            pass
        anime_users = ""
        try:
            anime_users = getText(soup.find('span', text="Members:").parent)
        except:
            pass
        anime_rank = ""
        try:
            anime_rank = getText(soup.find('span', text="Ranked:").parent)
        except:
            pass
        anime_popularity = ""
        try:
            anime_popularity = getText(soup.find('span', text="Popularity:").parent)
        except:
            pass
        anime_description = ""
        try:
            anime_description = getText(soup.find('h2', text="Synopsis").parent.parent.p)
        except:
            pass
        anime_related = []
        try: 
            anime_related = list(set(map(getText,soup.find('h2', text="Related Anime").parent.parent.tr.find_all(lambda tag:tag.name == "a" and tag.href != "")))) 
        except: 
            pass
        anime_characters = []
        try:
            anime_characters = list(filter(None,list(map(lambda tr: re.split('\n+', tr.find_all('td')[1].text.strip())[0], soup.find('h2', text="Characters & Voice Actors").find_next('div').find_all('tr')))))
            anime_characters = [e.split(", ") for e in anime_characters]
        except:
            pass
        anime_voices = []
        try:
            anime_voices = list(filter(None,list(map(lambda tr: re.split('\n+', tr.find_all('td')[0].text.strip())[0], soup.find('h2', text="Characters & Voice Actors").find_next('div').find_all('tr')))))
            anime_voices = [e.split(", ") for e in anime_voices]
        except:
            pass
        anime_staff = []
        try:
            anime_staff = list(map(lambda tr: [re.split("\n+", tr.text.strip())[0].split(', '), re.split("\n+", tr.text.strip())[1]], soup.find('h2', text="Staff").find_next('div').find_all('tr')))
        except:
            pass
        l = [anime_title,
                anime_type,
                anime_num_episodes,
                anime_aired,
                anime_status,
                anime_score,
                anime_users,
                anime_rank,
                anime_popularity,
                anime_description,
                anime_related,
                anime_characters,
                anime_voices,
                anime_staff]
        return l
    
    
##FUNCTIONS FOR QUESTION 2##

ps = PorterStemmer()
stopwords = stopwords.words()
def isNaN(string):
    return string != string
def preprocess(sentence):
    if sentence == "" or isNaN(sentence):
        return []
    text_tokens = nltk.word_tokenize(sentence)
    tokens_without_sw = [ps.stem(word) for word in text_tokens if not word in stopwords and word.isalnum()]
    return tokens_without_sw

def tf(term, desc_preprocessed):
    return desc_preprocessed.count(term)/ len(desc_preprocessed)
def idf(term, inverseindex_dict):
    return np.log(len(inverseindex_dict)/len(inverseindex_dict[term]))
def tf_idf(term, desc_preprocessed, inverseindex_dict):
    x = tf(term, desc_preprocessed) * idf(term, inverseindex_dict)
    return x

def cosine_similarity_dict(dict1, dict2):
    s = set(dict1.keys()).intersection(set(dict2.keys()))
    c = 0
    norm1 = 0
    norm2 = 0
    for i in s:
        c += dict1[i] * dict2[i]
    for i in dict1.values():
        norm1 += np.power(i, 2)
    norm1 = np.sqrt(norm1)
    
    for i in dict2.values():
        norm2 += np.power(i, 2)
    norm2 = np.sqrt(norm2)
    
    return c / (norm1 * norm2)


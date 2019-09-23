from convokit.model import Corpus, User
from convokit.transformer import Transformer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re

class Genderromantic(Transformer):

    def __init__(self):
        pass

    @staticmethod
    def genderDictionary(arg):
        name_to_gender = {}
        for user in arg.iter_users():
            if user.name not in name_to_gender:
                name_to_gender[user.name] = user.meta['gender']
        return name_to_gender

    def transform(self, corpus: Corpus):
        name_to_gender = genderDictionary(corpus)
        for utt in corpus.iter_utterances():
            speaker_name = utt.user.name
            speaker_gender = utt.user.meta['gender']
            gender_is_female = speaker_gender == 'female'
            character_entities = utt.meta['character_entities']
            
            contains_romantic = False
            male_about_female = False
            female_about_male = False
            
            romantic_words = [ps.stem(line.rstrip('\n')) for line in open('RomanticWords.txt', 'r')]
            tokens = utt.meta['tokens']
            for token in tokens:
                for word in token:
                    word=re.sub(r'[^\w\s]', '', word)
                    if ps.stem(word) in romantic_words:
                        contains_romantic = True
                        break
            
            if character_entities is not None:        
                for ces in character_entities:
                    if len(ce) == 0:
                        continue
                    for ce in ces:
                        if len(ce) == 0 or ce[2] == speaker_name or ce[2][0] == '#':
                            continue
                        if ce[2] not in name_to_gender:
                            if ce[2].startswith('Man'):
                                male_about_female = False
                                female_about_male = True & gender_is_female
                            elif ce[2].startswith('Woman'):
                                male_about_female = True & gender_is_female
                                female_about_male = False
                        elif 'female' in name_to_gender[ce[2]]:
                            male_about_female = True & gender_is_female
                            female_about_male = False
                        elif 'male' in name_to_gender[ce[2]]:
                            male_about_female = False
                            female_about_male = True & gender_is_female
            utt.add_meta('female_about_male', female_about_male)
            utt.add_meta('male_about_female', male_about_female)
            utt.add_meta('contains_romantic', contains_romantic)
        return corpus
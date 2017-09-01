# This example uses the question typology court corpus to reproduce figures 1A and 1B from
#   the asking too much paper (http://www.cs.cornell.edu/~cristian/Asking_too_much.html).
#
# The plots answer these questions:
# - ?

import os
import pkg_resources
import numpy as np

from convokit import Utterance, Corpus, QuestionTypology, download

DEBUG = False
DEBUG_DIR = '/Users/ishaanjhaveri/Google_Drive/git/Cornell-Conversational-Analysis-Toolkit/datasets/parliament-corpus/downloads_p'

#Initialize QuestionTypology class
data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads', 'parliament-spacy') if not DEBUG else DEBUG_DIR
# corpus = Corpus(filename=download('parliament-corpus')) if not DEBUG else Corpus(filename=DEBUG_DIR+'/full.json')
corpus = Corpus(filename="/Users/ishaanjhaveri/Google_Drive/git/Cornell-Conversational-Analysis-Toolkit/datasets/parliament-corpus/full.json")
questionTypology = QuestionTypology(corpus, data_dir)

#Output required data representations

# Need to fill in

#Output Figure 1A

# users_with_specific_affiliation = corpus.users(lambda u: u.info["is-admin"])
# [where instead of admins you would select users with a specific the affiliation]

# and
 
# log odds ratio (http://dept.stat.lsa.umich.edu/~kshedden/Python-Workshop/stats_calculations.html) 

# and of course they typology counts you would get using QuestionTypology
# questionTypology.display_question_type_log_odds_graph()

question_answer_pair = [
{
    "asked_tbl": True,
    "date": "1995-06-27",
    "govt": "major",
    "id": "1995-06-27a.670.6",
    "is_answer": False,
    "is_pmq": False,
    "is_question": True,
    "is_topical": False,
    "len_followups": 0,
    "major_name": "Education",
    "minor_name": "Parental Choice",
    "num_interjections": 0,
    "official_name": "Education",
    "pair_idx": "1995-06-27.1.0",
    "root": "1995-06-27a.670.6",
    "text": "My hon Friend is aware that Janet Dawson from his Department has visited the island to look at the admissions policy, and that the education officer has publicly stated that there was an error in the school numbers. What advice can my hon Friend offer about the coming year to the parents, who are frustrated, the angry school governors and head teachers? Does he agree that the promise of a review for the next year just will not do?",
    "user": "person/22639",
    "user-info": {
      "is_incumbent": False,
      "is_oppn": False,
      "party": "<UNKNOWN>"
    }
  },
  {
    "asked_tbl": True,
    "date": "1995-06-27",
    "govt": "major",
    "id": "1995-06-27a.670.7",
    "is_answer": True,
    "is_pmq": False,
    "is_question": False,
    "is_topical": False,
    "len_followups": 0,
    "major_name": "Education",
    "minor_name": "Parental Choice",
    "num_interjections": 0,
    "official_name": "Education",
    "pair_idx": "1995-06-27.1.0",
    "reply-to": "1995-06-27a.670.6",
    "root": "1995-06-27a.670.6",
    "text": "I obviously have considerable sympathy for my hon Friend's point and those made on behalf of the parents whom he represents. I can confirm that my officials have been in contact with the local education authority, although none of the cases raised with us to date provides scope for action by the Department. My hon Friend knows that the school must admit up to its minimum number, but it is entirely open to the LEA to agree higher admission numbers with that school. If it wishes to admit more pupils, and it cannot get the LEA to agree to that, it can write to my right hon Friend the Secretary of State.",
    "user": "person/22600",
    "user-info": {
      "is_incumbent": False,
      "is_oppn": False,
      "party": "<UNKNOWN>"
    }
  }
]

# questionTypology.classify_question(question_answer_pair)

print('5 examples for type 1-8:')
for i in range(8):
    questionTypology.display_motifs_for_type(i)
    questionTypology.display_answer_fragments_for_type(i)
    questionTypology.display_questions_for_type(i)

questionTypology.display_question_type_log_odds_graph()

# the Parliament script would create this dictionary and print samples of each contents (i.e., recreate Table 1, of course it does not need to
# be the exact same examples).   It would also use the function to assign types to a few new and existing questions.

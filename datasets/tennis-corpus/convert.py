# convert into toolkit json format
import json
import pandas as pd
import numpy as np

# Keys in original metadata.tsv
#question_text_idx
#answer_text_idx
#player
#question_text
#answer_text
#gender
#ranking
#date
#pair_idx
#match_id
#opponent
#result
#stage
#tournament
#tournament_type


# Keys in new full.json
ID = "id" # equal to question_text_idx if current utterance is a question
        # equal to answer_text_idx if current utterance is an answer
ROOT = "root" # this will always be equal to question_text_idx
REPLY_TO = "reply-to" # non existant if this utterance is a question,
                    # equal to question_text_idx if this is an answer
USER = "user" # equal to "REPORTER" if current utterance is a question
            # equal to player if current utterance is an answer
TEXT = "text" # equal to question_text if current utterance is a question
        # equal to answer_text if current utterance is an answer

IS_QUESTION = "is_question" # for redundancy
IS_ANSWER = "is_answer" # for redundancy
DATE = "date"
MATCH_ID = "match_id"
PAIR_IDX = "pair_idx"
OPPONENT = "opponent"
RESULT = "result"
STAGE = "stage"
TOURNAMENT = "tournament"
TOURNAMENT_TYPE = "tournament_type"

USER_INFO = "user-info" # container for extra information

# Keys within user-info
GENDER = "gender"
RANKING = "ranking"

utterances = []
question_df = pd.read_csv('metadata.tsv', index_col=0, sep='\t')
i = 0
for row in question_df.itertuples():
    i += 1
    question_utter = {}
    answer_utter = {}

    question_utter[ID] = row.question_text_idx
    answer_utter[ID] = row.answer_text_idx

    question_utter[ROOT] = row.question_text_idx
    answer_utter[ROOT] = row.question_text_idx

    answer_utter[REPLY_TO] = row.question_text_idx

    question_utter[USER] = row.player
    answer_utter[USER] = "REPORTER"

    question_utter[TEXT] = row.question_text
    answer_utter[TEXT] = row.answer_text

    question_utter[IS_QUESTION] = True
    answer_utter[IS_QUESTION] = False

    question_utter[IS_ANSWER] = False
    answer_utter[IS_ANSWER] = True

    question_utter[DATE] = row.date
    answer_utter[DATE] = row.date

    question_utter[MATCH_ID] = int(row.match_id)
    answer_utter[MATCH_ID] = int(row.match_id)

    question_utter[PAIR_IDX] = row.Index
    answer_utter[PAIR_IDX] = row.Index

    question_utter[OPPONENT] = row.opponent
    answer_utter[OPPONENT] = row.opponent

    question_utter[RESULT] = int(row.result)
    answer_utter[RESULT] = int(row.result)

    question_utter[STAGE] = row.stage
    answer_utter[STAGE] = row.stage

    question_utter[TOURNAMENT] = row.tournament
    answer_utter[TOURNAMENT] = row.tournament

    question_utter[TOURNAMENT_TYPE] = row.tournament_type
    answer_utter[TOURNAMENT_TYPE] = row.tournament_type

    question_utter[USER_INFO] = {
        GENDER: row.gender,
        RANKING: row.ranking,
    }
    answer_utter[USER_INFO] = {
        GENDER: row.gender,
        RANKING: row.ranking,
    }

    utterances.append(question_utter)
    utterances.append(answer_utter)

json.dump(utterances, open("full.json", "w"), indent=2, sort_keys=True)

print("Done", i, "pairs")

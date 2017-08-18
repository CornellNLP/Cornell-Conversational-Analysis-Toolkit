# convert into toolkit json format
import json
import pandas as pd
import numpy as np

# Keys in original metadata.tsv
# question_text_idx
# answer_text_idx
# asker
# answerer
# question_text
# answer_text

# date
# govt
# pair_idx (renamed to Index by pandas)
# is_pmq
# is_topical
# asked_tbl
# len_followups
# official_name
# major_name
# minor_name
# num_interjections

# party_asker
# party_answerer
# is_incumbent_asker
# is_incumbent_answerer
# is_oppn_asker
# is_oppn_answerer

# Keys in new full.json
ID = "id" # equal to question_text_idx if current utterance is a question
        # equal to answer_text_idx if current utterance is an answer
ROOT = "root" # this will always be equal to question_text_idx
REPLY_TO = "reply-to" # non existant if this utterance is a question,
                    # equal to question_text_idx if this is an answer
USER = "user" # equal to asker if current utterance is a question
            # equal to answerer if current utterance is an answer
TEXT = "text" # equal to question_text if current utterance is a question
        # equal to answer_text if current utterance is an answer

IS_QUESTION = "is_question" # for redundancy
IS_ANSWER = "is_answer" # for redundancy
DATE = "date"
GOVT = "govt"
PAIR_IDX = "pair_idx"
IS_PMQ = "is_pmq"
IS_TOPICAL = "is_topical"
ASKED_TBL = "asked_tbl"
LEN_FOLLOWUPS = "len_followups"
OFFICIAL_NAME = "official_name"
MAJOR_NAME = "major_name"
MINOR_NAME = "minor_name"
NUM_INTERJECTIONS = "num_interjections"

USER_INFO = "user-info" # container for extra information

# Keys within user-info
PARTY = "party" # to represent party_asker or party_answerer
IS_INCUMBENT = "is_incumbent" # to represent is_incumbent_asker or is_incumbent_answerer
IS_OPPN = "is_oppn" # to represent is_oppn_asker or is_oppn_answerer

utterances = []
question_df = pd.read_csv('metadata_10.tsv', index_col=0, sep='\t')
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

    question_utter[USER] = row.asker
    answer_utter[USER] = row.answerer

    question_utter[TEXT] = row.question_text
    answer_utter[TEXT] = row.answer_text

    question_utter[IS_QUESTION] = True
    answer_utter[IS_QUESTION] = False

    question_utter[IS_ANSWER] = False
    answer_utter[IS_ANSWER] = True

    question_utter[DATE] = row.date
    answer_utter[DATE] = row.date

    question_utter[GOVT] = row.govt
    answer_utter[GOVT] = row.govt

    question_utter[PAIR_IDX] = row.Index
    answer_utter[PAIR_IDX] = row.Index

    question_utter[IS_PMQ] = bool(row.is_pmq)
    answer_utter[IS_PMQ] = bool(row.is_pmq)

    question_utter[IS_TOPICAL] = bool(row.is_topical)
    answer_utter[IS_TOPICAL] = bool(row.is_topical)

    question_utter[ASKED_TBL] = bool(row.asked_tbl)
    answer_utter[ASKED_TBL] = bool(row.asked_tbl)

    question_utter[LEN_FOLLOWUPS] = int(row.len_followups)
    answer_utter[LEN_FOLLOWUPS] = int(row.len_followups)

    question_utter[OFFICIAL_NAME] = row.official_name
    answer_utter[OFFICIAL_NAME] = row.official_name

    question_utter[MAJOR_NAME] = row.major_name
    answer_utter[MAJOR_NAME] = row.major_name

    question_utter[MINOR_NAME] = row.minor_name
    answer_utter[MINOR_NAME] = row.minor_name

    question_utter[NUM_INTERJECTIONS] = int(row.num_interjections)
    answer_utter[NUM_INTERJECTIONS] = int(row.num_interjections)

    question_utter[USER_INFO] = {
        PARTY: row.party_asker,
        IS_INCUMBENT: bool(row.is_incumbent_asker),
        IS_OPPN: bool(row.is_oppn_asker)
    }
    answer_utter[USER_INFO] = {
        PARTY: row.party_answerer,
        IS_INCUMBENT: bool(row.is_incumbent_answerer),
        IS_OPPN: bool(row.is_oppn_answerer)
    }

    utterances.append(question_utter)
    utterances.append(answer_utter)

json.dump(utterances, open("full.json", "w"), indent=2, sort_keys=True)

print("Done", i, "pairs")

# convert into toolkit json format
import json

MaxUtterances = -1

KeyId = "id"
KeyUser = "user"
KeyConvoRoot = "root"
KeyReplyTo = "reply-to"
KeyUserInfo = "user-info"  # can store any extra data
import time
import datetime

KeyText = "text"

with open("supreme.gender.txt", "r", encoding="utf-8") as f:
    genders = {}
    for line in f:
        line = line[:-1]
        name, gender = line.split(" +++$+++ ")
        genders[name.lower()] = gender

convo_id = 0
last_utterance_id = None
u = {}
with open("supreme.conversations.dat", "r", encoding="utf-8") as f:
    utterances = []
    count = 0
    for line in f:
        if count % 1000 == 0: print(count)
        line = line[:-1]
        if line:
            fields = line.split(" +++$+++ ")
            if len(fields) == 8:
                user = fields[3].strip().lower()
                if user == "justice roberts": user = "chief " + user # fix typo
                if user.endswith("kenned"): user += "y" # fix typo

                case = fields[0].strip() 
                is_justice = fields[4].strip() == "JUSTICE"

                #user += "(case:" + fields[0].strip() + ")"
                #is_justice = fields[4].strip() == "JUSTICE"
                #if is_justice:
                #    if fields[5].strip() == fields[6].strip(): # favorable justice
                #        user += "{justice-fav}"
                #    else:
                #        user += "{justice-unfav}"

                is_reply = fields[2].strip() == "TRUE"
                if not is_reply:
                    convo_id += 1
                
                d = {
                    KeyId: fields[1],
                    KeyUser: user,
                    KeyConvoRoot: convo_id,
                    KeyText: fields[7],
                    KeyUserInfo: {
                        "case": case,
                        "gender": genders[user],
                        "is-justice": is_justice,
                        "side": fields[6].lower(),
                    }
                }
                if is_justice:
                    d[KeyUserInfo]["justice-vote"] = fields[5].lower()
                    d[KeyUserInfo]["justice-is-favorable"] = \
                            fields[5] == fields[6]

                if is_reply:
                    d[KeyReplyTo] = last_utterance_id
                    d[KeyConvoRoot] = user # \
                        #+ "->" + (
                        #"[justices]" if \
                        #u[last_utterance_id][KeyUser].endswith("fav}") else \
                        #"[lawyers]")
                u[fields[1]] = d
                utterances.append(d)

                last_utterance_id = fields[1]
                
                count += 1
                #if MaxUtterances > 0 and count > MaxUtterances:
                #    break

#udict = {ut["id"]: ut for ut in utterances}
#for i, ut in enumerate(utterances):
#    if KeyReplyTo in ut:
#        target = udict[ut[KeyReplyTo]][KeyUser]
#        if target.endswith("{justice-fav}"):
#            ut[KeyConvoRoot] = ut[KeyUser] + "->{justice-fav}"
#        elif target.endswith("{justice-unfav}"):
#            ut[KeyConvoRoot] = ut[KeyUser] + "->{justice-unfav}"  # target groups
#        #ut[KeyConvoRoot] = target  # target groups -- experimental
#        #ut[KeyConvoRoot] = ut[KeyUser]  # speaker groups
#        utterances[i] = ut
#    else:
#        del utterances[i][KeyConvoRoot]
        
if MaxUtterances > 0:
    #import random
    #random.shuffle(utterances)
    utterances = utterances[-MaxUtterances:]
json.dump(utterances, open("full.json", "w"), indent=2,
          sort_keys=True)

#print(len(usernames), len(usernames_cased))
print("Done")


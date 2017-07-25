# This example uses the supreme court corpus to compute some simple information:
# - Which justices coordinate the most to others?
# - Which justices are coordinated to the most?

import convokit

# set up corpus 
corpus = convokit.Corpus(filename=convokit.download("supreme-corpus"))
coord = convokit.Coordination(corpus)

# get set of all justices
justices = corpus.users(lambda user: user.info["is-justice"])
# get set of all users
everyone = corpus.users()

# compute coordination from each justice to everyone
print("Justices, ranked by how much they coordinate to others:")
justices_to_everyone = coord.score(justices, everyone)
for justice, score in sorted(justices_to_everyone.averages_by_user().items(),
    key=lambda x: x[1], reverse=True):
    print(justice.name, round(score, 5))
print()

# compute coordination from everyone to each justice
print("Justices, ranked by how much others coordinate to them:")
everyone_to_justices = coord.score(everyone, justices)
for justice, score in sorted(everyone_to_justices.averages_by_user().items(), 
    key=lambda x: x[1], reverse=True):
    print(justice.name, round(score, 5))
print()

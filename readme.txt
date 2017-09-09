UK Parliament Question Period Dataset
======================================

This dataset contains a collection of questions and answers along with relevant metadata for questions asked during the daily Question Period of the British House of Commons, from May 1979 to December 2016. For each question-answer pair, the text of the question and answer is given, along with information such as the asker and answerer's name, their party affiliation, whether they are a minister or not and some other information that is explicated below. Note that follow ups are not included, merely the question and immediate answer. In total there are 216,894 question-answer pairs in our data, occurring over 4,776 days and 6 prime-ministerships. The questions cover 1,975 different askers, 1,066 different answerers, and a variety of government departments with respon- sibilities ranging from defense to transport.

URL: www.url.com TODO: what is the correct URL?
Authors: Justine Zhang <jz727@cornell.edu>
     Arthur Spirling <arthur.spirling@nyu.edu>
     Cristian Danescu-Niculescu-Mizil <cristian@cs.cornell.edu>
Contact: Justine Zhang <jz727@cornell.edu>
Last updated: September 8, 2017
Version: 1.0

The dataset is further described in our paper:
  Asking too much? The rhetorical role of questions in political discourse
  Justine Zhang, Arthur Spirling, Cristian Danescu-Niculescu-Mizil 

Files
-----

* parliament-corpus.json - a JSON file containing the dataset
* README.txt - this readme

Code forthcoming. TODO: Should I include the code files here?

Question Period description
----------------------------

The House of Commons holds weekly, moderated question periods, in which MPs of all affiliations take turns to ask questions to (and theoretically receive answers from) government ministers for each department regarding their specific domains. Such events are a primary way in which legislators hold senior policy-makers responsible for their decisions. In practice, beyond narrow requests for information about specific policy points, MPs use their questions to critique or praise the government, or to self-promote; indeed, certain sessions, such as Questions to the Prime Minister, have gained renown for their partisan clashes, often fueled by the (mis)handling of a current crisis.

Format
------

The dataset is a JSON file:

It is structured as a dictionary with two entries for each of TODO: (should I put 216,894, 217,318 or 199,861 here?) question-answer pairs.

This is an example of one such question, and the corresponding answer entry, with fields explained:

# An example question
{
  'date': '2007-07-25', # The date this utterance was said
  'year': 2007.0, # The year this utterance was said
  'govt': 'brown', # The prime minister at the time this utterance was said
  'has_latent_repr': True, # TODO: 
  'id': '2007-07-25b.821.8', # A unique ID for this utterance
  'is_answer': False, # Whether this utterance is an answer
  'is_pmq': False, # TODO:
  'is_question': True, # Whether this utterance is a question
  'is_topical': False, # TODO
  'major_name': 'Northern Ireland', # TODO
  'minor_name': 'Policing',  # TODO
  'official_name': 'Northern Ireland', # TODO
  'pair_idx': '2007-07-25.0.0', # The ID of the spacy object in which this question-answer pair is stored
  'root': '2007-07-25b.821.8', # The root of this exchange, always equal to question_text_idx if this utterance is a question, equal to reply_to if this utterance is an answer
  'spans_per_question': 3, # The number of sentences in this question. TODO: I think?
  'text': 'I thank the Minister for his response . He will be aware that the Northern Ireland Policing Board and the Chief Constable are concerned about a possible reduction in the police budget in the forthcoming financial year , and that there are increasing pressures on the budget as a result of policing the past , the ongoing inquiries , and the cost of the legal advice that the police need to secure in order to participate in them . However , does he agree that it is right that the Government provide adequate funding for the ordinary policing in the community that tackles all the matters that concern the people of Northern Ireland ? Does he accept that there should not be a reduction in the police budget , given the increasing costs of the inquiries that I have mentioned ? Will the Government do something to reduce the cost of the inquiries , and ensure that adequate policing is provided for all the victims of crime in Northern Ireland ?', # The text of this utterance
  'user': 'person/10172', # The ID of the person uttering this utterance
  'user-info': { # Information about the utterer
    'age': 10.238356164383562, # The number of years this person has served as an MP. TODO: I think?
    'is_incumbent': False, # TODO
    'is_minister': True, # Whether this person is a minister
    'is_oppn': False, # Whether this person is in the opposition
    'name': 'Jeffrey M. Donaldson', # Name
    'party': 'dup' # The party affiliation of this person
  },
}

# An example answer

{
  'date': '2007-07-25', 
  'year': 2007.0,
  'govt': 'brown', 
  'has_latent_repr': True, 
  'id': '2007-07-25b.822.0', 
  'is_answer': True, 
  'is_pmq': False, 
  'is_question': False, 
  'is_topical': False, 
  'major_name': 'Northern Ireland', 
  'minor_name': 'Policing', 
  'official_name': 'Northern Ireland', 
  'pair_idx': '2007-07-25.0.0', 
  'reply-to': '2007-07-25b.821.8', 
  'root': '2007-07-25b.821.8', 
  'spans_per_question': 3, 
  'text': "Mr. Speaker , we had the Adjournment debate yesterday , and in it we covered much of the territory in the hon Gentleman 's question . However , I shall start with the facts . This year 's budget for the Police Service of Northern Ireland is £ 100 million more than it was two years ago . As for the CSR discussions , I assure him that we want to maintain the same police numbers—that is , 7,500 officers—as we have at the moment . For obvious reasons , that is rather more than one would find in the average police service in the rest of the UK. What matters too is how those resources are deployed . That is very important , and the Chief Constable 's commitment to neighbourhood policing is very welcome . The right hon Member for Lagan Valley ( Mr. Donaldson ) touches on another important issue that no doubt will be touched on later in our proceedings today . We can not go on spending on investigations into the past without there being a knock - on effect on both present and future services . The CSR considerations that are going on at the moment will be important in that respect , but I am committed to making sure that we continue to provide the necessary funding for the PSNI , and I am determined that that will happen .", 
  'user': 'person/10232', 
  'user-info': {
    'is_incumbent': True, 
    'name': 'Paul Goggins', 
    'party': 'labour'
  }, 
}


References
----------

TODO: What goes here?

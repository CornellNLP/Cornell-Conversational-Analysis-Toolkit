"""The objects used to represent a dataset."""

import json
from functools import total_ordering, reduce
from collections import defaultdict

pair_delim = '-q-a-'

@total_ordering
class User:
    """Represents a single user in a dataset.

    :param name: name of the user.
    :type name: str
    :param info: arbitrary dictionary of attributes associated
        with the user.
    :type info: dict

    :ivar name: name of the user.
    :ivar info: dictionary of attributes associated with the user.
    """

    def __init__(self, name=None, info={}):
        self._name = name
        self._info = info
        self._split_attribs = set()
        self._update_uid()

    def identify_by_attribs(self, attribs):
        """Identify a user by a list of attributes. Sets which user info
        attributes should distinguish users of the same name in equality tests.
        For example, in the Supreme Court dataset, users are labeled with the
        current case id. Call this method with attribs = ["case"] to count
        the same person across different cases as different users.

        :param attribs: Collection of attribute names.
        :type attribs: Collection
        """

        self._split_attribs = set(attribs)
        self._update_uid()

    def _get_name(self): return self._name
    def _set_name(self, value):
        self._name = value
        self._update_uid()
    name = property(_get_name, _set_name)

    def _get_info(self): return self._info
    def _set_info(self, value):
        self._info = value
        self._update_uid()
    info = property(_get_info, _set_info)

    def _update_uid(self):
        rep = {}
        rep["name"] = self._name
        if self._split_attribs:
            rep["attribs"] = {k: self._info[k] for k in self._split_attribs
                    if k in self._info}
        self._uid = "User(" + str(sorted(rep.items())) + ")"

    def __eq__(self, other):
        return self._uid == other._uid

    def __lt__(self, other):
        return self._uid < other._uid

    def __hash__(self):
        return hash(self._uid)

    def __repr__(self):
        return self._uid

class Utterance:
    """Represents a single utterance in the dataset.

    :param id: the unique id of the utterance. Can be any hashable type.
    :param user: the user giving the utterance.
    :param root: the id of the root utterance of the conversation.
    :param reply_to: id of the utterance this was a reply to.
    :param timestamp: timestamp of the utterance. Can be any
        comparable type.
    :param text: text of the utterance.
    :type text: str

    :ivar id: the unique id of the utterance.
    :ivar user: the user giving the utterance.
    :ivar root: the id of the root utterance of the conversation.
    :ivar reply_to: id of the utterance this was a reply to.
    :ivar timestamp: timestamp of the utterance.
    :ivar text: text of the utterance.
    """

    def __init__(self, id=None, user=None, root=None, reply_to=None,
            timestamp=None, text=None, other=None):
        self.id = id
        self.user = user
        self.root = root
        self.reply_to = reply_to
        self.timestamp = timestamp
        self.text = text
        self.other = other

    def get(self, key):
        if key == "id":
            return self.id
        elif key == "user":
            return self.user
        elif key == "root":
            return self.root
        elif key == "reply_to":
            return self.reply_to
        elif key == "timestamp":
            return self.timestamp
        elif key == "text":
            return text

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "Utterance(" + str(self.__dict__) + ")"

class Corpus:
    """Represents a dataset, which can be loaded from a JSON file, CSV file or a
    list of utterances.

    If a CSV file, the first row should consist of key names. Unknown key names
    will have their attributes added to the "user-info" entry.

    :param filename: path of json or csv file to load
    :param utterances: list of utterances to load
    :param merge_lines: whether to merge adjacent
        lines from the same user if the two utterances have the same root.
    :param subdivide_users_by: collection of strings corresponding to attribute
        names defined in the "user-info" entry. Use this if you want to count
        the same user as being different depending on attributes other than
        username. For example, in the Supreme Court dataset, users are annotated
        with the current case id. You could use this to count the same person
        across different cases as being different users.
    :param delim: if loading a csv, specifies the delimiter string.

    :ivar utterances: dictionary of utterances in the dataset, indexed by id.
    """

    def __init__(self, filename=None, utterances=None, merge_lines=False,
        subdivide_users_by=[], delim=","):
        KeyId = "id"
        KeyUser = "user"
        KeyConvoRoot = "root"
        KeyReplyTo = "reply-to"
        KeyTimestamp = "timestamp"
        KeyText = "text"
        DefinedKeys = set([KeyId, KeyUser, KeyConvoRoot, KeyReplyTo,
            KeyTimestamp, KeyText])
        KeyUserInfo = "user-info"  # can store any extra data

        if filename is not None:
            with open(filename, "r") as f:
                try:
                    utterances = json.load(f)
                except:
                    try:
                        utterances = self._load_csv(f, delim, DefinedKeys)
                    except:
                        raise ValueError("Couldn't load corpus:" +
                            " unknown file type")

            self.utterances = {}
            self.all_users = set()
            users_cache = {}   # avoids creating duplicate user objects
            #print(len(utterances))
            for i, u in enumerate(utterances):
                # print(u)
                #if i % 100000 == 0: print(i, end=" ", flush=True)
                u = defaultdict(lambda: None, u)
                user_key = (u[KeyUser], str(sorted(u[KeyUserInfo].items())) if
                    u[KeyUserInfo] is not None else None)
                if user_key not in users_cache:
                    users_cache[user_key] = User(name=u[KeyUser],
                        info=u[KeyUserInfo])
                user = users_cache[user_key]
                self.all_users.add(user)
                other_keys = list(u.keys())
                other_keys.remove(KeyText)
                other = {}
                for key in other_keys:
                    other[key] = u[key]
                ut = Utterance(id=u[KeyId], user=user,
                        root=u[KeyConvoRoot],
                        reply_to=u[KeyReplyTo], timestamp=u[KeyTimestamp],
                        text=u[KeyText], other=other)
                self.utterances[ut.id] = ut
        elif utterances is not None:
            self.all_users = set([u.user for u in utterances])
            self.utterances = { u.id: u for u in utterances }

        if merge_lines:
            new_utterances = {}
            for uid, u in self.utterances.items():
                merged = False
                if u.reply_to is not None and u.user is not None:
                    u0 = self.utterances[u.reply_to]
                    if u0.root == u.root and u0.user == u.user:
                        new_utterances[u0.id].text += " " + u.text
                        merged = True
                if not merged:
                    new_utterances[u.id] = u
            self.utterances = new_utterances

        if subdivide_users_by:
            self.subdivide_users_by(subdivide_users_by)

    def _load_csv(self, f, delim, defined_keys):
        keys = f.readline()[:-1].split(delim)
        utterances = []
        for line in f:
            values = line[:-1].split(delim)
            utterance = {"user-info": {}}
            for k, v in zip(keys, values):
                if k in defined_keys:
                    utterance[k] = v
                else:
                    utterance["user-info"][k] = v
            utterances.append(utterance)
        return utterances

    def subdivide_users_by(self, attribs):
        """Use this if you want to count the same user as being different
        depending on attributes other than username. For example, in the Supreme
        Court dataset, users are annotated with the current case id. You could
        use this to count the same person across different cases as being
        different users.

        Repeated calls to this method will override previous subdivisions.

        :param attribs: Collection of attribute names to subdivide users on.
        :type attribs: Collection
        """

        new_all_users = set()
        for u in self.utterances.values():
            if u.user is not None:
                u.user.identify_by_attribs(attribs)
                new_all_users.add(u.user)
        self.all_users = new_all_users

    def filter_utterances_by(self, regular_kv_pairs={},
        user_info_kv_pairs={}, other_kv_pairs={}):
        """
        Creates a subset of the utterances filtered by certain attributes. Irreversible.
        If the method is run again, it will filter the already filtered subset.
        Always takes the intersection of the specified key-pairs
        """
        new_utterances = {}
        regular_keys = list(regular_kv_pairs.keys())
        user_info_keys = list(user_info_kv_pairs.keys())
        other_keys = list(other_kv_pairs.keys())
        for uid, utterance in self.utterances.items():
            user_info = utterance.user._get_info()
            other_dict = utterance.other
            regular = all(utterance.get(key) == regular_kv_pairs[key] for key in regular_keys)
            user = all(user_info[key] == user_info_kv_pairs[key] for key in user_info_keys)
            other = all(other_dict[key] == other_kv_pairs[key] for key in other_keys)
            if regular and user and other:
                new_utterances[uid] = utterance

        self.utterances = new_utterances

    def users(self, selector=None):
        """Get users in the dataset.

        :param selector: optional function that takes in a
            `User` and returns True to include the user in the
            resulting list, or False otherwise.

        :return: Set containing all users selected by the selector function,
            or all users in the dataset if no selector function was
            used.
        """
        if selector is None:
            return self.all_users
        else:
            return set([u for u in self.all_users if selector(u)])

    def user_names(self, selector=None):
        """Get names of users in the dataset.

        :param selector: optional function that takes in a
            `User` and returns True to include the user's name in the
            resulting list, or False otherwise.

        :return: Set containing all user names selected by the selector
            function, or all user names in the dataset if no selector function
            was used.
        """
        return set([u.name for u in self.users(selector)])

    def speaking_pairs(self, selector=None, user_names_only=False):
        """Get all directed speaking pairs (a, b) of users such that a replies
            to b at least once in the dataset.

        :param selector: optional function that takes in
            a speaker user and a replied-to user and returns True to include
            the pair in the result, or False otherwise.
        :param user_names_only: if True, return just pairs of
            user names rather than user objects.
        :type user_names_only: bool

        :return: Set containing all speaking pairs selected by the selector
            function, or all speaking pairs in the dataset if no selector
            function was used.
        """
        pairs = set()
        for u2 in self.utterances.values():
            if u2.user is not None and u2.reply_to is not None and \
                u2.reply_to in self.utterances:
                u1 = self.utterances[u2.reply_to]
                if u1.user is not None:
                    if selector is None or selector(u2.user, u1.user):
                        pairs.add((u2.user.name, u1.user.name) if
                                user_names_only else (u2.user, u1.user))
        return pairs

    def pairwise_exchanges(self, selector=None, user_names_only=False):
        """Get all directed pairwise exchanges in the dataset.

        :param selector: optional function that takes in a
            speaker user and a replied-to user and returns True to include
            the pair in the result, or False otherwise.
        :param user_names_only: if True, index conversations
            by user names rather than user objects.
        :type user_names_only: bool

        :return: Dictionary mapping (speaker, target) tuples to a list of
            utterances given by the speaker in reply to the target.
        """
        pairs = defaultdict(list)
        for u2 in self.utterances.values():
            if u2.user is not None and u2.reply_to is not None:
                u1 = self.utterances[u2.reply_to]
                if u1.user is not None:
                    if selector is None or selector(u2.user, u1.user):
                        key = ((u2.user.name, u1.user.name) if
                                user_names_only else (u2.user, u1.user))
                        pairs[key].append(u2)
        return pairs


    def iterate_by(self, iter_type, is_utterance_question):
        """Iterator for utterances.

        Can give just questions, just answers or questions followed by their answers
        """
        i = -1
        for utterance in self.utterances.values():
            if utterance.reply_to is not None:
                root_text = self.utterances[utterance.reply_to].text
                if is_utterance_question(root_text):
                    i += 1
                    if iter_type == 'answers':
                        pair_idx = utterance.reply_to + pair_delim + utterance.id
                        yield utterance.id, utterance.text, pair_idx
                        continue
                    question = self.utterances[utterance.reply_to]
                    pair_idx = question.id + pair_delim + utterance.id
                    yield question.id, question.text, pair_idx
                    if iter_type == 'both':
                        pair_idx = utterance.reply_to + pair_delim + utterance.id
                        yield utterance.id, utterance.text, pair_idx

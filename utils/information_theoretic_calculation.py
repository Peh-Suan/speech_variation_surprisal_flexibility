import tqdm
import numpy as np

class Node:
    def __init__(self):
        self.followings = {}
        self.count = 0

    def add(self, phones, count):

        self.count += count

        if not phones:
            return

        phone = phones[0]

        if phone not in self.followings:
            self.followings[phone] = Node()

        self.followings[phone].add(
            phones[1:],
            count,
        )

    def get_count(self, phones):

        if not phones:
            return self.count

        phone = phones[0]

        if phone not in self.followings:
            return 0

        return self.followings[phone].get_count(
            phones[1:]
        )

    def __repr__(self):
        return (
            f"Node(count={self.count}, "
            f"followings={list(self.followings.keys())})"
        )

class InformationTheoreticCalculator():
    def __init__(self, pronunciation_counts, phone_counts, sep="_"):
        self.phone_counts = phone_counts
        self.sep = sep

        self.pronunciation_count_tree = self.get_pronunciation_count_tree(pronunciation_counts)

    def get_pronunciation_count_tree(self, pronunciation_counts):
        pronunciation_count_tree = Node()

        for pronunciation, count in tqdm.tqdm(pronunciation_counts.items()):
            phones = pronunciation.split(self.sep) + ["#"]

            pronunciation_count_tree.add(phones, count)

        return pronunciation_count_tree

    def get_contextual_surprisal(self, context, target):

        context = (
            context.split(self.sep)
            if context
            else []
        )

        target = (
            target.split(self.sep)
            if target
            else []
        )
        context_count = self.pronunciation_count_tree.get_count(context)
        context_target_count = self.pronunciation_count_tree.get_count(context + target)

        if context_target_count==0 or context_count==0:
            return None

        return -np.log(context_target_count/context_count)

    def get_target_pos_entropy(self, context):

        context_phones = (
            context.split(self.sep)
            if context
            else []
        )

        context_count = (
            self.pronunciation_count_tree
            .get_count(context_phones)
        )

        if context_count == 0:
            return None

        entropy = 0.0

        for target in list(self.phone_counts) + ["#"]:

            target_phones = target.split(self.sep)

            target_count = (
                self.pronunciation_count_tree
                .get_count(
                    context_phones + target_phones
                )
            )

            if target_count == 0:
                continue

            p = target_count / context_count

            entropy -= p * np.log(p)

        return entropy
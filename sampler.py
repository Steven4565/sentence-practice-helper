import random
import pandas as pd
from anki.collection import Collection
from typing import List

class WordSampler:
    dict_paths = {
        "indonesian": "~/Downloads/Indonesian_Vocabulary_beginner_to_intermediate_/collection.anki2",
        "korean": "/home/steven/.local/share/Anki2/User 1/collection.anki2"
    }

    def __init__(self, language: str):
        self.language = language.lower()
        if self.language not in self.dict_paths:
            raise ValueError(f"Language '{language}' not supported. Available: {list(self.dict_paths.keys())}")

        self.col = Collection(self.dict_paths[self.language])
        ids = self.col.find_cards("introduced:2")
        new_words = []
        for id in ids: 
            card = self.col.get_card(id)
            note = self.col.get_note(card.nid)
            new_words.append(note.fields[1])


        self.new_words = new_words

    def get_samples(self, num: int = 2):
        if num > len(self.new_words):
            raise ValueError(f"Requested {num} samples, but only {len(self.new_words)} words available.")
        return random.sample(self.new_words, num)

    def get_unknown_vocab(self, voc_list: List[str] ): 
        new_dict = {}
        for voc in voc_list: 
            cards = self.col.find_cards("-is:new Vocab:" + voc)
            if (cards):
                # If user has already learned card
                continue

            cards = self.col.find_cards("Vocab:" + voc)
            if (not cards): 
                # Not in anki dictionary
                new_dict[voc] = "Vocab not found"
                continue

            card = self.col.get_card(cards[-1])
            note = self.col.get_note(card.nid).fields[4] # TODO: don't hardcode this
            new_dict[voc] = note
        return new_dict


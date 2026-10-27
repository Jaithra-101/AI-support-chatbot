"""Domain-aware spelling correction.

Instead of a general dictionary, misspelled words are matched against the
bot's own vocabulary (FAQ phrasings + document text), so "refnd" becomes
"refund" and "shiping" becomes "shipping". Real English words are never
touched, so "about" stays "about".
"""
import difflib
import re

from nltk.corpus import stopwords, wordnet

_WORD = re.compile(r"[A-Za-z]+")
_STOP = set(stopwords.words("english"))


class SpellCorrector:
    def __init__(self):
        self.vocab: list[str] = []
        self.vocab_set: set[str] = set()

    def rebuild(self, texts: list[str]):
        self.vocab_set = {w.lower() for t in texts for w in _WORD.findall(t) if len(w) > 2}
        self.vocab = sorted(self.vocab_set)

    def _is_known(self, word: str) -> bool:
        return (len(word) < 4 or word in _STOP or word in self.vocab_set
                or bool(wordnet.synsets(word)))

    def correct(self, text: str) -> str:
        if not self.vocab:
            return text

        def fix(m: re.Match) -> str:
            word = m.group(0)
            low = word.lower()
            if self._is_known(low):
                return word
            # 0.82: real typos score 0.87+ ("refnd"->"refund" 0.91), while unknown names
            # must stay as they are ("spotify"->"notify" is only 0.77)
            match = difflib.get_close_matches(low, self.vocab, n=1, cutoff=0.82)
            return match[0] if match else word

        return _WORD.sub(fix, text)

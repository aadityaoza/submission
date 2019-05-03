
class LanguageModel:
    """Models prior probability of unigrams and bigrams."""

    def __init__(self, corpus_dir='pa2-data/corpus', lambda_=0.1):
        """Iterates over all whitespace-separated tokens in each file in
        `corpus_dir`, and counts the number of occurrences of each unigram and
        bigram. Also keeps track of the total number of tokens in the corpus.

        Args:
            corpus_dir (str): Path to directory containing corpus.
            lambda_ (float): Interpolation factor for smoothing by unigram-bigram
                interpolation. You only need to save `lambda_` as an attribute for now, and
                it will be used later in `LanguageModel.get_bigram_logp`. See Section
                IV.1.2. below for further explanation.
        """
        self.lambda_ = lambda_
        self.total_num_tokens = 0        # Counts total number of tokens in the corpus
        self.unigram_counts = Counter()  # Maps strings w_1 -> count(w_1)
        self.bigram_counts = Counter()   # Maps tuples (w_1, w_2) -> count((w_1, w_2))

        ### Begin your code
        
        import glob
        path = corpus_dir + '/'
        files = glob.glob(os.path.join(path, '*.*'))
        for file in files:
            with open(file) as f:
                str_ = f.read()
                tokens = str_.split()
                self.total_num_tokens += len(tokens)
                for token in tokens:
                    self.unigram_counts[token]+=1
                for first, second in zip(tokens, tokens[1:]):
                    self.bigram_counts[(first,second)] += 1
        ### End your code

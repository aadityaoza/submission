
# NOTE: Syntax on the following line just extends the `LanguageModel` class
class LanguageModel(LanguageModel):
    def get_unigram_logp(self, unigram):
        """Computes the log-probability of `unigram` under this `LanguageModel`.

        Args:
            unigram (str): Unigram for which to compute the log-probability.

        Returns:
            log_p (float): Log-probability of `unigram` under this
                `LanguageModel`.
        """
        ### Begin your code
        total_count = self.total_num_tokens
        prob = self.unigram_counts[unigram]/total_count
        return math.log(prob)
        ### End your code

    def get_bigram_logp(self, w_1, w_2):
        """Computes the log-probability of `unigram` under this `LanguageModel`.

        Note:
            Use self.lambda_ for the unigram-bigram interpolation factor.

        Args:
            w_1 (str): First word in bigram.
            w_2 (str): Second word in bigram.

        Returns:
            log_p (float): Log-probability of `bigram` under this
                `LanguageModel`.
        """
        ### Begin your code
        total_count = self.total_num_tokens
        prob_1 = self.unigram_counts[w_2]/total_count
        prob_2 = self.bigram_counts[(w_1,w_2)]/self.unigram_counts[w_1]
        prob = (self.lambda_ * prob_1) + ((1 - self.lambda_) * prob_2)
            
        return math.log(prob)
        ### End your code

    def get_query_logp(self, query):
        """Computes the log-probability of `query` under this `LanguageModel`.

        Args:
            query (str): Whitespace-delimited sequence of terms in the query.

        Returns:
            log_p (float): Log-probability assigned to the query under this
                `LanguageModel`.
        """
        ### Begin your code
        tokens = query.split()
        
        # Compute p(w1) - probability of first word in query
        log_prob = self.get_unigram_logp(tokens[0])
        
        # Add probabilities of bigrams
        for w_1, w_2 in zip(tokens, tokens[1:]):
            log_prob += self.get_bigram_logp(w_1,w_2)
        
        return log_prob
        
        ### End your code

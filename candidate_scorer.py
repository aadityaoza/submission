
class CandidateScorer:
    """Combines the `LanguageModel`, `EditProbabilityModel`, and
    `CandidateGenerator` to produce the most likely query Q given a raw query R.
    Since the candidate generator already uses the edit probability model, we
    do not need to take the edit probability model as an argument in the constructor.
    """
    def __init__(self, lm, cg, mu=1.):
        """
        Args:
            lm (LanguageModel): Language model for estimating P(Q).
            cg (CandidateGenerator): Candidate generator for generating possible Q.
            mu (float): Weighting factor for the language model (see write-up).
                Remember that our probability computations are done in log-space.
        """
        self.lm = lm
        self.cg = cg
        self.mu = mu

    def get_score(self, query, log_edit_prob):
        """Uses the language model and `log_edit_prob` to compute the final
        score for a candidate `query`. Uses `mu` as weighting exponent for P(Q).

        Args:
            query (str): Candidate query.
            log_edit_prob (float): Log-probability of candidate query given
                original query (i.e., log(P(R|Q), where R is `query`).

        Returns:
            log_p (float): Final score for the query, i.e., the log-probability
                of the query.
        """
        ### Begin your code
        query_prob = self.lm.get_query_logp(query)
        if query_prob is None:
            score = log_edit_prob
        else:
            score = log_edit_prob + self.mu * query_prob
        return score
        ### End your code

    def correct_spelling(self, r):
        """Corrects spelling of raw query `r` to get the intended query `q`.

        Args:
            r (str): Raw input query from the user.

        Returns:
            q (str): Spell-corrected query. That is, the query that maximizes
                P(R|Q)*P(Q) under the language model and edit probability model,
                restricted to Q's generated by the candidate generator.
        """
        ### Begin your code
        canditates_scores = []
        total_candidates = 0
        for candidate, candidate_logp in self.cg.get_candidates(r):
            if candidate == ' ' or candidate == '  ' or candidate == '   ':
                print(candidate, ' ',candidate_logp)
                continue
                
            score = self.get_score(candidate,candidate_logp)
            canditates_scores.append((candidate,score))
            total_candidates +=1
        
        if len(canditates_scores) == 0:
            # Kick out from here !!
            return r
        
        # Else sort remaining results
        canditates_scores = sorted(canditates_scores,key = lambda x: x[1], reverse = True)
        result = canditates_scores[0][0]
        result = " ".join(result.split())
        return result
        ### End your code


class CandidateGenerator:
    # Alphabet to use for insertion and substitution
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                ' ', ',', '.', '-']

    def __init__(self, lm, epm):
        """
        Args:
            lm (LanguageModel): Language model to use for prior probabilities, P(Q).
            epm (EditProbabilityModel): Edit probability model to use for P(R|Q).
        """
        self.lm = lm
        self.epm = epm

    def get_num_oov(self, query):
        """Get the number of out-of-vocabulary (OOV) words in `query`."""
        return sum(1 for w in query.strip().split()
                   if w not in self.lm.unigram_counts)

    def filter_and_yield(self, query, lp):
        if query.strip() and self.get_num_oov(query) == 0:
            yield query, lp
    
    #### Helper functions #####
    ###########################
    
    def cartesian(self,list_1,list_2):
        if len(list_1) == 0:
            return list_2
        list_3 = []
        for t1,p1 in list_1:
            for t2,p2 in list_2:
                t3 = t1 + ' ' + t2
                p3 = p1 + p2
                list_3.append((t3,p3))
        return list_3
    
    def lookup(self,t,term,valid,invalid):
        if self.get_num_oov(t) == 0: 
            valid.add((t,self.epm.get_edit_logp(t,term)))
        else:
            invalid.add((t,self.epm.get_edit_logp(t,term)))
    
    # Generate candidates one edit distance apart from input term
    # Returns candidates that are valid words from dictionary
    
    def genCandidates(self,term):
        valid_candidates = set()
        invalid_candidates = set()
        if self.get_num_oov(term) == 0:
            valid_candidates.add((term,self.epm.get_edit_logp(term,term)))
            
        # Delete a character at index i - only if term is greater than 
        # one character
        if len(term) > 1:
            i = 0
            while i < len(term):
                t = ''
                if i == 0:
                    t = term[1:]
                elif i == len(term) - 1:
                    t = term[:-1]
                else:
                    t = term[:i] + term[i+1:]
                self.lookup(t,term,valid_candidates,invalid_candidates)
                i+=1
        
        # Add character from alphabet into string
        i = 0
        while i < len(term):
            for a in self.alphabet:
                t = ''
                if i == len(term) - 1:
                    # Add characters to end of string when last character reached
                    t = term + a
                    self.lookup(t,term,valid_candidates,invalid_candidates)
                
                t = term[:i] + a + term[i:]
                self.lookup(t,term,valid_candidates,invalid_candidates)
            i+=1
        
        # Substitution - replace each character at every index and lookup
        # resulting word in dictionary
        i = 0
        while i < len(term): 
            t = ''
            for a in self.alphabet:
                if i == 0:
                    if len(term) == 1:
                        t = a + ''
                    else:
                        t = a + term[1:]
                elif i == len(term) - 1:
                    t = term[:-1] + a
                else:
                    t = term[:i] + a +term[i+1:]
                self.lookup(t,term,valid_candidates,invalid_candidates)
            i+=1
        
        #Transposition - swap a character with a previous one
        # Only valid for terms longer than 1 character
        if len(term) > 1:
            i = 1
            while i < len(term):
                t = list(term)
                t1 = t[i]
                t2 = t[i-1]
                t[i] = t2
                t[i-1] = t1
                t = ''.join(t)
                self.lookup(t,term,valid_candidates,invalid_candidates)
                i += 1

        return valid_candidates,invalid_candidates
    
    def compareUnigramsLM_ECM(self,candidates):
        updated_candidates = []
        for candidate, ep in candidates:
            up = 0
            if ' ' in candidate:
                tokens = candidate.split()
                for token in tokens:
                    up+= self.lm.get_unigram_logp(token)
            else:
                up = self.lm.get_unigram_logp(candidate)
            updated_candidates.append((candidate,up+ep))
            
        return updated_candidates

    def known(self,words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if self.get_num_oov(w) == 0)
    
    def edits1(self,word):
        "All edits that are one edit away from `word`."
        letters    = self.alphabet
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        
        result = set(deletes + transposes + replaces + inserts)
        result = self.known(result)
        return result
    
    ###########################
    ###########################
    def get_candidates(self, query):
        """Starts from `query`, and performs EDITS OF DISTANCE <=2 to get new
        candidate queries. To make scoring tractable, only returns/yields
        candidates that satisfy certain criteria (ideas for such criteria are
        described in bullet points above).

        Hint: We suggest you implement a helper function that takes a term and
            generates all possible edits of distance one from that term.
            It should probably only return edits that are in the vocabulary
            (i.e., edits for which `self.get_num_oov(edited) == 0`).

        Args:
            query (str): Starting query.

        Returns:
            Iterable over tuples (cdt, cdt_edit_logp) of candidates and
                their associated edit log-probabilities. Return value could be
                a list or a generator yielding tuples of this form.
        """
        # Yield the unedited query first
        # We provide this line as an example of how to use `self.filter_and_yield`
        #yield from self.filter_and_yield(query, self.epm.get_edit_logp(query, query))
        #yield from self.filter_and_yield(query, sum([self.epm.get_edit_logp(term, term) for term in query.split()]))
        ### Begin your code
        
        terms = query.split()
        candidates = []
        if self.get_num_oov(query) == 0:
            candidates.append((query, self.epm.get_edit_logp(query, query)))
        
        i = 0
        while i < len(terms):
            candidate_query = query.split()
            valid = self.edits1(terms[i])
            
            for edited in valid:
                candidate_query[i] = edited
                candidate = " ".join(candidate_query)
                edit_1 = self.epm.get_edit_logp(edited,terms[i])
                candidates.append((candidate,edit_1))
                
                
                ## Code to generate second edit after edit to first word
                ## It generates edit to a different word in candidate query
                ## Currently commented out
                
                j = 0
                while j < len(candidate_query):
                    if i != j:
                        cand_query_2 = candidate_query[:]
                        valid_2 = self.edits1(candidate_query[j])
                    
                        for edited_2 in valid_2:
                            cand_query_2[j] = edited_2
                            cand_2 = " ".join(cand_query_2)
                            edit_2 = self.epm.get_edit_logp(edited_2,edited)
                            candidates.append((cand_2,edit_1 + edit_2))
                            cand_query_2 = candidate_query[:]
                    j += 1
                
                # Re-compute original query
                candidate_query = query.split()
            i+=1
        
        #Remove white spaces at different places in string
        if len(terms) > 1:
            i = 1
            while i < len(terms):
                # Combine previous term and current term
                newTerm = terms[i-1] + terms[i]
                candidate = terms[0:i-1] + [newTerm] + terms[i+1:]
                candidate = " ".join(candidate)
                
                if self.get_num_oov(candidate) == 0:
                    candidates.append((candidate,self.epm.get_edit_logp(newTerm,terms[i])))
                
                ## Code to generate second edit after space
                ## Currently commented out
                
                #j = 0
                #while j < len(candidate.split()):
                #    candidate_query = candidate.split()
                #    valid, invalid = self.genCandidates(candidate_query[j])
                #    for edited, edit_p in valid:
                #        candidate_query[j] = edited
                #        candidate_with_space = " ".join(candidate_query)
                #       
                #        if self.get_num_oov(candidate_with_space) == 0:
                #            candidates.append((candidate_with_space,2*edit_p))
                #           
                #        candidate_query = candidate.split()
                #    j+=1
                i += 1    
        

        for cd, lp_cd in candidates:
            yield from self.filter_and_yield(cd,lp_cd)
        
        # Yield original query
        yield from self.filter_and_yield(query, self.epm.get_edit_logp(query, query))
        
         
        ### End your code

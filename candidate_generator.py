

class CandidateGenerator:
    # Alphabet to use for insertion and substitution
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                ' ', ',', '.', '-', '\'']

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
        # yield from self.filter_and_yield(query, self.epm.get_edit_logp(query, query))

        ### Begin your code    

        # split the query on spaces
        query_tkns = query.split()
        candidates = []
        i = 0
        one_edit_candidate_queries = set()
        while i < len(query_tkns):
            candidate_query = query.split()
            all_candidates = self.get_candidates_for_term(query_tkns[i])

            for edited, edit_p in all_candidates:
                candidate_query[i] = edited
                candidate = " ".join(candidate_query)
                one_edit_candidate_queries.add(candidate)

                k = 0
                total_query_editp = 0
                while k < len(query_tkns):
                    if i == k:
                        total_query_editp = total_query_editp + edit_p
                    else:
                        total_query_editp = total_query_editp + self.epm.get_edit_logp(query_tkns[k],query_tkns[k])
                    k = k+1
                
                candidates.append((candidate,total_query_editp))
                
                j = 0
                mistake = -1 
                while j < len(candidate_query):
                    word = candidate_query[j]
                    if self.get_num_oov(word) != 0:
                        mistake = j
                        break
                    j += 1
                
                if mistake != -1:
                    j = mistake
                    cand_query_2 = candidate_query[:]
                    all_candidates_2 = self.get_candidates_for_term(candidate_query[j])   
                    #all_candidates_2 = set(list(all_candidates_2)[:3])
                    for edited_2, edit_p_2 in all_candidates_2:
                        cand_query_2[j] = edited_2
                        cand_2 = " ".join(cand_query_2)
                        k = 0
                        total_query_editp = 0
                        while k < len(query_tkns):
                            if j == k:
                                total_query_editp = total_query_editp + edit_p + edit_p_2
                            else:
                                total_query_editp = total_query_editp + self.epm.get_edit_logp(query_tkns[k],query_tkns[k])
                            k = k+1
                            
                        one_edit_candidate_queries.add(candidate)
                        candidates.append((cand_2,total_query_editp))
                    #cand_query_2 = candidate_query[:]
                    
                candidate_query = query.split()
            i+=1
        
        #Remove white spaces at different places in string
        if len(query_tkns) > 1:
            i = 1
            while i < len(query_tkns):
                # Combine previous term and current term
                newTerm = query_tkns[i-1] + query_tkns[i]
                candidate = query_tkns[0:i-1] + [newTerm] + query_tkns[i+1:]
                candidate = " ".join(candidate)
                
                if self.get_num_oov(candidate) == 0: 
                    candidate_tkns = candidate.split()
                    k = 0
                    total_query_editp = 0
                    while k < len(query_tkns):
                        if k != len(query_tkns)-1 and candidate_tkns[k] == query_tkns[k] + query_tkns[k+1]:
                            # joined k, k+1
                            merged_word = candidate_tkns[k]
                            original_word = query_tkns[k] + " " + query_tkns[k+1]
                            total_query_editp = total_query_editp + self.epm.get_edit_logp(merged_word, original_word)
                            k = k+1
                        else:
                            total_query_editp = total_query_editp + self.epm.get_edit_logp(query_tkns[k],query_tkns[k])
                        k = k+1
                    candidates.append((candidate,total_query_editp))
                    
                candidate_query = candidate.split()
                
                # Correct spelling mistake in candidate with accidental space above
                j = 0
                mistake = -1 
                while j < len(candidate_query):
                    word = candidate_query[j]
                    if self.get_num_oov(word) != 0:
                        mistake = j
                        break
                    j += 1
                
                if mistake != -1:
                    j = mistake
                    cand_query_2 = candidate_query[:]
                    all_candidates_2 = self.get_candidates_for_term(candidate_query[j])   
                    #all_candidates_2 = set(list(all_candidates_2)[:3])
                    for edited_2, edit_p_2 in all_candidates_2:
                        cand_query_2[j] = edited_2
                        cand_2 = " ".join(cand_query_2)
                        k = 0
                        total_query_editp = 0
                        while k < len(query_tkns):
                            if j == k:
                                total_query_editp = total_query_editp + edit_p + edit_p_2
                            else:
                                total_query_editp = total_query_editp + self.epm.get_edit_logp(query_tkns[k],query_tkns[k])
                            k = k+1
                            
                        one_edit_candidate_queries.add(candidate)
                        candidates.append((cand_2,total_query_editp))
                
                i += 1
        
        #Remove white spaces at different places in string from one_edit_candidates
        
        """for one_one_edit_candidate_query in one_edit_candidate_queries:
            if len(one_one_edit_candidate_query) > 1:
                query_tkns_one_edit = one_one_edit_candidate_query.split()
                accidental_splits = self.get_accidental_splits(query_tkns_one_edit)
            
                for accidental_split in accidental_splits:
                    accidental_split_tkns = accidental_split.split()
                    k = 0
                    total_query_editp = 0
                    while k < len(query_tkns):
                        if k != len(query_tkns)-1 and accidental_split_tkns[k] == query_tkns[k] + query_tkns[k+1]:
                        # joined k, k+1
                            merged_word = accidental_split_tkns[k]
                            original_word = query_tkns[k] + " " + query_tkns[k+1]
                            total_query_editp = total_query_editp + self.epm.get_edit_logp(merged_word, original_word)
                            k = k+1
                        else:
                            total_query_editp = total_query_editp + self.epm.get_edit_logp(query_tkns[k],query_tkns[k])
                        k = k+1                
                    candidates.append((candidate,total_query_editp))
        """
        ### Insert hyphens - at different places in string
        ###################################################
        
                   
        ###################################################
        final_candidates = []
        for cd, lp_cd in candidates:
            if self.get_num_oov(cd) == 0:
                final_candidates.append((cd, lp_cd))
            #yield from self.filter_and_yild(cd,lp_cd)
        
        # Yield original query
        if self.get_num_oov(query) == 0:
            final_candidates.append((query, self.epm.get_edit_logp(query, query)))
        
        return final_candidates
            
    def get_candidates_for_term(self, term):
        all_candidates = set()
        candidates_with_one_edit_distance = set(); candidates_with_two_edit_distance = set()

        if term.isdigit():
            all_candidates.add((term, self.epm.get_edit_logp(term,term)))
            return all_candidates
        
        if self.get_num_oov(term) == 0:
            all_candidates.add((term, self.epm.get_edit_logp(term,term)))
        
        one_edit_tokens = self.get_one_edit_tokens(term)
        for one_edit_token in one_edit_tokens:
            if self.get_num_oov(one_edit_token) == 0:
                candidates_with_one_edit_distance.add((one_edit_token.strip(), self.epm.get_edit_logp(one_edit_token,term)))
        
        #candidates_with_two_edit_distance = candidates_with_two_edit_distance|candidates_with_one_edit_distance
        all_candidates = all_candidates |candidates_with_one_edit_distance
        return all_candidates
        
    def get_one_edit_tokens(self, token):
        one_edit_tokens = set()
        # get accidental insertions by deleting each character. 
        for x in range(len(token)):
            edited = token[:x] + token[x + 1:]
            one_edit_tokens.add(edited)
            
        # get accidental mismatches of characeters
        for x in range(len(token)):
            for c in self.alphabet:
                s_list = list(token)
                s_list[x] = c
                one_edit_tokens.add("".join(s_list))
                
        # get accidental deletes
        for x in range(len(token)+1):
            for c in self.alphabet:
                one_edit_tokens.add(token[:x] + c + token[x:])
        
        # get accidental substitutions
        x = 1
        while(x < len(token)):
            s_list = list(token)
            a = s_list[x]
            b = s_list[x-1]
            s_list[x] = b
            s_list[x-1] = a
            one_edit_tokens.add("".join(s_list))
            x = x+1
        return one_edit_tokens
    
    def get_accidental_splits(self, query_splits):
        words_without_splits = set()
        if len(query_splits) >= 2:
            for i in range(len(query_splits) - 1):
                join = query_splits[i] + query_splits[i+1]
                before = ""; after = ""
                for j in range(i):
                    before = before + query_splits[j] + " "
                j = i + 2
                while(j < len(query_splits)):
                    after = after + query_splits[j] + " "
                    j = j+1
                final = before.strip() + " " + join.strip() + " " + after.strip()
                if final.strip() != "":
                    words_without_splits.add(final.strip())
        return words_without_splits

    ### End your code   

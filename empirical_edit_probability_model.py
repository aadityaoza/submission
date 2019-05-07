

class Edit:
    """Represents a single edit in Damerau-Levenshtein distance.
    We use this class to count occurrences of different edits in the training data.
    """
    INSERTION = 1
    DELETION = 2
    TRANSPOSITION = 3
    SUBSTITUTION = 4

    def __init__(self, edit_type, c1=None, c2=None):
        """
        Members:
            edit_type (int): One of Edit.{NO_EDIT,INSERTION,DELETION,
                TRANSPOSITION,SUBSTITUTION}.
            c1 (str): First (in original) char involved in the edit.
            c2 (str): Second (in original) char involved in the edit.
        """
        self.edit_type = edit_type
        self.c1 = c1
        self.c2 = c2


class EmpiricalEditProbabilityModel(BaseEditProbabilityModel):

    START_CHAR = ''      # Used to indicate start-of-query
    NO_EDIT_PROB = 0.92  # Hyperparameter for probability assigned to no-edit

    def __init__(self, training_set_path='pa2-data/training_set/edit1s.txt'):
        """Builds the necessary data structures to compute log-probabilities of
        distance-1 edits in constant time. In particular, counts the unigrams
        (single characters), bigrams (of 2 characters), alphabet size, and
        edit count for insertions, deletions, substitutions, and transpositions.

        Hint: Use the `Edit` class above. It may be easier to write the `get_edit`
        function first, since you should call that function here.

        Note: We suggest using tqdm with the size of the training set (819722) to track
        the initializers progress when parsing the training set file.

        Args:
            training_set_path (str): Path to training set of empirical error data.
        """
        # Your code needs to initialize all four of these data structures
        self.unigram_counts = Counter()  # Maps chars c1 -> count(c1)
        self.bigram_counts = Counter()   # Maps tuples (c1, c2) -> count((c1, c2))
        self.alphabet_size = 0           # Counts all possible characters

        # Maps edit-types -> dict mapping tuples (c1, c2) -> count(edit[c1, c2])
        # Example usage: 
        #   > e = Edit(Edit.SUBSTITUTION, 'a', 'b')
        #   > edit_count = self.edit_counts[e.edit_type][(e.c1, e.c2)]
        self.edit_counts = {edit_type: Counter()
                            for edit_type in (Edit.INSERTION, Edit.DELETION,
                                              Edit.SUBSTITUTION, Edit.TRANSPOSITION)}
        
        with open(training_set_path, 'r') as training_set:
            for example in tqdm(training_set, total=819722):
                edited, original = example.strip().split('\t')
                ### Begin your code
                edit = self.get_edit(edited, original)
                # set the value in edit_counts
                if edit:
                    self.edit_counts[edit.edit_type][tuple((edit.c1, edit.c2))] += 1
                
                prev = '$'
                i = 0
                while (i < len(original)):
                    self.unigram_counts[original[i]] += 1
                    self.bigram_counts[tuple((prev, original[i]))] += 1
                    prev = original[i]
                    i = i+1
                
                self.bigram_counts[tuple((prev, '$'))] += 1
                
        self.alphabet_size = len(self.unigram_counts)
        print(len(self.bigram_counts))
        print(len(self.unigram_counts))

    def get_edit(self, edited, original):
        """Gets an `Edit` object describing the type of edit performed on `original`
        to produce `edited`.

        Note: Only edits with an edit distance of at most 1 are valid inputs.

        Args:
            edited (str): Raw query, which contains exactly one edit from `original`.
            original (str): True query. Want to find the edit which turns this into `edited`.

        Returns:
            edit (Edit): `Edit` object representing the edit to apply to `original` to get `edited`.
                If `edited == original`, returns None.
        """
        ### Begin your code
        edited_len = len(edited)
        original_len = len(original)
        edit = None
        if original_len == edited_len:
            # find out which character is mismatched. case of transposition|substitution
            mismatched_chars = {}
            i = 0
            while i < original_len:
                if original[i] != edited[i]:
                    mismatched_chars[i] = ((original[i], edited[i]))
                i = i+1
            
            no_of_mismatched_chars = len(mismatched_chars)
            if no_of_mismatched_chars == 1:
                # character has been substituted.
                index = list(mismatched_chars.keys())[0]
                char_tuple = mismatched_chars[index]
                edit = Edit(Edit.SUBSTITUTION, char_tuple[1], char_tuple[0])
            elif no_of_mismatched_chars == 2:
                # transposition case
                index = list(mismatched_chars.keys())[0]
                char_tuple = mismatched_chars[index]
                edit = Edit(Edit.TRANSPOSITION, char_tuple[1], char_tuple[0])
        else:
            # this is the case of insertion, deletion
            c1 = 'a'
            c2 = 'a' 
            prev = '$'
            mismatch_index = -1
            i = 0
            if edited_len > original_len:
                # this is insertion    
                while i < original_len:
                    if original[i] != edited[i]:
                        mismatch_index = i
                        c1 = prev
                        c2 = edited[i]
                        break
                    else:
                        prev = original[i]
                    i = i+1
                # finished traversing original string
                if mismatch_index == -1:
                    # extra char is at the end
                    c2 = edited[edited_len-1]
                
                # set the edit object.
                edit = Edit(Edit.INSERTION, c1, c2)
            else:
                # this is deletion, edited string will be shorted
                while i < edited_len:
                    if original[i] != edited[i]:
                        mismatch_index = i
                        c1 = prev
                        c2 = original[i]
                        break
                    else:
                        prev = original[i]
                    i = i+1
                # finished traversing original string
                if mismatch_index == -1:
                    # char deleted from end
                    c2 = original[original_len-1]
                
                # set the edit object.
                edit = Edit(Edit.DELETION, c1, c2)        
        return edit
        ### End your code

    def get_edit_logp(self, edited, original):
        """Gets the log-probability of editing `original` to arrive at `edited`.
        The `original` and `edited` arguments are both single terms that are at
        most one edit apart.
        
        Note: The order of the arguments is chosen so that it reads like an
        assignment expression:
            > edited := EDIT_FUNCTION(original)
        or, alternatively, you can think of it as a (unnormalized) conditional probability:
            > log P(edited | original)

        Args:
            edited (str): Edited term.
            original (str): Original term.

        Returns:
            logp (float): Log-probability of `edited` given `original`
                under this `EditProbabilityModel`.
        """
        ### Begin your code
        if edited == original:
            edit_prob =  math.log(self.NO_EDIT_PROB)
        else:
            edit = self.get_edit(edited, original)
            count = 0
            den = 0
            c1 = edit.c1
            c2 = edit.c2
            if self.edit_counts[edit.edit_type][(c1, c2)]:
                count = self.edit_counts[edit.edit_type][(c1,c2)]
            
            if edit.edit_type == Edit.INSERTION or edit.edit_type == Edit.SUBSTITUTION:
                if c1 in self.unigram_counts:
                    den = self.unigram_counts[c1] + self.alphabet_size
            elif edit.edit_type == Edit.DELETION or edit.edit_type == Edit.TRANSPOSITION:
                if (c1,c2) in self.bigram_counts:
                    den = self.bigram_counts[(c1,c2)] + (self.alphabet_size * self.alphabet_size)
            
            edit_prob = math.log(count + 1) - math.log(den + self.alphabet_size )
        
        return edit_prob
        ### End your code

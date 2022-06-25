import copy
from enum import Enum
from difflib import SequenceMatcher
from collections import namedtuple, Counter, defaultdict
import copy


def break_match(a: str, b: str, match: namedtuple):
    """Breaks a strings with a common substring into prefix, root, suffix.

    Args:
        a (str)
        b (str)
        match (NamedTuple): match coming from difflib.SequenceMatcher

    Returns:
        Tuple[Tuple[str]]: a tuple of tuples of string affixes
    """

    a_prefix = a[: match.a]
    a_root = a[match.a : match.a + match.size]
    a_suffix = a[match.a + match.size :]

    b_prefix = b[: match.b]
    b_root = b[match.b : match.b + match.size]
    b_suffix = b[match.b + match.size :]

    return (a_prefix, a_root, a_suffix), (b_prefix, b_root, b_suffix)


class EditType(Enum):
    EQL = 0
    INS = 1
    DEL = 2


class LemmaScriptGenerator:
    """Class with which a lemma script can be generated.

    Implements all the individual parts as described in Straka's works.
    """

    view = {
        EditType.EQL.value: (" ", "\033[39m"),
        EditType.INS.value: ("+", "\033[92m"),
        EditType.DEL.value: ("-", "\033[91m"),
    }

    def __init__(self, word_form: str, lemma: str):

        self.word_form = word_form

        self.lemma = lemma

    def _myers_diff(self, A: str, B: str):
        """A Myer's diff algorithm implementation.

        Taken from:
        B., Amar, myersdiff, (2020), GitHub repository, https://github.com/amar-b/myersdiff, commit 68bacb370e292cc2b3b0c009250182be4aaffac2

        Inspired by:
        Coglan, James, The Myers diff algorithm: part 1 (2017). The If Works, https://blog.jcoglan.com/2017/02/12/the-myers-diff-algorithm-part-1/
        Coglan, James, The Myers diff algorithm: part 2 (2017). The If Works, https://blog.jcoglan.com/2017/02/15/the-myers-diff-algorithm-part-2/

        Args:
            A (str): source string
            B (str): destination string
        """

        def _shortest_path():
            lenA, lenB = len(A), len(B)
            max_edit = lenA + lenB
            V = [0 for _ in range(2 * max_edit + 1)]
            T = []

            for D in range(max_edit + 1):
                T.append(copy.deepcopy(V))
                for k in range(-D, D + 1, 2):
                    x = (
                        V[k + 1]
                        if (k == -D or (k != D and V[k - 1] < V[k + 1]))
                        else V[k - 1] + 1
                    )
                    y = x - k
                    while x < lenA and y < lenB and A[x] == B[y]:
                        x += 1
                        y += 1

                    V[k] = x
                    if x >= lenA and y >= lenB:
                        return T
            return T

        def _backtrack(trace):
            def _line_diff(x, y, x_prev, y_prev):
                if x == x_prev:
                    return (-1, y_prev + 1, EditType.INS)
                elif y == y_prev:
                    return (x_prev + 1, -1, EditType.DEL)
                else:
                    return (x_prev + 1, y_prev + 1, EditType.EQL)

            x = len(A)
            y = len(B)

            for (D, v) in reversed(list(enumerate(trace))):
                k = x - y
                k_prev = (
                    k + 1 if (k == -D or (k != D and v[k - 1] < v[k + 1])) else k - 1
                )
                x_prev = v[k_prev]
                y_prev = x_prev - k_prev

                while x > x_prev and y > y_prev:
                    yield _line_diff(x, y, x - 1, y - 1)
                    x -= 1
                    y -= 1

                if D > 0:
                    yield _line_diff(x, y, x_prev, y_prev)

                x = x_prev
                y = y_prev

        if isinstance(A, str):
            A = list(A)

        if isinstance(B, str):
            B = list(B)

        return _backtrack(_shortest_path())

    def get_edit_script_affix(self, wf_affix: str, lm_affix: str):
        """Generates an edit string from word_form affix to lemma affix.

        Args:
            wf_affix (str): source string
            lm_affix (str): destination string

        Returns:
            str: edit script in string format
        """
        diff = self._myers_diff(wf_affix, lm_affix)

        result = ""
        for (_, new, editType) in diff:
            sign, _ = self.view[editType.value]

            if editType == EditType.DEL:
                value = ""
            elif editType == EditType.EQL:
                value = "*"
                sign = ""
            else:
                value = lm_affix[new - 1]

            result = f"{sign}{value}" + result

        return result

    def get_casing_script(self):
        """Generates a casing script from a lemma.

        Returns:
            str: casing rules in string format
        """

        case_script = []
        for i, c in enumerate(self.lemma):
            case = "U" if c.isupper() else "L"
            if i == 0 or case_script[-1][0] != case:
                case_script.append(case + str(i))

        case_script = ",".join(case_script)

        return case_script

    def get_lemma_script(self):
        """Generate the lemma script to convert word_form into lemma.

        Returns:
            str: lemma script in string format
        """

        # ======================================================================
        # Edit script
        # ======================================================================
        # First, get the longest-common-substring (LCS) to find 'root'
        match = SequenceMatcher(
            a=self.word_form.lower(), b=self.lemma.lower()
        ).find_longest_match(alo=0, ahi=len(self.word_form), blo=0, bhi=len(self.lemma))

        # Check for non-existent LCS, which implies irregular inflection
        if match.size == 0:
            edit_script = f"ign_{self.lemma}"

        # Else, proceed to produce edit scripts for pre- and suffix separately
        else:
            (
                (wf_prefix, wf_root, wf_suffix),
                (lm_prefix, lm_root, lm_suffix),
            ) = break_match(self.word_form.lower(), self.lemma.lower(), match)

            # Get the edit script, and check for empty edit special case ("d" for "do nothing")
            if len(wf_prefix) or len(lm_prefix):
                prefix_script = self.get_edit_script_affix(wf_prefix, lm_prefix)
            else:
                prefix_script = "d"

            if len(wf_suffix) or len(lm_suffix):
                suffix_script = self.get_edit_script_affix(wf_suffix, lm_suffix)
            else:
                suffix_script = "d"

            edit_script = f"{prefix_script}|{suffix_script}"

        # ======================================================================
        # Casing script
        # ======================================================================

        casing_script = self.get_casing_script()

        lemma_script = f"{casing_script}|{edit_script}"

        return lemma_script


def apply_edit_script(word_form: str, edit_rules: list, verbose: bool = True):
    """Applies an edit script to a word_form to produce a lemma. Does not case.

    Args:
        word_form (str)
        lemma (str)
        edit_rules (list): list of strings, where each string is an edit script
            applied to the prefix or suffix of the word_form

    Returns:
        str: the lemma
    """

    if "ign" not in edit_rules[0]:
        # If not irregular, apply the edit script

        # First split the word_form based on the edit rules for the affixes
        prefix_counts = Counter(edit_rules[0])
        prefix_length = prefix_counts["-"] + prefix_counts["*"]
        wf_prefix = word_form[:prefix_length].lower()

        suffix_counts = Counter(edit_rules[1])
        suffix_length = suffix_counts["-"] + suffix_counts["*"]
        wf_suffix = word_form[len(word_form) - suffix_length :].lower()

        wf_root = word_form[prefix_length : len(word_form) - suffix_length].lower()

        edit_result = [wf_root]
        for i, (affix, rule) in enumerate(zip([wf_prefix, wf_suffix], edit_rules)):

            if rule == "d":
                # If the rule is do nothing, skip
                affix_ = affix

            else:
                # Else, iterate though the string and apply the edits
                pointer = 0
                affix_ = list(copy.deepcopy(affix))
                rule_ = list(copy.deepcopy(rule))
                while len(rule_):
                    try:
                        edit = rule_.pop(0)

                        if edit == "-":
                            affix_.pop(pointer)

                        elif edit == "+":
                            affix_.insert(pointer, rule_.pop(0))
                            pointer += 1

                        elif edit == "*":
                            pointer += 1

                    # THIS GIVES SO MANY ERRORS ON LISA...
                    except IndexError:
                        if verbose:
                            try:
                                wf_ = word_form.encode("utf-8").decode("latin-1")
                                print(
                                    f"Returning intermediate result. Word form, {wf_}, shorter than required by script, {edit_rules}"
                                )

                            except UnicodeEncodeError:
                                print(
                                    f"Returning intermediate result. Word form, <CANT DISPLAY>, shorter than required by script, <CANT DISPLAY>"
                                )
                        break

                affix_ = "".join(affix_)

            if i == 0:
                # If prefix, prepend
                edit_result.insert(0, affix_)
            else:
                # If suffix, append
                edit_result.append(affix_)

        # Construct the lemmatized string
        edit_result = "".join(edit_result)

    else:
        edit_result = edit_rules[0].rsplit("_")[-1]

    return edit_result


def apply_casing_script(lemma: str, casing_script: str):
    """Applies a casing script to a string.

    Assumes the string has already been lemmatized.

    Args:
        lemma (str)
        casing_script (str, optional): casing script in string form

    Returns:
        str: string with applied casing
    """

    script_ = casing_script.rsplit(",")

    cased_string = ""

    rule = script_.pop(-1)
    case, pos = rule[0], int(rule[1:])

    for i, c in enumerate(lemma.lower()[::-1]):
        # Iterate backwards through the string
        # Continually load the next rule
        # First character always has a rule

        if len(lemma) - i == pos:
            rule = script_.pop(-1)
            case, pos = rule[0], int(rule[1:])

        cased_c = c.upper() if case == "U" else c.lower()

        cased_string = cased_c + cased_string

    return cased_string


def apply_lemma_script(word_form: str, lemma_script: list, verbose: bool = True):

    rules = lemma_script.rsplit("|")
    case_rules, edit_rules = rules[0], rules[1:]

    edit_result = apply_edit_script(word_form, edit_rules, verbose)

    case_result = apply_casing_script(edit_result, case_rules)

    return case_result

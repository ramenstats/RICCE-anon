"""TemporalSwap-SFT: rule-based semantic reversal operator rho for captions.

rho: construction<->demolition, appearance<->disappearance,
expansion<->contraction (increase<->decrease); no-change -> no-change;
objects / locations are left untouched. Swapped captions are RULE-GENERATED
(word/verb substitution), NOT human-written. A caption is reversible only if
every directional word in it is covered by the table and all directional
words belong to ONE direction group; otherwise it is dropped (never guessed).
"""
import re

# surface form -> (reversed surface form, direction group of the ORIGINAL word)
# groups: app (appearance/construction), dis (disappearance/demolition),
#         exp (expansion), con (contraction)
_T = {
    # appearance / construction  ->  disappearance / demolition
    'built': ('demolished', 'app'), 'constructed': ('demolished', 'app'),
    'appear': ('disappear', 'app'), 'appears': ('disappears', 'app'),
    'appeared': ('disappeared', 'app'),
    'show up': ('disappear', 'app'), 'shows up': ('disappears', 'app'),
    # disappearance / demolition -> appearance / construction
    'demolished': ('built', 'dis'),
    'disappear': ('appear', 'dis'), 'disappears': ('appears', 'dis'),
    'disappeared': ('appeared', 'dis'),
    'vanish': ('appear', 'dis'), 'vanishes': ('appears', 'dis'), 'vanished': ('appeared', 'dis'),
    'is removed': ('appears', 'dis'), 'are removed': ('appear', 'dis'),
    'has been removed': ('has appeared', 'dis'), 'have been removed': ('have appeared', 'dis'),
    # expansion -> contraction
    'widened': ('narrowed', 'exp'), 'broadened': ('narrowed', 'exp'),
    'expanded': ('contracted', 'exp'), 'expands': ('contracts', 'exp'), 'expand': ('contract', 'exp'),
    'increased': ('decreased', 'exp'), 'increases': ('decreases', 'exp'),
    'enlarged': ('reduced', 'exp'),
    # contraction -> expansion
    'narrowed': ('widened', 'con'), 'contracted': ('expanded', 'con'),
    'contracts': ('expands', 'con'), 'contract': ('expand', 'con'),
    'decreased': ('increased', 'con'), 'decreases': ('increases', 'con'),
    'reduced': ('increased', 'con'),
}
GROUP_ACTIONS = {'app': {'constructed', 'appearance'}, 'dis': {'disappearance', 'demolished'},
                 'exp': {'expansion'}, 'con': {'contraction'}}
# rho on parsed action ids
RHO_ACTION = {'constructed': 'demolished', 'demolished': 'constructed',
              'appearance': 'disappearance', 'disappearance': 'appearance',
              'expansion': 'contraction', 'contraction': 'expansion', 'no_change': 'no_change'}
DIRECTIONAL_ACTIONS = set(RHO_ACTION) - {'no_change'}

_TABLE_RE = re.compile(r'\b(' + '|'.join(sorted((re.escape(k) for k in _T), key=len, reverse=True)) + r')\b')
# every word that signals a temporal/directional event; anything matched here
# must be inside a table match, otherwise the caption is NOT reversible.
_LEX_RE = re.compile(
    r'\b(?:built|build|construct(?:ed|s|ion|ions|ing)?|appear\w*|disappear\w*|remov\w*|show(?:s|n)? up|develop\w*|erect\w*|'
    r'add(?:ed|s)?|new|newly|vanish\w*|clear(?:ed|s)?|cut|leveled|flatten\w*|demolish\w*|razed|torn|knocked|'
    r'expand\w*|increas\w*|decreas\w*|enlarg\w*|grew|grow\w*|grown|widen\w*|broaden\w*|extend\w*|contract\w*|'
    r'shr[ai]nk\w*|shrunk|reduc\w*|narrow\w*|diminish\w*|replac\w*|becom\w*|become|turn\w*|convert\w*|'
    r'transform\w*|rebuil\w*|reconstruct\w*|renovat\w*|complet\w*|form(?:ed|s)?|plant\w*|restor\w*|fad\w*|'
    r'wither\w*|thicker|lusher|greener|bigger|larger|wider|smaller|higher|risen|fewer|less|extra|additional|'
    r'original|former|old|existing|remaining|previous\w*|fill(?:ed|s)?|abandon\w*|finish\w*|reshap\w*|'
    r'divid\w*|rise[sn]?|comes?|make|made)\b')
_NEG_RE = re.compile(r"\b(?:no|not|without|never|nothing|none|n't|neither|nor)\b|n't")


def reverse_caption(caption):
    """Return (reversed_caption, info) or (None, reason).
    info = {'group': 'app'|'dis'|'exp'|'con', 'n_subs': int}."""
    c = caption.lower()
    if _NEG_RE.search(c):
        return None, 'negated_or_nochange_wording'
    subs = list(_TABLE_RE.finditer(c))
    if not subs:
        return None, 'no_directional_word'
    spans = [(m.start(), m.end()) for m in subs]
    for lm in _LEX_RE.finditer(c):
        if not any(s <= lm.start() and lm.end() <= e for s, e in spans):
            return None, 'uncovered_directional_word:' + lm.group(0)
    groups = {_T[m.group(1)][1] for m in subs}
    if len(groups) != 1:
        return None, 'compound_multi_direction'
    out = _TABLE_RE.sub(lambda m: _T[m.group(1)][0], c)
    return out, {'group': next(iter(groups)), 'n_subs': len(subs)}


def rho_action(action):
    return RHO_ACTION.get(action)


def check_purity(orig_caption, swapped_caption, parser, taxes=None):
    """Re-parse both captions; True iff both parse to exactly one event, the
    swapped action == rho(original action) and (for directional) the pair is
    an is_action_reversal. Returns (ok, reason, orig_action, swapped_action)."""
    from ..metrics.event import is_action_reversal
    eo = parser.parse(orig_caption, allow_multi_event=True)
    es = parser.parse(swapped_caption, allow_multi_event=True)
    if len(eo) != 1 or len(es) != 1:
        return False, 'reparse_not_single_event', None, None
    eo, es = eo[0], es[0]
    if es.action != rho_action(eo.action):
        return False, 'purity_action_mismatch', eo.action, es.action
    if eo.action != 'no_change' and not is_action_reversal(eo, es, taxes):
        return False, 'purity_not_reversal', eo.action, es.action
    if es.negated != eo.negated:
        return False, 'purity_negation_mismatch', eo.action, es.action
    return True, 'ok', eo.action, es.action

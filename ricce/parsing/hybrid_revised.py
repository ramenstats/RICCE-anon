"""Mapper-coverage-improvement variant (README
sec. 5: 'Improve mapper coverage -- especially source/target/location').

Subclasses HybridParser and only ADDS behaviour -- it never edits hybrid.py
or configs/ontology/*.yaml, so the frozen parser and every existing headline
number are completely unaffected. Three additive changes, each directly
requested in the README:
  (a) deterministic singularization fallback for plural/modified-noun forms
      ("residential buildings" -> also tries "residential building"), tried
      only when the unmodified text does not already resolve a field, so it
      can only ADD resolutions, never remove or change an existing one;
  (b) a broadened conversion source/target directional-phrase grammar:
      "converted from X to Y", active-voice "X replaced Y" (as well as the
      already-supported passive "Y was replaced by X"), and "became";
  (c) a companion ontology (configs/ontology_revised/) that adds a handful
      of common surface forms the README names explicitly
      (bungalow(s), villa(s), cottage(s), apartment(s), factories,
      warehouses) as ordinary aliases -- content, not code.
n-gram containment matching (already implemented in hybrid.py but off by
default) is turned on for this variant, since it is what actually resolves
reordered spatial phrases such as "right upper corner" against the existing
"upper right" alias (order-invariant token-set containment) -- no new
location aliases were needed for that case, just enabling the existing
opt-in stage.
"""
import re
from .hybrid import HybridParser

_IRREGULAR_SINGULAR = {
    'houses': 'house', 'buildings': 'building', 'structures': 'structure', 'roads': 'road', 'bridges': 'bridge',
    'trees': 'tree', 'plants': 'plant', 'streets': 'street', 'fields': 'field', 'shrubs': 'shrub',
    'villas': 'villa', 'bungalows': 'bungalow', 'cottages': 'cottage', 'apartments': 'apartment',
    'factories': 'factory', 'warehouses': 'warehouse', 'residences': 'residence', 'ponds': 'pond',
    'reservoirs': 'reservoir', 'railroads': 'railroad',
}


def _singularize_token(tok):
    if tok in _IRREGULAR_SINGULAR:
        return _IRREGULAR_SINGULAR[tok]
    if len(tok) > 3 and tok.endswith('ies'):
        return tok[:-3] + 'y'
    if len(tok) > 4 and tok.endswith(('ches', 'shes', 'xes', 'zes', 'ses')):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith('s') and not tok.endswith('ss'):
        return tok[:-1]
    return tok


def singularized(text):
    return ' '.join(_singularize_token(t) for t in text.split())


class HybridParserRevised(HybridParser):
    _CONVERSION_FROM_TO_RE = re.compile(r'(?:converted|changed)\s+from\s+(.+?)\s+to\s+(.+)')
    _CONVERSION_DIR_RE = re.compile(
        r'(.+?)\s+(?:was |were |has been |have been )?'
        r'(?:converted into|converted to|transformed into|changed into|changed to|turned into|turned to|became|'
        r'replaced by|replaced with)\s+(.+)')
    _CONVERSION_ACTIVE_REPLACED_RE = re.compile(r'(.+?)\s+(?:has |have |had )?replaced\s+(?!by\b|with\b)(.+)')

    def _exact(self, text, tax):
        best = super()._exact(text, tax)
        if best is not None:
            return best
        st = singularized(text)
        return super()._exact(st, tax) if st != text else None

    def _ngram(self, text, tax, max_n=3):
        best = super()._ngram(text, tax, max_n)
        if best is not None:
            return best
        st = singularized(text)
        return super()._ngram(st, tax, max_n) if st != text else None

    def _conversion_source_target(self, text):
        m = self._CONVERSION_FROM_TO_RE.search(text)
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
            return self.resolve_field(left, self.object_tax, 'object'), self.resolve_field(right, self.object_tax, 'object')
        m = self._CONVERSION_DIR_RE.search(text)
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
            return self.resolve_field(left, self.object_tax, 'object'), self.resolve_field(right, self.object_tax, 'object')
        m = self._CONVERSION_ACTIVE_REPLACED_RE.search(text)
        if m:
            new, old = m.group(1).strip(), m.group(2).strip()
            return self.resolve_field(old, self.object_tax, 'object'), self.resolve_field(new, self.object_tax, 'object')
        return None, None

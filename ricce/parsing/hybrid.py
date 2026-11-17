import re,json
from ..schema import Event

NUMBER_WORDS={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}

class HybridParser:
    # Negation trigger, applied to the normalized caption text (apostrophes
    # are stripped by normalize(), so "didn't" -> "didn t" etc.). "\bno\b" is
    # excluded when immediately followed by "longer" so that the genuine
    # disappearance alias "no longer present" is not misread as a negated
    # action -- every other "no X" / "not X" / contracted-negative phrasing
    # is treated as negation of the resolved action.
    _NEGATION_RE=re.compile(
        r'\bnot\b|\bwithout\b|\bno\b(?!\s+longer\b)|'
        r'\bdidn t\b|\bwasn t\b|\bweren t\b|\bhasn t\b|\bhaven t\b|\bisn t\b|\bdoesn t\b|\bhadn t\b'
    )
    # Directional "X was converted into/replaced by/turned into Y" pattern:
    # captures the source (group 1) and target (group 2) phrases on either
    # side of the directional marker so a conversion's source/target can be
    # resolved to distinct object nodes, enabling swap detection.
    _CONVERSION_DIR_RE=re.compile(
        r'(.+?)\s+(?:was |were |has been |have been )?'
        r'(?:converted into|converted to|transformed into|changed into|turned into|replaced by|replaced with)\s+(.+)'
    )

    def __init__(self,object_tax,action_tax,location_tax,quantity_tax,use_embeddings=True,embedding_model=None,similarity_threshold=0.58,
                 use_ngram=False,ngram_jaccard_threshold=0.5,use_llm_fallback=False,llm_model=None,use_ancestor_fallback=False,ancestor_fallback_max_depth=1):
        # Defaults preserve the exact/alias + semantic-retrieval pipeline
        # used for every already-computed result in this project (E04-E10,
        # ablations, significance). The n-gram / LLM-fallback / ancestor-
        # fallback stages are real and callable but opt-in, so enabling
        # them for the parser-stage ablation cannot silently change any
        # already-published number.
        self.object_tax=object_tax; self.action_tax=action_tax; self.location_tax=location_tax; self.quantity_tax=quantity_tax
        self.similarity_threshold=similarity_threshold; self.embedder=None; self._label_cache={}
        self.use_ngram=use_ngram; self.ngram_jaccard_threshold=ngram_jaccard_threshold
        self.use_llm_fallback=use_llm_fallback; self._llm=None; self._llm_model_name=llm_model
        self.use_ancestor_fallback=use_ancestor_fallback; self.ancestor_fallback_max_depth=ancestor_fallback_max_depth
        self.stage_counts={'exact':0,'ngram':0,'semantic':0,'llm_fallback':0,'ancestor_fallback':0,'unresolved':0}
        if use_embeddings and embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder=SentenceTransformer(embedding_model)
            except Exception: self.embedder=None
    def _label_embeddings(self,tax):
        key=id(tax)
        if key not in self._label_cache:
            labels=[]; ids=[]
            for nid,n in tax.nodes.items(): labels.append(n.label+(f'. {n.description}' if n.description else '')); ids.append(nid)
            emb=self.embedder.encode(labels,normalize_embeddings=True)
            self._label_cache[key]=(ids,emb)
        return self._label_cache[key]
    @staticmethod
    def normalize(text):
        text=text.lower().replace('-',' '); text=re.sub(r'[^a-z0-9\s]',' ',text); return re.sub(r'\s+',' ',text).strip()
    def _exact(self,text,tax):
        best=None; best_len=-1
        for nid,phrase in tax.labels_and_aliases():
            ph=self.normalize(phrase)
            if ph and re.search(rf'\b{re.escape(ph)}\b',text) and len(ph)>best_len: best=nid; best_len=len(ph)
        return best
    def _ngram(self,text,tax,max_n=3):
        # Containment match (all/most alias tokens present somewhere in the
        # caption, regardless of the caption's other words): catches
        # phrasings where alias words appear out of order or with
        # intervening tokens, which the exact substring/regex stage above
        # misses. A full-text Jaccard score is unsuitable here since a long
        # caption dilutes even a perfect alias match; this scores against
        # the alias's own token count instead.
        toks=set(text.split())
        if not toks: return None
        best=None; best_len=-1; best_score=0.0
        for nid,phrase in tax.labels_and_aliases():
            ph_toks=set(self.normalize(phrase).split())
            if len(ph_toks)<2: continue  # single-token aliases are exact-match's job
            n=min(max_n,len(ph_toks))
            inter=len(toks&ph_toks)
            score=inter/len(ph_toks)
            if inter>=n and score>=self.ngram_jaccard_threshold and len(ph_toks)>best_len:
                best=nid; best_len=len(ph_toks); best_score=score
        return best
    def _semantic(self,text,tax):
        if self.embedder is None: return None,0.0,None
        import numpy as np
        try:
            ids,label_emb=self._label_embeddings(tax)
            te=self.embedder.encode([text],normalize_embeddings=True)[0]
        except Exception:
            # A GPU embedder can fail mid-run (e.g. OOM under shared-GPU
            # contention) after having loaded successfully; degrade to the
            # exact/alias/n-gram/ancestor-fallback stages for the rest of
            # this run rather than crashing the whole experiment.
            self.embedder=None; return None,0.0,None
        s=label_emb@te; i=int(np.argmax(s)); score=float(s[i])
        return (ids[i] if score>=self.similarity_threshold else None),score,ids[i]
    def resolve_field(self,text,tax,field_name):
        best=self._exact(text,tax)
        if best is not None: self.stage_counts['exact']+=1; return best
        if self.use_ngram:
            best=self._ngram(text,tax)
            if best is not None: self.stage_counts['ngram']+=1; return best
        best,score,top_candidate=self._semantic(text,tax)
        if best is not None: self.stage_counts['semantic']+=1; return best
        if self.use_llm_fallback:
            best=self._llm_schema_fallback(text,tax,field_name)
            if best is not None: self.stage_counts['llm_fallback']+=1; return best
        if self.use_ancestor_fallback and top_candidate is not None:
            best=self._ancestor_fallback(top_candidate,tax)
            if best is not None: self.stage_counts['ancestor_fallback']+=1; return best
        self.stage_counts['unresolved']+=1
        return None
    def _load_llm(self):
        if self._llm is not None or not self.use_llm_fallback: return self._llm
        try:
            from ..vlm.qwen import TextOnlyLLM
            self._llm=TextOnlyLLM(self._llm_model_name)
        except Exception:
            self._llm=False
        return self._llm
    def _llm_schema_fallback(self,text,tax,field_name):
        # Constrained JSON parsing: only invoked once exact/n-gram/semantic
        # all fail to resolve, so this stage only ever fires on the hard
        # residual cases the cheaper stages could not handle.
        llm=self._load_llm()
        if not llm: return None
        candidates=sorted({nid for nid,_ in tax.labels_and_aliases()})
        prompt=(f'Caption fragment: "{text}"\n'
                f'Pick the single best matching {field_name} taxonomy id from this list, or null if none fit:\n'
                f'{json.dumps(candidates)}\n'
                f'Respond with ONLY a JSON object: {{"id": "<taxonomy_id_or_null>"}}')
        try:
            raw=llm.generate(prompt,max_new_tokens=32)
            m=re.search(r'\{.*\}',raw,re.S)
            if not m: return None
            obj=json.loads(m.group(0)); nid=obj.get('id')
            return nid if nid in tax.nodes else None
        except Exception:
            return None
    def _ancestor_fallback(self,best_id,tax):
        # Degrade an unresolved/low-confidence match to a coarse ancestor
        # node instead of leaving the field empty, capped at a shallow
        # depth so the fallback stays honestly vague rather than guessing.
        if best_id is None or best_id not in tax.nodes: return None
        anc=tax.ancestors(best_id)
        if len(anc)<=self.ancestor_fallback_max_depth+1: return anc[0] if anc else None
        return anc[self.ancestor_fallback_max_depth]
    def _quantity(self,text):
        m=re.search(r'\b(\d+)\b',text); exact=int(m.group(1)) if m else None
        if exact is None:
            for w,n in NUMBER_WORDS.items():
                if re.search(rf'\b{w}\b',text): exact=n; break
        if exact==0:return 'none',exact
        if exact==1:return 'single',exact
        if exact is not None and exact<=3:return 'few',exact
        if exact is not None and exact<=6:return 'several',exact
        if exact is not None and exact>6:return 'many',exact
        return self._exact(text,self.quantity_tax),exact
    # Conjunctions/punctuation that MAY join two independent clauses in a
    # compound caption (e.g. "a lane appears on the left and several houses
    # are built at the bottom."). Splitting on these blindly is unsafe --
    # "and" very often joins two objects/subjects within a single event
    # ("some houses and vegetation appear around the road" is ONE appearance
    # event with a compound object, not two events). _split_clauses only
    # proposes a split point; parse() below only accepts the split if BOTH
    # resulting clauses independently resolve to their own action verb,
    # which is the actual signal that two independent change events are
    # being described rather than one event with a compound argument.
    _CLAUSE_SPLIT_RE=re.compile(r'\s*;\s*|,?\s+\band\b\s+|\s+\bwhile\b\s+|\s+\bwhereas\b\s+',re.IGNORECASE)

    def _split_clauses(self,caption):
        parts=[p.strip(' .') for p in self._CLAUSE_SPLIT_RE.split(caption)]
        parts=[p for p in parts if p]
        return parts if len(parts)>=2 else None

    def _parse_single(self,caption):
        text=self.normalize(caption)
        action=self.resolve_field(text,self.action_tax,'action')
        obj=self.resolve_field(text,self.object_tax,'object')
        loc=self.resolve_field(text,self.location_tax,'location')
        qty,exact=self._quantity(text)
        if action=='no_change' or 'no change' in text or 'unchanged' in text:
            return [Event(change_state='no_change',action='no_change',text=caption)]
        negated=bool(self._NEGATION_RE.search(text))
        if negated and action and action!='no_change':
            return [Event(object=obj,action=action,location=loc,quantity=qty,exact_count=exact,change_state='no_change',negated=True,text=caption,confidence=0.9)]
        source=target=None
        if action in {'constructed','appearance'}: target=obj
        elif action in {'demolished','disappearance'}: source=obj
        elif action=='conversion':
            src_id,tgt_id=self._conversion_source_target(text)
            source=src_id; target=tgt_id
        state='change' if action and action!='no_change' else None
        return [Event(object=obj,action=action,source=source,target=target,location=loc,quantity=qty,exact_count=exact,change_state=state,text=caption,confidence=1.0 if action or obj else 0.5)]

    def _conversion_source_target(self,text):
        # For a directional "X was converted into/replaced by Y" pattern,
        # resolve source and target to *distinct* object taxonomy nodes on
        # each side of the directional marker so that source/target swap
        # (vegetation->building vs building->vegetation) is detectable as a
        # genuine contradiction rather than both sides collapsing to the
        # same single "object" match. Falls back to (None, None) when no
        # directional marker is present -- resolve_field's single global
        # object match is used elsewhere in that case.
        m=self._CONVERSION_DIR_RE.search(text)
        if not m: return None,None
        left,right=m.group(1).strip(),m.group(2).strip()
        src=self.resolve_field(left,self.object_tax,'object')
        tgt=self.resolve_field(right,self.object_tax,'object')
        return src,tgt

    def parse(self,caption,allow_multi_event=False):
        """Default behaviour (allow_multi_event=False) is byte-for-byte the
        original single-event parser -- every already-published result in
        this project called parse() with no arguments, so this stays exactly
        as before unless a caller opts in. allow_multi_event=True additionally
        tries to split compound captions into independent per-clause events
        (see _split_clauses / _CLAUSE_SPLIT_RE): a split is only accepted if
        EVERY resulting clause independently resolves its own action verb
        that isn't no_change -- a compound-subject caption like "some houses
        and vegetation appear" has only one action verb across all splits, so
        it correctly stays a single event."""
        if allow_multi_event:
            clauses=self._split_clauses(caption)
            if clauses:
                actions=[self.resolve_field(self.normalize(c),self.action_tax,'action') for c in clauses]
                if all(a and a!='no_change' for a in actions):
                    events=[]
                    for c in clauses: events.extend(self._parse_single(c))
                    return events
        return self._parse_single(caption)

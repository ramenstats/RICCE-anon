from ricce.ontology.tree import Taxonomy,Node
from ricce.schema import Event
from ricce.metrics.event import ricce_scores,event_similarity_pr

def small_tax():return Taxonomy([Node('root','root',None,[]),Node('built','built','root',[]),Node('building','building','built',[]),Node('residential','residential','building',[]),Node('road','road','built',[])],'root')
def action_tax():return Taxonomy([Node('temporal_state','temporal_state',None,[]),Node('change','change','temporal_state',[]),Node('appearance','appearance','change',[]),Node('constructed','constructed','appearance',[]),Node('disappearance','disappearance','change',[]),Node('demolished','demolished','disappearance',[]),Node('no_change','no_change','temporal_state',[])],'temporal_state')
def simple(root):return Taxonomy([Node(root,root,None,[])],root)

def test_coarse_correct():
    hp,hr,_=small_tax().hierarchical_scores('building','residential');assert hp==1.0 and 0<hr<1

def test_lca():
    t=small_tax();assert t.lca('building','road')=='built' and t.tree_distance('building','road')==2

def test_reversal():
    taxes={'object':small_tax(),'action':action_tax(),'location':simple('image'),'quantity':simple('quantity')};w={'object':1,'action':2,'change_state':1}
    s=ricce_scores([Event(object='building',action='demolished',change_state='change')],[Event(object='building',action='constructed',change_state='change')],taxes,w);assert s['ce_f1']<0.8

def test_nochange():
    taxes={'object':small_tax(),'action':action_tax(),'location':simple('image'),'quantity':simple('quantity')};w={'action':1,'change_state':2}
    s=ricce_scores([Event(action='constructed',change_state='change')],[Event(action='no_change',change_state='no_change')],taxes,w);assert s['ce_f1']==0

def test_unmatched_penalty():
    taxes={'object':small_tax(),'action':action_tax(),'location':simple('image'),'quantity':simple('quantity')};w={'object':1,'action':1,'change_state':1}
    g=[Event(object='building',action='constructed',change_state='change')];p=[Event(object='building',action='constructed',change_state='change'),Event(object='road',action='constructed',change_state='change')]
    s=ricce_scores(p,g,taxes,w);assert s['ce_precision']<s['ce_recall']

# --- Property tests for the CE-P/CE-R redesign (design-review critical
# review, item 2 / Section 7.1): hierarchical precision and recall must
# stay separate through matching and aggregation, not collapse to one hF
# scalar before being divided by event counts. ---

def test_ancestor_high_precision_lower_recall():
    tax=small_tax();taxes={'object':tax,'action':action_tax(),'location':simple('l'),'quantity':simple('q')}
    sp,sr=event_similarity_pr(Event(object='building'),Event(object='residential'),taxes,{'object':1})
    assert sp==1.0 and 0.0<sr<1.0

def test_descendant_lower_precision_high_recall():
    tax=small_tax();taxes={'object':tax,'action':action_tax(),'location':simple('l'),'quantity':simple('q')}
    sp,sr=event_similarity_pr(Event(object='residential'),Event(object='building'),taxes,{'object':1})
    assert 0.0<sp<1.0 and sr==1.0

def test_sibling_lower_precision_and_recall():
    tax=small_tax();taxes={'object':tax,'action':action_tax(),'location':simple('l'),'quantity':simple('q')}
    sp,sr=event_similarity_pr(Event(object='building'),Event(object='road'),taxes,{'object':1})
    assert sp<1.0 and sr<1.0

def test_missing_field_penalizes_recall_only():
    tax=small_tax();taxes={'object':tax,'action':action_tax(),'location':tax,'quantity':simple('q')}
    sp,sr=event_similarity_pr(Event(object='building'),Event(object='building',location='road'),taxes,{'object':1,'location':1})
    assert sp==1.0 and sr<1.0

def test_extra_field_penalizes_precision_only():
    tax=small_tax();taxes={'object':tax,'action':action_tax(),'location':tax,'quantity':simple('q')}
    sp,sr=event_similarity_pr(Event(object='building',location='road'),Event(object='building'),taxes,{'object':1,'location':1})
    assert sr==1.0 and sp<1.0

def test_ce_p_and_ce_r_diverge_with_equal_event_counts():
    # This is the exact defect the review flagged: under the old metric,
    # CE-P == CE-R == CE-F whenever predicted/reference event *counts*
    # matched, because hP/hR were collapsed into one scalar before ever
    # being divided by those counts. A single coarse-but-correct event must
    # now show CE-P > CE-R even though both sides have exactly one event.
    tax=small_tax();taxes={'object':tax,'action':action_tax(),'location':simple('l'),'quantity':simple('q')}
    s=ricce_scores([Event(object='building')],[Event(object='residential')],taxes,{'object':1})
    assert s['ce_precision']>s['ce_recall']

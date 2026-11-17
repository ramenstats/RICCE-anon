from dataclasses import dataclass
from collections import defaultdict
import yaml

@dataclass
class Node:
    id: str
    label: str
    parent: str|None
    aliases: list[str]
    description: str=''

class Taxonomy:
    def __init__(self,nodes,root):
        self.root=root; self.nodes={n.id:n for n in nodes}; self.children=defaultdict(list)
        for n in nodes:
            if n.parent is not None: self.children[n.parent].append(n.id)
        if root not in self.nodes: raise ValueError(f'Root {root} missing')
    @classmethod
    def from_yaml(cls,path):
        d=yaml.safe_load(open(path,'r',encoding='utf-8'))
        return cls([Node(n['id'],n.get('label',n['id']),n.get('parent'),list(n.get('aliases',[])),n.get('description','')) for n in d['nodes']],d['root'])
    def ancestors(self,node_id,include_self=True):
        if node_id not in self.nodes: return []
        out=[]; cur=node_id
        while cur is not None:
            out.append(cur); cur=self.nodes[cur].parent
        out.reverse(); return out if include_self else out[:-1]
    def depth(self,node_id): return max(0,len(self.ancestors(node_id))-1)
    def lca(self,a,b):
        aa,bb=self.ancestors(a),self.ancestors(b); l=self.root
        for x,y in zip(aa,bb):
            if x!=y: break
            l=x
        return l
    def tree_distance(self,a,b):
        l=self.lca(a,b); return self.depth(a)+self.depth(b)-2*self.depth(l)
    def hierarchical_scores(self,pred,gt,exclude_root=False):
        """Ancestor-overlap hierarchical precision/recall/F.
        exclude_root=False (default) counts the taxonomy root as a shared
        ancestor (historical evaluator, event.py). exclude_root=True removes
        it from both ancestor sets (Kiritchenko-style augmentation), keeping
        identity for pred==gt. The final evaluator never passes a root label
        here: it treats root-level labels as absent fields before calling
        this (event_v2.py, root_mode='unspecified')."""
        ap=set(self.ancestors(pred)); ag=set(self.ancestors(gt))
        if exclude_root:
            if pred==gt: return 1.0,1.0,1.0  # identity holds even for the root node itself
            ap=ap-{self.root}; ag=ag-{self.root}
        if not ap or not ag: return 0.0,0.0,0.0
        inter=len(ap&ag); hp=inter/len(ap); hr=inter/len(ag); hf=2*hp*hr/(hp+hr) if hp+hr else 0.0
        return hp,hr,hf
    def deepest_common_ancestor(self,node_ids):
        ids=[x for x in node_ids if x in self.nodes]
        if not ids: return self.root
        cur=ids[0]
        for n in ids[1:]: cur=self.lca(cur,n)
        return cur
    def labels_and_aliases(self):
        rows=[]
        for nid,n in self.nodes.items():
            rows.append((nid,n.label)); rows.extend((nid,a) for a in n.aliases)
        return rows

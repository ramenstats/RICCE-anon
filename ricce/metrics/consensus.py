from .event import event_similarity

def consensus_events(reference_event_sets,taxes,weights,merge_threshold=0.80):
    clusters=[]; total=max(1,len(reference_event_sets))
    for ridx,events in enumerate(reference_event_sets):
        for e in events:
            bi=None; bs=-1
            for i,c in enumerate(clusters):
                s=event_similarity(e,c['prototype'],taxes,weights)
                if s>bs: bi,bs=i,s
            if bi is not None and bs>=merge_threshold:
                clusters[bi]['events'].append(e); clusters[bi]['refs'].add(ridx)
            else: clusters.append({'prototype':e,'events':[e],'refs':{ridx}})
    out=[]
    for c in clusters:
        d=c['prototype'].to_dict(); d['support']=len(c['refs'])/total; out.append(d)
    return out

from __future__ import annotations
import math, re
from collections import Counter

TOKEN=re.compile(r"[\w-]+",re.UNICODE)

def tokenize(text: str):
    return [m.group(0).lower() for m in TOKEN.finditer(text)]

class BM25:
    def __init__(self,k1=1.5,b=0.75):
        self.k1=k1; self.b=b; self.docs={}

    def upsert(self,doc_id,text):
        self.docs[str(doc_id)]=tokenize(text)

    def delete(self,doc_id):
        self.docs.pop(str(doc_id),None)

    def search(self,query,limit=10):
        q=tokenize(query); n=len(self.docs)
        if not q or not n: return []
        lengths={k:len(v) for k,v in self.docs.items()}
        avg=sum(lengths.values())/n if n else 0.0
        dfs=Counter()
        for toks in self.docs.values():
            for term in set(toks): dfs[term]+=1
        out=[]
        for doc_id,toks in self.docs.items():
            tf=Counter(toks); score=0.0
            for term in q:
                df=dfs.get(term,0)
                idf=math.log(1.0+(n-df+0.5)/(df+0.5))
                freq=tf.get(term,0)
                denom=freq+self.k1*(1-self.b+self.b*(len(toks)/avg if avg else 0))
                if denom: score += idf*(freq*(self.k1+1))/denom
            if score > 0.0:
                out.append((doc_id,score))
        return sorted(out,key=lambda x:(-x[1],x[0]))[:limit]

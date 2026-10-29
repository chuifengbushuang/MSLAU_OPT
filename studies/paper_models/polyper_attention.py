"""Author Polyper refinement with bounded temporary attention allocation.

Queries are independent along the softmax batch dimension: processing query
rows in chunks leaves the attention formula intact. Keep the author's no_grad
spatial branch and clone/assignment gradient semantics, including empty regions.
"""
import torch
from torch.nn import functional as F
from einops import rearrange


def refine_forward(self,x,mask_d,mask_e,query_chunk=256):
    b,c,h,w = x.shape
    x = self.norm(x)
    y1,y2 = x*(1-mask_e),x*(1-mask_d)
    out_sa = x.clone()
    with torch.no_grad():
        for i in range(b):
            rows_d,cols_d = torch.where(mask_d[i,0] == 1)
            rows_e,cols_e = torch.where(mask_e[i,0] == 1)
            zd = x[i,:,rows_d,cols_d]
            ze = x[i,:,rows_e,cols_e]
            q = F.normalize(rearrange(zd,'(head c) z -> head z c',head=self.num_heads),dim=-1)
            k = F.normalize(rearrange(ze,'(head c) z -> head z c',head=self.num_heads),dim=-1)
            v = rearrange(ze,'(head c) z -> head z c',head=self.num_heads)
            for start in range(0,q.shape[1],query_chunk):
                qc = q[:,start:start+query_chunk]
                attention = (qc@k.transpose(-2,-1)).softmax(dim=-1)
                output = rearrange(attention@v+qc,'head z c -> (head c) z',head=self.num_heads)
                out_sa[i,:,rows_d[start:start+query_chunk],cols_d[start:start+query_chunk]] = output
    k2 = rearrange(y2,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    v2 = k2
    q2 = rearrange(y1,'b (head c) h w -> b head c (h w)',head=self.num_heads)
    q2,k2 = F.normalize(q2,dim=-1),F.normalize(k2,dim=-1)
    out2 = (q2@k2.transpose(-2,-1)).softmax(dim=-1)@v2+q2
    out2 = rearrange(out2,'b head c (h w) -> b (head c) h w',head=self.num_heads,h=h,w=w)
    return x+out_sa+out2

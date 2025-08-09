import torch 
import urllib.request
import json
import math 
import sys 
sys.path.append("../")
from tasks.objective import Objective

class DbHintObjective(Objective):
    ''' Find k good hints for 50 queries
    ''' 
    def __init__(
        self,
        dim=1,
        **kwargs,
    ):
        self.qids = [42, 102, 127, 128, 147, 160, 178, 232, 317, 350, 352, 354, 361, 410, 413, 471, 498, 521, 593, 719, 769, 833, 837, 914, 975, 1106, 1138, 1139, 1153, 1189, 1197, 1200, 1272, 1280, 1286, 1294, 1324, 1328, 1346, 1368, 1374, 1391, 1450, 1453, 1461, 1582, 1775, 1842, 1852, 1884]
        assert len(self.qids) == 50 # T=50
        self.already_executed = {}

        super().__init__(
            dim=dim,
            lb=0,
            ub=603,
            **kwargs,
        ) 

    def f(self, x):
        hid = math.floor(x) 
        if hid == 603:
            hid = 602 # rare case would only happen if we selected point exactly at boundary of space (x==603)
        if hid in self.already_executed:
            ys = self.already_executed[hid]
        else:
            ys = []
            for qid in self.qids:
                runtime = self.execute_query(qid=qid, hid=hid)
                reward = -1*runtime # turn min problem into max problem 
                ys.append(reward)
            self.already_executed[hid] = ys 
            self.num_calls += 1

        ys = torch.tensor(ys).to(dtype=self.dtype) # (T,)
        ys = ys.unsqueeze(0) # (1,T) 
        return ys 

    def execute_query(self, qid, hid):
        assert type(qid) is int, "qid must be an int"
        assert qid >= 0, "qid must be greater than or equal to 0"
        assert qid < 2000, "qid must be less than 2000"
        assert type(hid) is int, "hid must be an int"
        assert hid >= 0, "hid must be greater than or equal to 0"
        assert hid < 603, "hid must be less than 603"
        contents = urllib.request.urlopen(f"http://rmarcus.info/metaapi/execute?qid={qid}&hid={hid}")
        output_dict = json.load(contents)
        runtime = output_dict['latency'] # runtime in seconds
        # timeout = output_dict['timeout'] # boolean T/F, did this query time out
        return runtime


if __name__ == "__main__":
    obj = DbHintObjective()
    runtime = obj.execute_query(3, 15)
    print(f"Runtime should be around 30 and is: {runtime}")
    # 
    N = 13
    x_next = torch.rand(N, obj.dim)*(obj.ub - obj.lb) + obj.lb
    print(x_next.shape) # (13,1) (N,d)
    out_dict = obj(x_next) 
    y_next = out_dict["ys"]
    print(
        "Scores:", y_next, y_next.shape, # torch.Size([13, 50]) (N,T)
    )

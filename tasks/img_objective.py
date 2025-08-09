import torch 
import numpy as np 
import sys 
sys.path.append("../")
from tasks.objective import Objective
from tasks.utils.tonemap import pipeline, loadExposureSeq, map_parameters
import cv2 as cv
import time 
import pyiqa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImgObjective(Objective):
    ''' Img param optimization task 
        Eaach task is a metric (see self.metrics), solve with set of k<T sets of params 
    ''' 
    def __init__(
        self,
        version,
        dim=13,
        **kwargs,
    ):
        self.version = version 
        if version ==1:
            print("\n\n\n USING IMAGE TASK V1 \n\n\n")
            images, times = loadExposureSeq("../tasks/utils/hdr")
            calibrate = cv.createCalibrateDebevec()
            response = calibrate.process(images, times)
            merge_debevec = cv.createMergeDebevec()
            hdr = merge_debevec.process(images, times, response)
        elif version == 2:
            print("\n\n\n USING IMAGE TASK V2 \n\n\n")
            hdr = cv.imread("../tasks/utils/img2/cadik-desk01.hdr", cv.IMREAD_ANYDEPTH)
        else:
            assert 0, f"unrecognized image version: {version}"
        self.hdr = hdr/np.max(hdr)

        # Load metrics (all higher better in this case)
        # m1 = pyiqa.create_metric("qalign", device=device), # (1,5) # ** remove bc takes up 10GB yikes 
        times_to_load_metrics = []
        start_time = time.time()
        m1 = pyiqa.create_metric("nima", device=device) # (0,10), 
        load_time = time.time() - start_time
        print(f"Metric Load time: {load_time}\n")
        times_to_load_metrics.append(load_time)
        start_time = time.time()
        m2 = pyiqa.create_metric("nima-vgg16-ava", device=device) # (0,10), 
        load_time = time.time() - start_time
        print(f"Metric Load time: {load_time}\n")
        times_to_load_metrics.append(load_time)
        start_time = time.time()
        m3 = pyiqa.create_metric("topiq_iaa_res50", device=device) # (1,10), 
        load_time = time.time() - start_time
        print(f"Metric Load time: {load_time}\n")
        times_to_load_metrics.append(load_time)
        start_time = time.time()
        m4 = pyiqa.create_metric("laion_aes", device=device) # (1,10), 
        load_time = time.time() - start_time
        print(f"Metric Load time: {load_time}\n")
        times_to_load_metrics.append(load_time)
        start_time = time.time()
        m5 = pyiqa.create_metric("hyperiqa", device=device) # (0,1), 
        load_time = time.time() - start_time
        print(f"Metric Load time: {load_time}\n")
        times_to_load_metrics.append(load_time)
        start_time = time.time()
        m6 = pyiqa.create_metric("tres", device=device) # (0,100), 
        load_time = time.time() - start_time
        print(f"Metric Load time: {load_time}\n")
        times_to_load_metrics.append(load_time)
        start_time = time.time()
        m7 = pyiqa.create_metric("liqe", device=device) # (1,5),
        load_time = time.time() - start_time
        print(f"Metric Load time: {load_time}\n")
        times_to_load_metrics.append(load_time)
        # put metrics in list 
        self.metrics = [m1,m2,m3,m4,m5,m6,m7]

        # check min and max of metrics (for normalization)
        self.best_max_possible = [] # [10, 10, 10, 10, 1, 100, 5]
        self.worst_min_possible = [] # [0,  0, 1,  1,  0,  0,  1]
        # multiply by -1 for minimization ones so all scores are maximization
        # make_all_maximization_metrics = [] # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] all max! Pointless to use now.
        for metric in self.metrics: 
            score_range = metric.score_range
            score_range = score_range.replace("~", "").replace(",", "").split(" ")
            min_poss_score = int(score_range[0])
            max_poss_score = int(score_range[1])
            if metric.lower_better:
                assert 0, f"metric.lower_better is true for {metric}, no longer supported"
                # make_all_maximization_metrics.append(-1.0)
                self.best_max_possible.append(min_poss_score*-1)
                self.worst_min_possible.append(max_poss_score*-1)
            else:
                # make_all_maximization_metrics.append(1.0)
                self.best_max_possible.append(max_poss_score)
                self.worst_min_possible.append(min_poss_score)
        # self.make_all_maximization_metrics = torch.tensor(make_all_maximization_metrics)

        self.n_detail_layers = 3
        for ix, time_ in enumerate(times_to_load_metrics):
            print(f"Time to load metric{ix + 1}: {time_}")

        super().__init__(
            dim=dim,
            lb=0.0,
            ub=1.0,
            **kwargs,
        ) 

    def unnormalize_ys(self, ys):
        # y (T,)
        ys = ys.squeeze()
        assert ys.shape[0] == len(self.metrics)
        assert torch.is_tensor(ys)
        # un-normalize 
        ys_unnormed = torch.zeros_like(ys) # (T,)
        for t_ in range(len(ys)):
            min_poss = self.worst_min_possible[t_]
            max_poss = self.best_max_possible[t_]
            ys_unnormed[t_] = (ys[t_] * (max_poss - min_poss )) + min_poss  
        # Then, undo the making negative stuff for min/max 
        # ys_unnormed = ys_unnormed*self.make_all_maximization_metrics.to(dtype=self.dtype) 
        ys_unnormed = ys_unnormed.unsqueeze(0) # (1,T) 
        return ys_unnormed

    def f(self, x):
        self.num_calls += 1
        params = x.squeeze().cpu().numpy()
        params_mapped = map_parameters(params, self.n_detail_layers)
        out           = pipeline(self.hdr.copy(), self.n_detail_layers, params_mapped)
        out_rgb       = out[...,::-1].copy()
        out_tensor    = torch.tensor(out_rgb, dtype=self.dtype).permute([2,1,0]).unsqueeze(0)
        ys = list(map(lambda metric : metric(out_tensor).item(), self.metrics))
        ys = torch.tensor(ys).to(dtype=self.dtype) # (T,)
        # multiply lower_better metrics by -1 so all problems are maximization 
        # ys = ys*self.make_all_maximization_metrics.to(dtype=self.dtype) 
        # normalize each individual metric to a 0-1 range so they are weighted equally 
        for t_ in range(len(ys)):
            min_poss = self.worst_min_possible[t_]
            max_poss = self.best_max_possible[t_]
            ys[t_] = (ys[t_] - min_poss) / (max_poss - min_poss )
        ys = ys.unsqueeze(0) # (1,T) 
        return ys 


def bechmark_timing(version):
    obj = ImgObjective(version=version)
    for n_eval in [1,1,1,10,20,80]:
        x_next1 = torch.rand(n_eval, obj.dim)*(obj.ub - obj.lb) + obj.lb
        print("input x shape:", x_next1.shape)
        start_time = time.time()
        out_dict1 = obj(x_next1)
        time_taken = time.time() - start_time 
        y_next1 = out_dict1["ys"]
        print(f"Time taken to call oracle: {time_taken}, y shape:", y_next1.shape)


def test_oracle(version):
    N1 = 3
    N2 = 2
    obj = ImgObjective(version=version)
    x_next1 = torch.rand(N1, obj.dim)*(obj.ub - obj.lb) + obj.lb
    print(x_next1.shape)
    out_dict1 = obj(x_next1)
    y_next1 = out_dict1["ys"]

    x_next2 = torch.rand(N2, obj.dim)*(obj.ub - obj.lb) + obj.lb
    print(x_next2.shape)
    out_dict2 = obj(x_next2)
    y_next2 = out_dict2["ys"]
    print(
        "Scores:", y_next1, y_next2, y_next1.shape, y_next2.shape, 
    )

if __name__ == "__main__":
    version = int(sys.argv[1])
    # test_oracle(version=version)
    bechmark_timing(version=version)


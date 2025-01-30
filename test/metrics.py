# metrics.py

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance
import Levenshtein

def _sequence_distance(seq1, seq2):
    """
    Levenshtein distance between two sequences of activities.
    We join them with a special char, then do string distance.
    Normalized by max length.
    """
    s1 = "\u001f".join(seq1)
    s2 = "\u001f".join(seq2)
    dist = Levenshtein.distance(s1, s2)
    m = max(len(seq1), len(seq2), 1)
    return dist / float(m)


def _traces_from_log(df):
    """
    group by case -> sorted by start_time -> list of activities
    """
    traces = []
    for cid, group in df.groupby("case_id"):
        group_sorted = group.sort_values("start_time")
        acts = group_sorted["activity"].tolist()
        traces.append(acts)
    return traces


def cfld(alog, glog):
    """
    Control-Flow Log Distance
    """
    a_traces = _traces_from_log(alog)
    g_traces = _traces_from_log(glog)

    if not a_traces or not g_traces:
        return 0.0

    cost = np.zeros((len(a_traces), len(g_traces)), dtype=float)
    for i, at in enumerate(a_traces):
        for j, gt in enumerate(g_traces):
            cost[i, j] = _sequence_distance(at, gt)

    row_ind, col_ind = linear_sum_assignment(cost)
    total = cost[row_ind, col_ind].sum()
    avg = total / len(a_traces)
    return avg


def _get_ngrams(df, n=2):
    freq = {}
    for cid, group in df.groupby("case_id"):
        group_sorted = group.sort_values("start_time")
        acts = group_sorted["activity"].tolist()
        for i in range(len(acts) - n + 1):
            gram = tuple(acts[i:i+n])
            freq[gram] = freq.get(gram, 0) + 1
    return freq


def ngd(alog, glog, n=2):
    """
    N-Gram Distance
    """
    freqA = _get_ngrams(alog, n)
    freqG = _get_ngrams(glog, n)
    all_grams = set(freqA.keys()).union(set(freqG.keys()))
    vA = []
    vG = []
    for gram in sorted(all_grams):
        vA.append(freqA.get(gram, 0))
        vG.append(freqG.get(gram, 0))

    dist = wasserstein_distance(vA, vG)
    return dist


def _times_in_hours(df):
    # array of start_time minus min
    st = df["start_time"].dropna().tolist()
    if not st:
        return np.array([])
    mn = min(st)
    arr = []
    for t in st:
        delta = (t - mn).total_seconds() / 3600.0
        arr.append(delta)
    return np.array(arr)


def aed(alog, glog):
    """
    Absolute Event Distribution
    """
    arrA = _times_in_hours(alog)
    arrG = _times_in_hours(glog)
    if len(arrA)==0 or len(arrG)==0:
        return 0.0
    return wasserstein_distance(arrA, arrG)


def red(alog, glog):
    """
    Relative Event Distribution
    """
    def rel_times(df):
        out = []
        for cid, group in df.groupby("case_id"):
            sorted_ = group.sort_values("start_time")
            if len(sorted_)==0:
                continue
            mn = sorted_["start_time"].min()
            for t in sorted_["start_time"]:
                if pd.isna(t) or pd.isna(mn):
                    continue
                out.append((t - mn).total_seconds())
        return np.array(out)

    arrA = rel_times(alog)
    arrG = rel_times(glog)
    if len(arrA)==0 or len(arrG)==0:
        return 0.0
    return wasserstein_distance(arrA, arrG)


def ced(alog, glog):
    """
    Circadian Event Distribution
    """
    def hour_hist(df):
        freq = [0]*24
        for t in df["start_time"]:
            if pd.isna(t):
                continue
            freq[t.hour] += 1
        return np.array(freq, dtype=float)

    hA = hour_hist(alog)
    hG = hour_hist(glog)
    sumA = hA.sum()
    sumG = hG.sum()
    if sumA == 0 or sumG == 0:
        return 0.0

    cdfA = np.cumsum(hA)/sumA
    cdfG = np.cumsum(hG)/sumG
    dist = 0.0
    for i in range(23):
        dist += abs(cdfA[i] - cdfG[i])
    return dist


def cwd(alog, glog):
    """
    Circadian Workforce Distance
    """
    def hour_resource(df):
        # sets of resources per hour
        from collections import defaultdict
        hrset = [set() for _ in range(24)]
        for idx,row in df.iterrows():
            st = row["start_time"]
            r = row["resource"]
            if pd.isna(st) or not isinstance(r, str):
                continue
            hrset[st.hour].add(r)
        freq = [len(s) for s in hrset]
        return np.array(freq, dtype=float)

    fA = hour_resource(alog)
    fG = hour_resource(glog)
    sumA = fA.sum()
    sumG = fG.sum()
    if sumA==0 or sumG==0:
        return 0.0
    cdfA = np.cumsum(fA)/sumA
    cdfG = np.cumsum(fG)/sumG
    dist = 0.0
    for i in range(23):
        dist += abs(cdfA[i]-cdfG[i])
    return dist


def ctd(alog, glog):
    """
    Case Throughput Distribution
    """
    def durations(df):
        out = []
        for cid,group in df.groupby("case_id"):
            st = group["start_time"].min()
            et = group["end_time"].max()
            if pd.isna(st) or pd.isna(et):
                continue
            d = (et - st).total_seconds()
            out.append(d)
        return np.array(out, dtype=float)

    dA = durations(alog)
    dG = durations(glog)
    if len(dA)==0 or len(dG)==0:
        return 0.0
    return wasserstein_distance(dA, dG)


def compute_all_metrics(alog, glog):
    """
    Return a dict with the 7 metrics.
    """
    return {
        "CFLD": cfld(alog, glog),
        "NGD":  ngd(alog, glog, n=2),
        "AED":  aed(alog, glog),
        "RED":  red(alog, glog),
        "CED":  ced(alog, glog),
        "CWD":  cwd(alog, glog),
        "CTD":  ctd(alog, glog)
    }

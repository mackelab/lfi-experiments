import numpy as np
import pickle


def merge_raw_pickles(path):
    new_keys = [
        "bAP_somatic_spike",
        "bAP",
        "BAC_somatic",
        "BAC_caSpike",
        "BAC_spikecount",
        "step_mean_frequency",
        "step_AI_ISIcv",
        "step_doublet_ISI",
        "step_AP_height",
        "step_time_to_first_spike",
        "step_AHP_depth",
        "step_AHP_slow_time",
        "step_AP_width",
    ]

    merge_keys = [
        ["bAP_APwidth", "bAP_APheight", "bAP_spikecount"],
        ["bAP_att2", "bAP_att3"],
        ["BAC_ahpdepth", "BAC_APheight", "BAC_ISI"],
        ["BAC_caSpike_height", "BAC_caSpike_width"],
        ["BAC_spikecount"],
        ["mf1", "mf2", "mf3"],
        ["AI1", "AI2", "ISIcv1", "ISIcv2", "AI3", "ISIcv3"],
        ["DI1", "DI2"],
        ["APh1", "APh2", "APh3"],
        ["TTFS1", "TTFS2", "TTFS3"],
        ["fAHPd1", "fAHPd2", "fAHPd3", "sAHPd1", "sAHPd2", "sAHPd3"],
        ["sAHPt1", "sAHPt2", "sAHPt3"],
        ["APw1", "APw2", "APw3"],
    ]

    mydict = {nk: [] for nk in new_keys}
    thetas = []
    for file_ind in range(1749):
        with open(path+f"{file_ind}.pickle.hay_objectives", "rb") as handle:
            data = pickle.load(handle)

        for data_ind in range(len(data)):
            thetas.append(data[data_ind][0])
            for nk, mk in zip(new_keys, merge_keys):
                relevant_features = [data[data_ind][2][k] for k in mk]
                max_feature = max(relevant_features)
                mydict[nk].append(max_feature)

    np.savez(path+"theta_and_x.npz", theta=thetas, x=mydict)
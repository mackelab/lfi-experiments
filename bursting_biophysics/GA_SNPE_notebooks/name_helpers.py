import numpy as np
import pickle


def drop_columns(data_pd):
    data_no_ss = data_pd.drop(
        columns=[
            "BAC_APheight",
            "BAC_ISI",
            "BAC_ahpdepth",
            "BAC_caSpike_height",
            "BAC_caSpike_width",
            "BAC_spikecount",
            "bAP_APheight",
            "bAP_APwidth",
            "bAP_att2",
            "bAP_att3",
            "bAP_spikecount",
        ]
    )
    data_only_params = data_no_ss.drop(
        columns=[
            "model_id",
            "type_",
            "morphology",
            "lv",
            "seed",
            "max_",
            "gen",
            "hot_zone.outsidescale_sections",
            "morphology.filename",
            "BAC.stim.dist",
            "hot_zone.max_",
            "bAP.hay_measure.recSite2",
            "bAP.hay_measure.recSite1",
            "hot_zone.min_",
            "BAC.hay_measure.recSite",
            "sort_column",
        ]
    )
    return data_only_params


def select_some_morphologies(df, morph="85"):

    condition1 = df["morphology"] == morph
    df = df[condition1]
    return df


def drop_some_morphologies(df):

    condition1 = df["morphology"] == "84"
    df = df[np.invert(condition1)]

    condition2 = df["morphology"] == "85"
    df = df[np.invert(condition2)]

    condition3 = df["morphology"] == "91"
    df = df[np.invert(condition3)]

    return df

params_name_mapping = {
'ephys.NaTa_t.soma.gNaTa_tbar':'s.Na_t',
'ephys.Nap_Et2.soma.gNap_Et2bar':'s.Na_p',
'ephys.K_Pst.soma.gK_Pstbar':'s.K_p',
'ephys.K_Tst.soma.gK_Tstbar':'s.K_t',
'ephys.SK_E2.soma.gSK_E2bar':'s.SK',
'ephys.SKv3_1.soma.gSKv3_1bar':'s.Kv_3.1',
'ephys.Ca_HVA.soma.gCa_HVAbar':'s.Ca_H',
'ephys.Ca_LVAst.soma.gCa_LVAstbar':'s.Ca_L',
'ephys.CaDynamics_E2.soma.gamma':'s.Y',
'ephys.CaDynamics_E2.soma.decay':'s.T_decay',
 
'ephys.none.soma.g_pas':'s.leak',
'ephys.none.axon.g_pas':'ax.leak',
'ephys.none.dend.g_pas':'b.leak',
'ephys.none.apic.g_pas':'a.leak',
 
'ephys.NaTa_t.axon.gNaTa_tbar':'ax.Na_t',
'ephys.Nap_Et2.axon.gNap_Et2bar':'ax.Na_p',
'ephys.K_Pst.axon.gK_Pstbar':'ax.K_p',
'ephys.K_Tst.axon.gK_Tstbar':'ax.K_t',
'ephys.SK_E2.axon.gSK_E2bar':'ax.SK',
'ephys.SKv3_1.axon.gSKv3_1bar':'ax.Kv_3.1',
'ephys.Ca_HVA.axon.gCa_HVAbar':'ax.Ca_H',
'ephys.Ca_LVAst.axon.gCa_LVAstbar':'ax.Ca_L',
'ephys.CaDynamics_E2.axon.gamma':'ax.Y',
'ephys.CaDynamics_E2.axon.decay':'ax.T_decay',
 
'ephys.Im.apic.gImbar':'a.I_m',
'ephys.NaTa_t.apic.gNaTa_tbar':'a.Na_t',
'ephys.SKv3_1.apic.gSKv3_1bar':'a.Kv_3.1',
'ephys.Ca_HVA.apic.gCa_HVAbar':'a.Ca_H',
'ephys.Ca_LVAst.apic.gCa_LVAstbar':'a.Ca_L',
'ephys.SK_E2.apic.gSK_E2bar':'a.SK',
'ephys.CaDynamics_E2.apic.gamma':'a.Y',
'ephys.CaDynamics_E2.apic.decay':'a.T_decay',
 
'ephys.SKv3_1.apic.offset':'a.Kv_3.1_offset',
'ephys.SKv3_1.apic.slope':'a.Kv_3.1_slope',
'scale_apical.scale': 'a.scale'
}

def load_param_names():

    return ['ephys.CaDynamics_E2.apic.decay',
            'ephys.CaDynamics_E2.apic.gamma',
            'ephys.CaDynamics_E2.axon.decay',
            'ephys.CaDynamics_E2.axon.gamma',
            'ephys.CaDynamics_E2.soma.decay',
            'ephys.CaDynamics_E2.soma.gamma',
            'ephys.Ca_HVA.apic.gCa_HVAbar',
            'ephys.Ca_HVA.axon.gCa_HVAbar',
            'ephys.Ca_HVA.soma.gCa_HVAbar',
            'ephys.Ca_LVAst.apic.gCa_LVAstbar',
            'ephys.Ca_LVAst.axon.gCa_LVAstbar',
            'ephys.Ca_LVAst.soma.gCa_LVAstbar',
            'ephys.Im.apic.gImbar',
            'ephys.K_Pst.axon.gK_Pstbar',
            'ephys.K_Pst.soma.gK_Pstbar',
            'ephys.K_Tst.axon.gK_Tstbar',
            'ephys.K_Tst.soma.gK_Tstbar',
            'ephys.NaTa_t.apic.gNaTa_tbar',
            'ephys.NaTa_t.axon.gNaTa_tbar',
            'ephys.NaTa_t.soma.gNaTa_tbar',
            'ephys.Nap_Et2.axon.gNap_Et2bar',
            'ephys.Nap_Et2.soma.gNap_Et2bar',
            'ephys.SK_E2.apic.gSK_E2bar',
            'ephys.SK_E2.axon.gSK_E2bar',
            'ephys.SK_E2.soma.gSK_E2bar',
            'ephys.SKv3_1.apic.gSKv3_1bar',
            'ephys.SKv3_1.apic.offset',
            'ephys.SKv3_1.apic.slope',
            'ephys.SKv3_1.axon.gSKv3_1bar',
            'ephys.SKv3_1.soma.gSKv3_1bar',
            'ephys.none.apic.g_pas',
            'ephys.none.axon.g_pas',
            'ephys.none.dend.g_pas',
            'ephys.none.soma.g_pas',
            'scale_apical.scale'
    ]


def load_short_names():
    names = load_param_names()
    short_names = []
    for n in names:
        short_names.append(params_name_mapping[n])
    return short_names
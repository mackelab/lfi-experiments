def load_prior_max():
    """
    Returns the upper bound for the uniform prior as used by Arco.
    """
    return [200.0,
     0.050000000000000003,
     1000.0,
     0.050000000000000003,
     1000.0,
     0.050000000000000003,
     0.0050000000000000001,
     0.001,
     0.001,
     0.20000000000000001,
     0.01,
     0.01,
     0.001,
     1.0,
     1.0,
     0.10000000000000001,
     0.10000000000000001,
     0.040000000000000001,
     4.0,
     4.0,
     0.01,
     0.01,
     0.01,
     0.10000000000000001,
     0.10000000000000001,
     0.040000000000000001,
     1.0,
     0.0,
     2.0,
     2.0,
     0.0001,
     5.0000000000000002e-05,
     0.0001,
     5.02e-05,
     3.0]

def load_prior_min():
    """
    Returns the lower bound for the uniform prior as used by Arco.
    """
    return [20.0,
     0.00050000000000000001,
     20.0,
     0.00050000000000000001,
     20.0,
     0.00050000000000000001,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     -3.0,
     0.0,
     0.0,
     3.0000000000000001e-05,
     2.0000000000000002e-05,
     3.0000000000000001e-05,
     2.0000000000000002e-05,
     0.33300000000000002]

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

def load_ground_truth_params():
    return [ 137.86213603423801,
             0.0005793520824526776,
             199.12980481497891,
             0.00061080490759830625,
             152.16474193930151,
             0.0074243065368466803,
             0.0010965218089651857,
             0.00081867706020657862,
             0.00011435310571497434,
             0.0022763084379226854,
             0.0036986082079423594,
             0.00013883334761566004,
             3.2474860530531394e-06,
             0.0054268374162654382,
             0.10568666421909532,
             0.048126766921039982,
             0.094826660872338001,
             0.013854989311151315,
             3.9010342040060975,
             3.8851157748263354,
             0.009964343408409574,
             0.0060004974488750964,
             0.0012602755616811401,
             0.013922406480998821,
             0.062837104215625134,
             6.6838213839617901e-05,
             0.08311048073340864,
             -2.9836949894223825,
             1.9642986130169147,
             1.2999358521956366,
             4.4931548434199036e-05,
             2.062212836678345e-05,
             4.2205984329741197e-05,
             2.2409802171891654e-05,
             1.7109080877160283
             ]

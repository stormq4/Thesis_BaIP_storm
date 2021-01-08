

def local_results(i, AUC, QALY):
    name = {
        'Agent': i,
        'AUC-ROC Score': AUC,
        'Cost Reduction': QALY
    }
    return name
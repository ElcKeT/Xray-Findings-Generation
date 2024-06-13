import nlgeval

metrics_dict = nlgeval.compute_metrics(
    hypothesis="outputs/x_NLMCXR_ClsGenInt_DenseNet121_MaxView2_NumLabel114_History_Hyp.txt",
    references=[
        "outputs/x_NLMCXR_ClsGenInt_DenseNet121_MaxView2_NumLabel114_History_Ref.txt"
    ],
    no_skipthoughts=True,
    no_glove=True,
)

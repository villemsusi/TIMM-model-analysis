import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


parser = argparse.ArgumentParser(description="Test image classifier")


parser.add_argument("--rawdata", type=str)
parser.add_argument("--modeldata", type=str)

args = parser.parse_args()

if (args.rawdata is not None and args.modeldata is None) or (args.rawdata is None and args.modeldata is not None):
    parser.error("--rawdata and --modeldata need to specified together.")



def AddPrediction(raw_data):
    ## Map each label with ["nr of samples", ["nr of True Positives", "nr of False Positives "]]
    pred_counts = {}
    
    for ind in raw_data.index:
        model = raw_data["Model"][ind]
        target = raw_data["Target"][ind]
        prediction = raw_data["Predicted"][ind]
    
        # Initialise map entry
        if model not in pred_counts:
            pred_counts[model] = {}
        if target not in pred_counts[model]:
            pred_counts[model][target] = [0, [0, 0]]
        if prediction not in pred_counts[model]:
            pred_counts[model][prediction] = [0, [0, 0]]

        # Increase TP and FP counters
        if target.lower() == prediction.lower():
            pred_counts[model][target][1][0] += 1
        else:
            pred_counts[model][prediction][1][1] += 1
        pred_counts[model][prediction][0] += 1

    return pred_counts

def MetricCalc(pred_counts):
    ## Map each model with [precision, recall, f1]
    model_metrics = {}
    
    for model in pred_counts:
        if model not in model_metrics:
            model_metrics[model] = [0, 0, 0]
        model_counts = pred_counts[model]
        model_metric = model_metrics[model]
        for label in model_counts:
            label_counts = model_counts[label]
            # Precision
            model_metric[0] += label_counts[1][0] / (label_counts[1][0] + label_counts[1][1])
            # Recall
            model_metric[1] += label_counts[1][0] / label_counts[0]
        # Average
        model_metric[0] /= len(model_counts)
        model_metric[1] /= len(model_counts)

        model_metric[2] = 2*model_metric[0]*model_metric[1]/(model_metric[0]+model_metric[1])
    return model_metrics

def main():
    if args.rawdata != None:
        raw_data = pd.read_csv(args.rawdata)
        model_data = pd.read_csv(args.modeldata)
    else:
        raw_data = pd.read_csv(f"{source_dir}/pi_predictions/pi_pred_raw_large.csv")
        model_data = pd.read_csv(f"{source_dir}/pi_predictions/pi_pred_large.csv")
    
    model_data.sort_values(by="Model", inplace=True)

    
    pred_counts = AddPrediction(raw_data)

    model_metrics = MetricCalc(pred_counts)   

    sample_amount = len(raw_data["Target"]) / len(model_data["Model"])
    models = list(model_metrics.keys())
    f1_scores = [round(x[2], 3) for x in list(model_metrics.values())]
    inf_speeds = [round(x, 3) for x in list(model_data["AVG Time"])]
    flops = [round(x, 3) for x in list(model_data["FLOPs"] / list(model_data["AVG Time"]) / (sample_amount))]
    term = list(map(lambda x, y, z: x / y / z, f1_scores, flops, inf_speeds))
    normalised_term = [round(x, 3) for x in normalize([term], 'max')[0]]

    metric_df = pd.DataFrame({
        "Model": models,
        "F1": f1_scores,
        "Speed": inf_speeds,
        "FLOPs": flops,
        "Term": normalised_term

    })
    metric_df.sort_values(by="Model", inplace=True, ascending=False)


    fx = metric_df.plot.barh(x="Model", y="F1", legend=False, color=plt.cm.Dark2([x%7 for x in range(30)]))
    fx.bar_label(fx.containers[0], padding=-100)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(f"{source_dir}/graphs/f1.pdf", bbox_inches="tight")

    ix = metric_df.plot.barh(x="Model", y="Speed", legend=False, color=plt.cm.Dark2([x%7 for x in range(30)]))
    ix.bar_label(ix.containers[0], padding=10)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(f"{source_dir}/graphs/speed.pdf", bbox_inches="tight")

    ox = metric_df.plot.barh(x="Model", y="FLOPs", legend=False, color=plt.cm.Dark2([x%7 for x in range(30)]))
    ox.bar_label(ox.containers[0], padding=10)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(f"{source_dir}/graphs/flops.pdf", bbox_inches="tight")
    
    tx = metric_df.plot.barh(x="Model", y="Term", legend=False, color=plt.cm.Dark2([x%7 for x in range(30)]))
    tx.bar_label(tx.containers[0], padding=10)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(f"{source_dir}/graphs/term.pdf", bbox_inches="tight")
        
    #plt.show()
    


if __name__ == "__main__":
    source_dir = Path(__file__).resolve().parent.parent
    main()
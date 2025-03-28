import sys
sys.path.append('../')
import os
import gc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from gears import PertData, GEARS
from gears.inference import evaluate, compute_metrics, deeper_analysis
import scanpy as sc
predict_perturbation = 'MAP2K3+MAP2K6'
model_path = 'Norman2019 self filter Data_train_condition_0.25_uncertainty'

def plot(stats_summary, metric_summary):
    print(type(stats_summary))
    print(type(metric_summary))
    if isinstance(stats_summary, list) and len(stats_summary) > 0:
        print(f"📌 stats_summary 첫 번째 요소 타입: {type(stats_summary[0])}")
        print(f"📌 stats_summary 예제 데이터 (앞 5개): {stats_summary[:5]}")
        
    if isinstance(metric_summary, list) and len(metric_summary) > 0:
        print(f"📌 metric_summary 첫 번째 요소 타입: {type(metric_summary[0])}")
        print(f"📌 metric_summary 예제 데이터 (앞 5개): {metric_summary[:5]}")

    ax = sns.regplot(x = np.array(stats_summary),
                    y = np.array(metric_summary), 
                    color = pal[2],
                    ci = None
                    )

    sns.despine()

    plt.ylabel("Pearson Correlation with True Delta \n Expression Across All Genes",labelpad=10)
    plt.xlabel("Predicted Uncertainty",labelpad=10)
    plt.tick_params(axis='x', which='major', pad=10)
    plt.tick_params(axis='y', which='major', pad=5)
    plt.savefig(f'{model_path}/uncertainty.pdf', bbox_inches='tight')
    plt.close()
    top = np.quantile(np.array(list(pert2unc.values())), 0.95)

    m = 'pearson_delta'
    ax = sns.distplot([i[m] for k, i in out.items() if (pert2unc[k][0] < top) and (m in i)], hist = False, color = "black")
    ax = sns.distplot([i[m] for k, i in out.items() if (pert2unc[k][0] > top) and (m in i)], hist = False, color = "Red")

    print('---- ' + m + '----')
    total_mean = np.mean([i[m] for k, i in out.items() if (pert2unc[k][0] < top) and (m in i)])
    after_filter_mean = np.mean([i[m] for k, i in out.items() if (pert2unc[k][0] > top) and (m in i)])
    print('Total Mean: ' + str(total_mean))
    print('After uncertainty filter Mean: ' + str(after_filter_mean))
    print('Enrichment: ' + str((after_filter_mean - total_mean)/total_mean))

    sns.despine()
    ax.set_xlim((-0.24,1.24))
    plt.xlabel("Pearson Correlation with True Delta \n Expression Across All Genes",labelpad=10)
    plt.ylabel("Density of Perturbations",labelpad=10)
    plt.tick_params(axis='x', which='major', pad=10)
    plt.tick_params(axis='y', which='major', pad=5)
    plt.savefig(f'{model_path}/prioritize_uncertainty.pdf', bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    pert_data = PertData(data_path = './data', model_path = model_path)
    # pert_data.load(data_name = 'norman')
    # pert_data.new_data_process("GEARS_Norman_raw_data", adata=sc.read_h5ad(r"C:\Users\USER\Desktop\GEARS2\data\GEARS_Norman_raw_data.h5ad"))
    pert_data.MyfileLoad(data_name="processing_real_norman" ,adata_path = r"C:\Users\USER\Desktop\GEARS2\data\processing_real_norman\perturb_processed.h5ad", dataset_fname=r"C:\Users\USER\Desktop\GEARS2\data\processing_real_norman\data_pyg\cell_graphs.pkl")
    pert_data.prepare_split(split = 'simulation', seed = 1)
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 64)
    gears_model = GEARS(pert_data, device = 'cuda',
                            weight_bias_track = False,
                            proj_name = 'pertnet',
                            exp_name = 'pertnet',
                            model_path = model_path)
    if os.path.exists(os.path.join(model_path, 'config.pkl')):
        gears_model.load_pretrained(gears_model.model_path)
        if gears_model.config['uncertainty'] :
            gears_model.plot_perturbation(predict_perturbation,f"{model_path}/{predict_perturbation}_Best_Model", box_plot = True)
            test_res = evaluate(gears_model.dataloader['test_loader'], gears_model.model, gears_model.config['uncertainty'], gears_model.device)
            test_metrics, test_pert_res = compute_metrics(test_res)
            out = deeper_analysis(gears_model.adata, test_res)
            pert2unc = pd.DataFrame(tuple(zip(test_res['pert_cat'], np.mean(test_res['logvar'], axis = 1)))).groupby(0).agg(np.mean)
            pert2unc = dict(zip(pert2unc.index.values, pert2unc.values))

            metric = 'pearson_delta'
            stats_summary = [np.exp(-pert2unc[i][0]) for i in test_pert_res.keys() if metric in out[i]]
            metric_summary = [out[i][metric] for i in test_pert_res.keys() if metric in out[i]]
            print(pearsonr(stats_summary, metric_summary))
            
            sns.set(rc={'figure.figsize':(6,6)})
            sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.8)
            pal = sns.color_palette("Set2").as_hex()
            plot(stats_summary, metric_summary)
        else:
            gears_model.Generation([['MYC'],['ctrl'],['CCND3','MYC']], num_cells = 3000)
    else:
        gears_model.model_initialize(hidden_size = 32, uncertainty = True)
        # gears_model.tunable_parameters()
        gears_model.train(predict_perturbation = predict_perturbation, epochs = 10, lr = 1e-3)
        gears_model.save_model(gears_model.model_path)
        gears_model.load_pretrained(gears_model.model_path)
        # gears_model.predict([['FEV'], ['FEV', 'AHR']])
        # gears_model.plot_perturbation(predict_perturbation,f"{model_path}/{predict_perturbation}_Best_Model")
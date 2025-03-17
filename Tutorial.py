if __name__ == "__main__":
    import sys
    sys.path.append('../')
    import os
    import gc
    from gears import PertData, GEARS
    import scanpy as sc

    predict_perturbation = 'MAP2K3+MAP2K6'
    model_path = 'Tutorial Norman Data_train_condition_0.75'
    pert_data = PertData(data_path = './data', model_path = model_path)
    pert_data.load(data_name = 'norman')
    # pert_data.new_data_process("processing_real_norman", adata=sc.read_h5ad(r"C:\Users\USER\Desktop\GEARS2\data\processing_real_norman.h5ad"))
    # pert_data.MyfileLoad(data_name="processing_real_norman" ,adata_path = r"C:\Users\USER\Desktop\GEARS2\data\processing_real_norman\perturb_processed.h5ad", dataset_fname=r"C:\Users\USER\Desktop\GEARS2\data\processing_real_norman\data_pyg\cell_graphs.pkl")
    pert_data.prepare_split(split = 'simulation', seed = 1)
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 64)
    gears_model = GEARS(pert_data, device = 'cuda',
                            weight_bias_track = False, 
                            proj_name = 'pertnet',
                            exp_name = 'pertnet',
                            model_path = model_path)
    gears_model.model_initialize(hidden_size = 32)
    # gears_model.tunable_parameters()
    gears_model.train(predict_perturbation = predict_perturbation, epochs = 20, lr = 1e-3)
    gears_model.save_model(gears_model.model_path)
    gears_model.load_pretrained(gears_model.model_path)
    # gears_model.predict([['FEV'], ['FEV', 'AHR']])
    gears_model.plot_perturbation(predict_perturbation,f"{predict_perturbation}_Best_Model")

    # model_path = 'test_model_train_mydata_1.0'
    # if os.path.exists(model_path):
    #     iteration = 0
    #     while iteration < 20:
    #         gears_model = GEARS(pert_data, device = 'cuda', 
    #                         weight_bias_track = False, 
    #                         proj_name = 'pertnet',
    #                         exp_name = 'pertnet')
    #         gears_model.load_pretrained(model_path)
    #         gears_model.train(epochs=1, lr=1e-3)
    #         gears_model.save_model(model_path)
    #         iteration += 1
    #         save_flie = f"MAP2K3+MAP2K6_{iteration}"
    #         gears_model.plot_perturbation('MAP2K3+MAP2K6', save_flie)

    #         del gears_model
    #         gc.collect()



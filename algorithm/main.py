from psoCNN import psoCNN
import numpy as np
import time
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
from config import Config 
import json

def run_algorithm():
    config = Config()
    print(f"Using device: {config.device}")
    
    if not os.path.exists(config.results_path):
            os.makedirs(config.results_path)
    
    with open(config.results_path + "info.json", "w") as outfile:
        json.dump(config.__dict__, outfile, indent = 4, default=lambda o: str(o))
        # outfile.write(json_data)

    all_gBest_metrics = np.zeros((config.number_runs, 2))
    
    runs_time = []
    best_gBest_acc = -1
    best_hist_test = []
    best_hist_train = []

    for i in range(config.number_runs):
        print("Run number: " + str(i))
        start_time = time.time()        
        pso = psoCNN(config)
        pso.fit(config.Cg, config.dropout)
        print(pso.gBest_acc)

        print('gBest architecture: ')
        print(pso.gBest)
    
        end_time = time.time()

        running_time = end_time - start_time
        runs_time.append(running_time)

        # Fully train the gBest model found
        pso.fit_gBest(epochs=config.epochs_full_training, dropout_rate=config.dropout)

        # Evaluate the fully trained gBest model
        gBest_metrics = pso.evaluate_gBest()
        
        if gBest_metrics[1] >= best_gBest_acc:
            best_gBest_acc = gBest_metrics[1]
            best_hist_train = pso.gBest_acc
            best_hist_test = pso.gBest_test_acc
            best_hist_measure = pso.gBest_measure

            # Save best gBest model structure 
            with open(config.results_path + "best-gBest-model.txt", "w") as f:
                f.write(pso.gBest.model.__str__())
            
            # Save best gBest model weights to pt file
            torch.save(pso.gBest.model.state_dict(), config.results_path + "best-gBest-weights.pt" )
            

        all_gBest_metrics[i, 0] = gBest_metrics[0]
        all_gBest_metrics[i, 1] = gBest_metrics[1]

        # print("This run took: " + str(running_time) + " seconds.")

        # Compute mean accuracy of all runs
        all_gBest_mean_metrics = np.mean(all_gBest_metrics, axis=0)

        results = {
            # "all_gBest_acc": list(all_gBest_metrics)[:,1],
            "runs_time": runs_time, 
            "all_gBest_mean_loss": all_gBest_mean_metrics[0],
            "all_gBest_mean_acc": all_gBest_mean_metrics[1],
            "best_hist_test": list(best_hist_test),
            "best_hist_train": list(best_hist_train),
            "best_hist_measure": list(best_hist_measure),
        }
        
        with open(config.results_path + "results.json", "w") as outfile:
            # json.dump(results, outfile, indent = 4, default=lambda o: '<not serializable>')
            json.dump(results, outfile, indent = 4)
    
if __name__ == '__main__':
    run_algorithm()
    
    
    
    
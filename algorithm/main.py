from psoCNN import psoCNN
import numpy as np
import time
import os
import matplotlib
import matplotlib.pyplot as plt
import argparse
import torch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", help="root path at which data is stored", type=str, default="../data")
    parser.add_argument(
        "--gpu", help="whether to use gpu or not", type=int, default=0)
    args = parser.parse_args()
    
    
    ######## Algorithm parameters ##################
    
    if torch.cuda.is_available() and args.gpu != 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset = "mnist"
    dataset = "cifar10"
        
    number_runs = 10
    number_iterations = 10
    population_size = 20

    batch_size_pso = 32
    batch_size_full_training = 32

    epochs_pso = 1
    epochs_full_training = 100

    max_conv_output_channels = 256
    max_fully_connected_neurons = 300

    min_layer = 3
    max_layer = 20

    # Probability of each layer type (should sum to 1)
    probability_convolution = 0.6
    probability_pooling = 0.3
    probability_fully_connected = 0.1

    max_conv_kernel_size = 7

    Cg = 0.5
    dropout = 0.5

    ########### Run the algorithm ######################
    results_path = "../results/" + dataset + "/"

    if not os.path.exists(results_path):
            os.makedirs(results_path)

    all_gBest_metrics = np.zeros((number_runs, 2))
    runs_time = []
    all_gbest_par = []
    best_gBest_acc = 0
    
    

    for i in range(number_runs):
        print("Run number: " + str(i))
        start_time = time.time()
        pso = psoCNN(dataset=dataset, n_iter=number_iterations, pop_size=population_size,
                     batch_size=batch_size_pso, epochs=epochs_pso, min_layer=min_layer, max_layer=max_layer,
                     conv_prob=probability_convolution, pool_prob=probability_pooling,
                     fc_prob=probability_fully_connected, max_conv_kernel=max_conv_kernel_size,
                     max_out_ch=max_conv_output_channels, max_fc_neurons=max_fully_connected_neurons,
                     dropout_rate=dropout, root=args.root, device=device)

        pso.fit(Cg=Cg, dropout_rate=dropout)

        print(pso.gBest_acc)

        # Plot current gBest
        matplotlib.use('Agg')
        plt.plot(pso.gBest_acc)
        plt.xlabel("Iteration")
        plt.ylabel("gBest acc")
        plt.savefig(results_path + "gBest-iter-" + str(i) + ".png")
        plt.close()

        print('gBest architecture: ')
        print(pso.gBest)
    
        np.save(results_path + "gBest_inter_" + str(i) + "_acc_history.npy", pso.gBest_acc)

        np.save(results_path + "gBest_iter_" + str(i) + "_test_acc_history.npy", pso.gBest_test_acc)

        end_time = time.time()

        running_time = end_time - start_time

        runs_time.append(running_time)

        # Fully train the gBest model found
        acc = pso.fit_gBest(batch_size=batch_size_full_training, epochs=epochs_full_training, dropout_rate=dropout)
        # all_gbest_par.append(n_parameters)

        # Evaluate the fully trained gBest model
        gBest_metrics = pso.evaluate_gBest(batch_size=batch_size_full_training)

        if gBest_metrics[1] >= best_gBest_acc:
            best_gBest_acc = gBest_metrics[1]

            # Save best gBest model structure 
            with open(results_path + "best-gBest-model.txt", "w") as f:
                f.write(pso.gBest.model.__str__())
            
            # Save best gBest model weights to HDF5 file
            torch.save(pso.gBest.model.state_dict(), results_path + "best-gBest-weights.pt" )
            # pso.gBest.model.save_weights(results_path + "best-gBest-weights.h5")

        all_gBest_metrics[i, 0] = gBest_metrics[0]
        all_gBest_metrics[i, 1] = gBest_metrics[1]

        print("This run took: " + str(running_time) + " seconds.")

         # Compute mean accuracy of all runs
        all_gBest_mean_metrics = np.mean(all_gBest_metrics, axis=0)

        np.save(results_path + "/time_to_run.npy", runs_time)

        # Save all gBest metrics
        np.save(results_path + "/all_gBest_metrics.npy", all_gBest_metrics)

        # Save results in a text file
        output_str = "All gBest number of parameters: " + str(all_gbest_par) + "\n"
        output_str = output_str + "All gBest test accuracies: " + str(all_gBest_metrics[:,1]) + "\n"
        output_str = output_str + "All running times: " + str(runs_time) + "\n"
        output_str = output_str + "Mean loss of all runs: " + str(all_gBest_mean_metrics[0]) + "\n"
        output_str = output_str + "Mean accuracy of all runs: " + str(all_gBest_mean_metrics[1]) + "\n"

        print(output_str)

        with open(results_path + "/final_results.txt", "w") as f:
            try:
                print(output_str, file=f)
            except SyntaxError:
                print >> f, output_str
from simulation import Simulation 
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')



if __name__ == '__main__':
  X_sim = Simulation(TensorInfoBucket([n,n,n], k = 15, rank = 10, s=30), \
        RandomInfoBucket(random_seed = 1), gen_typ = 'id', noise_level=0.1) 
  




# Define the simulation object (input matrics: Tinfo_bucket, Rinfo_bucket, gen_typ, noise_level )

# Run and Save the raw simulation result --> Define simulation_result object 

# Evaluate the mse and running time from the simulation result  
# Plot is associated with the evaluation function. 
from location import location
from tools import MainTools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# parameter setup
Rm = 100                    # Radius of Macrocell
R0 = 10 
N = 3                       # number of v2v pairs
num_veh = 6                 # number of vehicles

Nmonte = 100                     # monte carlo iteration
number_of_iterations = 1000

User_antenna_gain_dB = 4
User_antenna_gain = 10 ** (User_antenna_gain_dB/10)
BS_antenna_gain_dB = 17
BS_antenna_gain = 10 ** (BS_antenna_gain_dB/10)
BS_antenna_height = 25                  # BS height in meters
BS_NoiseFigure = 5                      # BS noise figure 5 dB

veh_antenna_height_tx = 1.5             # vehicle antenna height transmitter, in meters
veh_antenna_height_rx = 1.5             # vehicle antenna height receiver, in meters
veh_antenna_gain = 3                    # vehicle antenna gain 3 dBi
veh_NoiseFigure = 9                     # vehicle noise figure 9 dB

environment_rows = 2                     # states
environment_columns = 10                 # actions

pi_max_dBm = 23                            # max power in dBm
pi_min_dBm = 0                             # min power in dBm
steps = 9

pi_max = (10 ** (pi_max_dBm/10))/1000              # max power in watt
pi_min = (10 ** (pi_min_dBm/10))/1000              # min power in watt

epsilon = 0.5
alpha = 0.1                             # learning_rate
discount_factor = 0.9

std_Shadowing = 3                        # shadowing std deviation for v2v

threshold_dB = 0                            # for sinr values in dB
threshold = 10 ** (threshold_dB/10)         # for sinr values in watt

sigma_dBm = -114                         # noise power in dBm
sigma = (10 ** (sigma_dBm/10))/1000

fc = 2                               # carrier frequency 2 GHz
w = 1                                # bandwidth (1 Mhz) 

# main code

q_values_mc_list = [[None for x in range(number_of_iterations)] for y in range(Nmonte)]
q_values_unv = []

mt = MainTools(N=N)


for mc in tqdm(range(Nmonte)):
    
    count = 0

    q_values = []
    mat_state = []
    for i in range(N):
        
        q_values.append(np.zeros((environment_rows,environment_columns)))
        mat_state.append(np.zeros((1,environment_columns)))
    
    # print('initial q value matrices = ','\n',q_values,'\n')    
    # print('initial state matrices = ','\n',mat_state,'\n') 

    distance = location(R0,Rm,N)
    # print('distance between transmitters and receivers = ','\n',distance,'\n')
    
    
    
    large_scale_fading = mt.large_scale_fading(std_Shadowing,distance,veh_antenna_height_tx,veh_antenna_height_rx,fc,veh_antenna_gain,veh_NoiseFigure)
    # print('large scale fading values = ','\n',large_scale_fading,'\n')

    

    small_scale_fading = mt.small_scale_fading()
    # print('small scale fading values = ','\n',small_scale_fading,'\n')
    

    channel_power_gain = mt.channel_power_gain(small_scale_fading,large_scale_fading)
    # print('channel power gain values = ','\n',channel_power_gain,'\n')

    

    action_vector = mt.action_list(pi_min,pi_max,steps)
    # print('action list values = ','\n',action_vector,'\n')

    
    
    initial_s = []
    new_s = None

    pi_array_b_vector = mt.epsilon_greedy(epsilon,pi_min,pi_max,steps,count,q_values,initial_s,new_s)
    # print('selected actions and their columns number = ','\n',pi_array_b_vector,'\n')

    

    gamma_val = mt.sinr(pi_array_b_vector,channel_power_gain,sigma)
    # print('SINR values = ','\n',gamma_val,'\n')

    

    reward_array = mt.reward_function(w,gamma_val,threshold,pi_array_b_vector)
    # print('reward values = ','\n',reward_array,'\n')

    initial_a = pi_array_b_vector[1][:]

    for i in range(N):

        if gamma_val[i] >= threshold:
            initial_s.append(1)
        else:
            initial_s.append(0)

    # print('initial state values = ','\n',initial_s,'\n')
    # print('initial action values = ','\n',initial_a,'\n')


    for episode in range (number_of_iterations):
        
        count += 1

        pi_array_b_vector = mt.epsilon_greedy(epsilon,pi_min,pi_max,steps,count,q_values,initial_s,new_s)
        # print('selected actions and their columns number = ','\n',pi_array_b_vector,'\n')

        gamma_val = mt.sinr(pi_array_b_vector,channel_power_gain,sigma)
#         print('SINR values = ','\n',gamma_val,'\n')

        reward_array = mt.reward_function(w,gamma_val,threshold,pi_array_b_vector)
#         print('reward values = ','\n',reward_array,'\n') 

        for i in range(N):

            if episode == 0:
                a = initial_a[i]
                s = initial_s[i]
            else:
                s = new_s
                a = new_a

            if gamma_val[i] >= threshold:
                new_s = 1
                new_a = pi_array_b_vector[1][i]
#                 mat_state[i][0,new_a] = 1
                q_max = np.max(q_values[i][1,:])
            else:
                new_s = 0
                new_a = pi_array_b_vector[1][i]
#                 mat_state[i][0,new_a] = 0
                q_max = np.max(q_values[i][0,:])

            new_q_value = q_values[i][s,a] + (alpha * (reward_array[i] + (discount_factor * (q_max)) - q_values[i][s,a]))
            new_q_value_2 = float("{:.3f}".format(new_q_value))
            q_values[i][s,a] = new_q_value_2
        
        q_values_mc_list[mc][episode] = np.copy(q_values)
        epsilon -= epsilon / number_of_iterations

# print ('q values matrices for monte carlo = ','\n', q_values_mc_list,'\n')    
    
q_values_mc_list_2 = [[None for x in range(number_of_iterations)] for y in range(Nmonte)]
for i in range (Nmonte):
    for j in range(number_of_iterations):
        a = np.sum(q_values_mc_list[i][j],axis=0)
        b = a / N
        q_values_mc_list_2[i][j] = b
        
# print(q_values_mc_list_2)

q_values_mc_list_3 = []
for j in range(number_of_iterations):
    answer = []
    for i in range(Nmonte):
        answer.append(q_values_mc_list_2[i][j])
    u = np.sum(answer,axis=0)
    o = u / Nmonte
    q_values_mc_list_3.append(o)    

# print(q_values_mc_list_3)

max_values_list = []
for i in range (number_of_iterations):
    pp = np.max(q_values_mc_list_3[i])
    max_values_list.append(pp)
    
# print(max_values_list)

plt.figure(1)
x_axis = np.arange(number_of_iterations)
plt.plot(x_axis,max_values_list,'r')
plt.xlabel('Number Of Iterations')
plt.ylabel('Max Q Values')
plt.title('Convergence Figure - Q learning')
plt.show()
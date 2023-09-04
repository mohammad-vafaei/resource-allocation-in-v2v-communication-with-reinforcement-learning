import numpy
import math

class MainTools():
    
    def __init__(self,N):
        self.N = N
    
    def large_scale_fading(self,std_Shadowing,distance,veh_antenna_height_tx,veh_antenna_height_rx,fc,veh_antenna_gain,veh_NoiseFigure):

        d_bp = 4 * (veh_antenna_height_tx-1) * (veh_antenna_height_rx-1) * fc * 10 ** 9 / (3 * 10 ** 8)
        A = 22.7
        B = 41.0 
        C = 20
        combinedPL_matrix = numpy.zeros((self.N,self.N))

        for i in range (self.N):
            for j in range (self.N):
                if distance[i][j] <= 3:
                    PL = A * math.log10(3) + B + C * math.log10(fc/5)
                    combinedPL_matrix[i,j] = PL
                elif distance[i][j] <= d_bp:
                    PL = A * math.log10(distance[i][j]) + B + C * math.log10(fc/5)
                    combinedPL_matrix[i,j] = PL
                else:
                    PL = (40 * (math.log10(distance[i][j]))) + (9.45) - (17.3 * (math.log10((veh_antenna_height_tx-1) * (veh_antenna_height_rx-1)))) + (2.7 * (math.log10(fc/5)))
                    combinedPL_matrix[i,j] = PL

        combinedPL_dB = -(combinedPL_matrix + (numpy.random.normal(0,1) * std_Shadowing)) + (2*veh_antenna_gain) - (veh_NoiseFigure)
        combinedPL = 10 ** (combinedPL_dB/10)
        return combinedPL
    def small_scale_fading(self):

        initial_matrix = numpy.ones((self.N,self.N))
        small_fading_matrix = ((numpy.random.normal(0,1)*initial_matrix) + 1j * (numpy.random.normal(0,1)*initial_matrix))/numpy.sqrt(2)

        return small_fading_matrix
    def channel_power_gain(self,small_scale_fading,large_scale_fading):

        channel_power_gain_matrix = large_scale_fading * (abs(small_scale_fading) ** 2)

        return channel_power_gain_matrix
    def action_list(self,pi_min,pi_max,steps):
        a = (pi_max-pi_min) / (steps)
        action_array = [pi_min]
        for actions in range(steps):
            pi_min = round(pi_min+a,3)
            action_array.append(pi_min)
        return action_array 
    def epsilon_greedy(self,epsilon,pi_min,pi_max,steps,count,q_values,initial_s,new_s):

        pi_vector = []
        b_vector = []

        for i in range(self.N):
            if numpy.random.random() < epsilon:
                k = numpy.random.randint(0,9)
                pi = self.action_list(pi_min,pi_max,steps)[k]
                b_vector.append(k)
                pi_vector.append(pi)
            else:

                if count == 0:
                    argmax_val = numpy.argmax(q_values[i][0,:])
                elif count == 1:
                    argmax_val = numpy.argmax(q_values[i][initial_s[i],:])
                else:
                    argmax_val = numpy.argmax(q_values[i][new_s,:])

                b_vector.append(argmax_val)
                pi = self.action_list(pi_min,pi_max,steps)[argmax_val]
                pi_vector.append(pi) 
        return pi_vector,b_vector
    def sinr(self,pi_array_b_vector,channel_power_gain,sigma):

        sinr_vector = []
        denum = 0

        for i in range(self.N):
            gamma_numerator = (pi_array_b_vector[0][i] * channel_power_gain[i][i])

            for j in range(self.N):
                if j != i :
                    left_denum = (pi_array_b_vector[0][j] * channel_power_gain[j][i])
                    denum += left_denum   

            gamma_denumerator = denum + (sigma)
            gamma = (gamma_numerator) / (gamma_denumerator)
            sinr_vector.append(gamma)

        return sinr_vector
    def reward_function(self,w,gamma_val,threshold,pi_array_b_vector):

        reward_vector = []

        for i in range(self.N):

            if gamma_val[i] >= threshold:
                r = (w * (numpy.log(1 + gamma_val[i]))) / (pi_array_b_vector[0][i])
                reward_vector.append(r)
            else:
                r = 0
                reward_vector.append(r)

        return reward_vector
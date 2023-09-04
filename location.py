import numpy as np
import math

def location(R0,Rm,N):
    # Location Tx of v2v respect to Origin
    r_tx = R0 + (Rm - R0) * (np.random.random_sample(size = N))          # distance of Tx v2v from mbs
    print('distance of Tx V2V from mbs respect to Origin = ','\n',r_tx,'\n')
    theta_tx = 2 * (math.pi) * (np.random.random_sample(size = N))       # angle of Tx V2V
    print('angle of Tx V2V respect to Origin = ','\n',theta_tx,'\n') 

    x_tx_vector=[]
    y_tx_vector=[]

    for i in range (N):

        x_tx = r_tx[i] * math.cos(theta_tx[i])
        y_tx = r_tx[i] * math.sin(theta_tx[i])

        x_tx_vector.append(x_tx)
        y_tx_vector.append(y_tx)

    print('x_tx_vector = ','\n',x_tx_vector,'\n')
    print('y_tx_vector = ','\n',y_tx_vector,'\n')

    # Location of Rx of V2V respect to its Tx
    d_tx_rx = 10 + (50 - 10) * (np.random.random_sample(size = N))
    print('Location of Rx of V2V respect to its Tx = ','\n',d_tx_rx,'\n')
    theta_tx_rx = 2 * (math.pi) * (np.random.random_sample(size = N))
    print('angle of Rx of V2V respect to its Tx = ','\n',theta_tx_rx,'\n')

    # distance between ith transmitter and ith reciever
    distance = np.diag(d_tx_rx)                          # we add monte carlo variable later (distance(i,i,nm))
    print('distance between ith transmitter and ith reciever = ','\n',distance,'\n')

    # Location of Rx of V2V respect to Origin
    x_rx_vector=[]
    y_rx_vector=[]

    for i in range (N):

        x_rx = r_tx[i] * math.cos(theta_tx[i]) + d_tx_rx[i] * math.cos(theta_tx_rx[i])
        y_rx = r_tx[i] * math.sin(theta_tx[i]) + d_tx_rx[i] * math.sin(theta_tx_rx[i])

        x_rx_vector.append(x_rx)
        y_rx_vector.append(y_rx)

    print('x_rx_vector = ','\n',x_rx_vector,'\n')
    print('y_rx_vector = ','\n',y_rx_vector,'\n')

    r_rx_vector=[]
    theta_rx_vector=[]

    for i in range (N):

        r_rx = np.sqrt((x_rx_vector[i] ** 2) + (y_rx_vector[i] ** 2))
        theta_rx = math.atan2(y_rx_vector[i] , x_rx_vector[i])

        r_rx_vector.append(r_rx)
        theta_rx_vector.append(theta_rx)

    print('distance of Rx V2V from mbs respect to Origin = ','\n',r_rx_vector,'\n')
    print('angle of Rx V2V respect to Origin = ','\n',theta_rx_vector,'\n')

    # distance between ith transmitter and jth recievers
    Ri = distance
    for i in range(N):
        for j in range(N):
            if i != j:
                r_txi_rxj = np.sqrt((r_rx_vector[i] ** 2) + (r_tx[j] ** 2) - (2 * r_rx_vector[i] * r_tx[j] * math.cos(theta_rx_vector[i] - theta_tx[j])))
                Ri[i,j] = r_txi_rxj
    
    
    return Ri

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:18:10 2023

@author: marabrandsen
"""

#Lotte Gritter 1804790
#Mara Brandsen 7251637

#Total run time is 1112.8254775410023s = 18.5min

#%%
import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.optimize import curve_fit

#%%

#you can edit the parameters of the experiment here
n = 15                  #square lattice width
temp_max = 20           #the range of temperatures (K)
number_of_temps = 10   #number of temperatures tested
num_of_trials = 6       #number of trials per temperature
aantal = 10         #number of runs/(attempted flips) per trial

#%%
#it is never necessart to change these
J = 1                   #coupling constant
kB= 1                   #Boltzmann constant
np.random.seed(23)      #arbitrary seed (23)
Temperature = np.linspace(0.01, temp_max, number_of_temps)

#%%

#for each of the following lists, there are lists included for each temperature 
#in which the corresponding component described is included
#each value included is the mean of all different trials for that temperature
E_lijst = []                #all mean energies per temperature
m_lijst = []                #all mean magnetisations per temperature
all_clusters = []           #list of coordinates with the matching neighbours per temperature
all_clusters4 = []          #list of all different clusters per temperature
lengte_all = []             #mean number of clusters per temperature
lengte_up = []              #mean number of clusters per temperature (only spin up)
lengte_down = []            #mean number of clusters per temperature (only spin down)
lijst_all = []              #mean cluster size per temperature
lijst_up = []               #mean (up spin) cluster size per temperature
lijst_down = []             #mean (down spin) cluster size per temperature
sd_collect_E = []           #sd of lattice energy per temperature
sd_collect_m = []           #sd of magnetisation per temperature
sd_collect_lijst_all = []   #sd of cluster size per temperature
sd_collect_lijst_up = []    #sd of cluster size per temperature (spin up)
sd_collect_lijst_down = []  #sd of cluster size per temperature (spin down)
sd_collect_lengte_all = []  #sd of number of clusters per temperature
sd_collect_lengte_up = []   #sd of number of clusters per temperature (spin up)
sd_collect_lengte_down = [] #sd of number of clusters per temperature (spin down)
    
#%%
#All the functions are defined in this section


#This function calculates the average energy of a matrix - see theory 
def energy(matrix_in):
    E_waarde = []
    
    for j in range(n):
        for k in range(n):
          p1 = matrix_in[(j+1)%n][k]
          p2 = matrix_in[(j-1)%n][k]
          p3 = matrix_in[j][(k+1)%n]
          p4 = matrix_in[j][(k-1)%n]   
          
          E_waarde.append(-J*matrix_in[j][k]*(p1 + p2 + p3 + p4))   
       
    E = sum(E_waarde)/(2*n**2) 
    return E

#This function creates a random matrix of 1's and -1's
def matrix_func(n):
    matrix_func = np.ones((n,n))
    for i in range(n):
        for j in range(n):
            matrix_func[i][j] = np.random.choice([-1, 1]) 
        
    return matrix_func

#This is the flip function which attemps and then either accepts or rejects 
#the flip of a the spin of a particle
def flip(matrix_in, Tred):
    T = Tred * J / kB
    
    #first a random coordinate is selected
    x = np.random.randint(0, n)
    y = np.random.randint(0, n)
    
    #now the energy difference between the original and flipped state is calculated
    val = matrix_in[x][y]
    x1, x2 = (x+1) % n, (x-1) % n
    y1, y2 = (y+1) % n, (y-1) % n
    
    p1 = matrix_in[x1][y]
    p2 = matrix_in[x2][y]
    p3 = matrix_in[x][y1]
    p4 = matrix_in[x][y2]
    sum = p1 + p2 + p3 + p4
    
    dif_E = (-J *-val*sum) - (-J *val*sum )
    
    #We use a random number R and compare it to the boltzmann probability 
    #Combine this with the theory we  know wether to accept or reject
    R = np.random.random()
    p = np.exp(-dif_E / (kB * T))

    matrix_in[x][y] *= -1
    if dif_E > 0 and R >= p:
        matrix_in[x][y] *= -1
    
    return matrix_in

#This function repeats the above determined flip function for the amount of runs
#we want to do, named "aantal" which is the dutch word for number/count
def flip_T(matrix_in, Tred):        
    for i in range(aantal):
        matrix_in = flip(matrix_in, Tred)
    return matrix_in


#We want to find the clusters: the following functions help eventually create
#a list per temperature, per trial which inckudes a list per cluster which then 
#includes all the coordinates of the particles in the cluster without duplicates

#First we check for each coordinate which of its direct neighbours match 
def single_cluster(j, k, matrix_in):    
    cluster_number = [[j,k]]

    val = matrix_in[j][k]
    j1, j2 = (j+1) % n, (j-1) % n
    k1, k2 = (k+1) % n, (k-1) % n

    if val == matrix_in[j1][k]: 
        cluster_number.append([j1, k])

    if val == matrix_in[j2][k]: 
        cluster_number.append([j2, k])     

    if val == matrix_in[j][k1]: 
        cluster_number.append([j, k1])     

    if val == matrix_in[j][k2]: 
         cluster_number.append([j, k2])  
    
    return cluster_number

#The single_cluster is repeated for the entire lattice, to get a collection
#of all coordinates of the lattice and each direct matching neighbours
def total_cluster(n, matrix_in, T, j):
    cluster = []
    
    for r in range(n):
        for p in range(n):
            cluster.append(single_cluster(r, p, matrix_in)) 
    
    print(T, j)            
    return cluster

#Now we want to collect the lists in total_cluster with matching coordinates
#to create one list per cluster, also removing duplicates while doing so  
def collect_clusters_temp2(clusters_all_temp_trial):
    merged_clusters = []

    for cluster in clusters_all_temp_trial:
        merged = False

        for merged_cluster in merged_clusters:
            if any(item in merged_cluster for item in cluster):
                merged_cluster.extend(item for item in cluster if item not in merged_cluster)
                merged = True
                break

        if not merged:
            merged_clusters.append(cluster)
            
    return merged_clusters


#This function creates a plot of a dataset 
def plot(x, y, xaxis_label, yaxis_label, title, errorbars = None, ylabel=None, yy=None, yylabel = None):
    plt.figure()
    plt.plot(x, y, "c.", label = ylabel) 
    
    plt.errorbar(x, y, yerr = errorbars[0], fmt='c.', elinewidth = 0.5)
   
    #This is in case two different data sets have to be plotted on one graph
    #It checks if there is a second data set and plots it
    if yy != None: 
        plt.plot(x, yy, "k.", label = yylabel) 
        plt.errorbar(x, yy, yerr = errorbars[1], fmt='m.',  elinewidth = 0.5)
        plt.legend()
    
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label) 
    plt.savefig(title+'3')
    return plt.show()

#To calculate the r^2 value of a given fit
def r_sq(x, y):
    residuals = y- func(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

#%%

#In order to time the program:
start_all = timeit.default_timer()

#First for each temperature and then each trial, the energy, magnetisation
#and the list with all clusters is collected
for i in range(len(Temperature)):
    #These list are created per Temperature so that we can find the averages
    #of the necessary values per temperature 
    verzamel_e = []
    verzamel_m = []
    all_clusters_temp = []
    
    Tred = Temperature[i]*kB/J
    
    #We find the energy, magnetisation at the end of all (attemped) flips
    #as well as the list with all the clusters
    for j in range(num_of_trials):        
        matrix_made = matrix_func(n)    
        matrix_final = flip_T(matrix_made, Tred)
        
        verzamel_e.append(energy(matrix_final))
        verzamel_m.append(np.mean(matrix_final))
        
        all_clusters_temp.append(total_cluster(n, matrix_final, Tred, j))
        
        #This shows the final matrix of each trial 
        #plt.figure()
        #plt.imshow(matrix_final, cmap = 'winter')

    #We collect all the averages of each temperature in the following lists
    m_lijst.append((np.mean(verzamel_m)))    
    E_lijst.append(np.mean(verzamel_e))
    
    #Here we will get a list which includes a list for each temp
    #which then includes a list for each trial which then has lists for all the coordinates
    all_clusters.append(all_clusters_temp)

    #To find the errors of each data point weuse np.std on the list which includes
    #all the different values per trial for one temperature
    sd_collect_E.append(np.std(verzamel_e))
    sd_collect_m.append(np.std(verzamel_m))


#To find the size and the number of clusters
for temps in range(len(all_clusters)):
    all_clusters3 = []
    lengte = []
    lengte_up_extra = []
    lengte_down_extra = []
    lengte_upp = 0
    lengte_downn = 0
    lijst = []
    lijst_upp = []
    lijst_downn = []
    
    
    for trials in range(len(all_clusters[temps])):
        #Here all the coordinates are made into cluster lists 
        all_clusters2 = collect_clusters_temp2(all_clusters[temps][trials])
        all_clusters3.append(all_clusters2)
        
        #To find the number of clusters regardless of their spin
        lengte.append(len(all_clusters2))
        
        #To find the number or size based on their spin value
        for elem in all_clusters2:
            #To pick the first element in the cluster to be used to determine the 
            #spin of the entire cluster
            aa = elem[0][0] 
            bb = elem[0][1]
            if matrix_final[aa][bb] == 1:
                lijst_upp.append(len(elem)) #To add the size of the cluster 
                lengte_upp = lengte_upp + 1 #To show there is another cluster
            if matrix_final[aa][bb] == -1:
                lijst_downn.append(len(elem)) #To add the size of the cluster 
                lengte_downn = lengte_downn + 1 #To show there is another cluster
            lijst.append(len(elem)) #To add the size of the cluster regardless of spin
        lengte_up_extra.append(lengte_upp) #To collect all number of clusters values to be able to determine the error
        lengte_down_extra.append(lengte_downn) #To collect all cluster size values to be able to determine the error
        

    #now the averages of each data set are calculated to find a value per temperature
    lijst_all.append(np.mean(lijst)) #
    lijst_up.append(np.mean(lijst_upp))
    lijst_down.append(np.mean(lijst_downn))
    
    lengte_up.append(lengte_upp)
    lengte_down.append(lengte_downn)
    lengte_all.append(np.mean(lengte))
    
    
    #To keep a record which includes all the clusters: a list with a list for each temperature
    #which then includes a list for each trial which then includes lists for each cluster with coordinates
    all_clusters4.append(all_clusters3)
    
    #To calculate the errors 
    sd_collect_lijst_all.append(np.std(lijst))
    sd_collect_lijst_up.append(np.std(lijst_upp)) 
    sd_collect_lijst_down.append(np.std(lijst_downn)) 
    sd_collect_lengte_all.append(np.std(lengte))
    sd_collect_lengte_up.append(np.std(lengte_up_extra))
    sd_collect_lengte_down.append(np.std(lengte_down_extra)) 

#To create lists with the errors that will be plotted on the same figure
sd_collect_number = [sd_collect_lengte_up, sd_collect_lengte_down]
sd_collect_size = [sd_collect_lijst_up, sd_collect_lijst_down]

#%%

#To plot all the different data collected
plot(Temperature, E_lijst, 'Temperature [k]', 'Energy [J]', 'Temperature [K] vs Energy [J]', errorbars = sd_collect_E, ylabel=None, yy=None, yylabel = None)
plot(Temperature, m_lijst, 'Temperature [k]', 'Magnetisation', 'Temperature [K] vs Magnetisation', errorbars = sd_collect_m, ylabel=None, yy=None, yylabel = None)
plot(Temperature, lengte_up, 'Temperature [k]', 'Number of clusters', 'Temperature [K] vs Number of clusters', errorbars = sd_collect_number , ylabel='Magnetisation = +1', yy=lengte_down, yylabel = 'Magnetisation = -1')
plot(Temperature, lijst_up, 'Temperature [k]', 'Clusters size', 'Temperature [K] vs Clusters size',  ylabel='Magnetisation = +1', errorbars = sd_collect_size, yy=lijst_down, yylabel = 'Magnetisation = -1')

#%%

#For two of the data sets we want to also fit a function

def func(x, a, b, c, d):
    return a - b*np.exp(-c * x + d)

#Creating the fit
popt, _ = curve_fit(func, Temperature, lengte_all)
x_new = Temperature
a, b, c, d = popt
y_new = func(x_new, a, b, c, d)

#Calculating the r^2 value
r_squared_num = r_sq(Temperature, lengte_all)
print('r squared is equal to', r_squared_num)

#Plotting the data with the fit 
plt.figure()
plt.plot(Temperature, lengte_all, "c.", label = 'Results') 
plt.errorbar(Temperature, lengte_all, yerr = sd_collect_lengte_all, fmt = 'c.', elinewidth=1)
plt.plot(x_new, y_new, "k", label = 'Fit') 
plt.xlabel('Temperature [k]')
plt.ylabel('Number of clusters') 
plt.legend()
plt.savefig('Temperature [K] vs Number of clusters - fit3')
plt.show()

print('For a - b*np.exp(-c * x + d), the parameters are: ', a, b, c, d)

###

#Creating the fit
popt, _ = curve_fit(func, Temperature, lijst_all)
x_new = Temperature
a, b, c, d = popt
y_new = func(x_new, a, b, c, d)

#Calculating the r^2 value
r_squared_size = r_sq(Temperature, lijst_all)
print('r squared is equal to ', r_squared_size)

#Plotting the data with the fit 
plt.figure()
plt.plot(Temperature, lijst_all, "c.", label = 'Results') 
plt.errorbar(Temperature, lijst_all, yerr = sd_collect_lijst_all, fmt='c.', elinewidth=1)
plt.plot(x_new, y_new, "k", label = 'Fit') 
plt.xlabel('Temperature [k]')
plt.ylabel('Clusters size') 
plt.legend()
plt.savefig('Temperature [K] vs Clusters size - fit3')
plt.show()

print('For a - b*np.exp(-c * x + d), the parameters are: ', a, b, c, d)


#%%

#To calculate and print the total run time
print('Total run time is ', timeit.default_timer() - start_all, 's')


#!/usr/bin/env python
from operator import xor
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spy
import cv2
def arrayPrint(array):
    string = ''
    for i in range(0, len(array)):
        string += str(array[i]) + ' '
    print(string)


def arrayShift(array):
    tmp = array[0]
    for i in range(0, len(array) - 1):
        array[i] = array[i + 1]
    array[len(array) - 1] = tmp
    return array

def additiveScrambler(signal):
    arrayPrint(signal)
    r = len(signal)
    seed = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, r):
    	
        xorA = xor(seed[-1], seed[-2])
        arrayShift(seed)
        seed[0] = xorA

        signal[i] = xor(xorA, signal[i])
    return signal


def multiplicativeScrambler(signal):
    arrayPrint(signal)
    r = len(signal)
    seed = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]
    output = []
    for i in range(0, r):
        xorA = xor(seed[17], seed[22])
        arrayShift(seed)
        xorB = xor(xorA, signal[i])
        seed[0] = xorB
        output.append(xorB)
    return output

def multiplicativeDescrambler(signal):
    arrayPrint(signal)
    r = len(signal)
    seed = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]
    output = []
    for i in range(0, r):
        xorA = xor(seed[17], seed[22])
        arrayShift(seed)
        seed[0] = signal[i]
        output.append(xor(xorA, signal[i]))
    arrayPrint(output)
    return

def polynomialScrambler7(signal):  
    counter = 0
    r = len(signal)
    seedOrigin = [1, 0, 0, 1, 0, 1, 0]
    seed = seedOrigin
    for i in range(0, r):
        if i%200 == 0:
            seed = seedOrigin
        xorA = xor(seed[6], seed[5])
        arrayShift(seed)
        seed[0] = xorA
        if(signal[i] == 1):
            counter += 1
        else:
            counter = 0
        if(counter == 7):
            seed = seedOrigin
        signal[i] = xor(xorA, signal[i])
    return signal

def polynomialScrambler43(signal):
    r = len(signal)
    seed = [1,0,1,0,0,0,0,0,1,0,1,1,0,1,0,0,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,0,0,0,0,1,1,1,1,1,1,0,1]
    for i in range(0, r):
        xorA = xor(seed[42], signal[i])
        arrayShift(seed)
        seed[0] = xorA
        signal[i] = xorA
    return signal

def polynomialDescrambler43(signal):
    arrayPrint(signal)
    r = len(signal)
    seed = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 0, 1]
    output = []
    for i in range(0, r):
        xorA = xor(signal[i], seed[42])
        arrayShift(seed)
        seed[0] = signal[i]
        output.append(xorA)
    return output

def polynomialScrambler25(signal):
    r = len(signal)
    seed = [0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,0,1,0,0,1,0,1,1]
    for i in range(0, r):
        xorA = xor(seed[24], signal[i])
        arrayShift(seed)
        seed[0] = xorA
        signal[i] = xorA
    return signal

def polynomialDescrambler25(signal):
    r = len(signal)
    seed = [0,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,0,1,0,0,1,0,1,1]
    output = []
    for i in range(0, r):
        xorA = xor(signal[i], seed[24])
        arrayShift(seed)
        seed[0] = signal[i]
        output.append(xorA)
    return output

def polynomialScrambler17(signal):  
    counter = 0
    r = len(signal)
    seedOrigin = [1,0,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1]
    seed = seedOrigin
    for i in range(0, r):
        if i%200 == 0:
            seed = seedOrigin
        xorA = xor(seed[16], seed[9])
        arrayShift(seed)
        seed[0] = xorA
        if(signal[i] == 1):
            counter += 1
        else:
            counter = 0
        if(counter == 7):
            seed = seedOrigin
        signal[i] = xor(xorA, signal[i])
    return signal

def histogram(signal, xlabel, ylabel):
    _ = plt.hist(signal, 7, color = 'g')
    _ = plt.xlabel(xlabel)
    _ = plt.ylabel(ylabel)
    plt.show()

def crc32(signal):
	crc = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1]
	#crc = [1, 0, 1, 1]
	crc_len = len(crc)
	for i in range(0, crc_len-1):
		signal.append(0)
	i = 0
	while not isZero(signal[:len(signal)-crc_len-1]):
		if signal[i] == 0:
			i += 1
		else:
			for j in range(0, crc_len):
				if(i+j >= len(signal)):
					continue
				signal[i+j] = xor(crc[j], signal[i+j])
	return signal[-(crc_len-1):]

def decodecrc32(signal):
	crc = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1]
	#crc = [1, 0, 1, 1]
	crc_len = len(crc)
	i = 0
	while not isZero(signal[:len(signal)-crc_len-1]):
		if signal[i] == 0:
			i += 1
		else:
			for j in range(0, crc_len):
				if(i+j >= len(signal)):
					continue
				signal[i+j] = xor(crc[j], signal[i+j])
	return signal[-(crc_len-1):]
	
def isZero(signal):
	for i in range(0, len(signal)):
		if signal[i] == 1:
			return False
	return True

def histData(signal, number):
	tabCounter = []
	for i in range(0, 100):
	    tabCounter.append(0)
	
	for j in range(0, 100):
	    counter2 = 0
	    for i in range(0, 1000):
	        if signal[i] == 1:
	            counter2 += 1
	        else:
	            tabCounter[counter2] += 1
	            counter2 = 0

	for j in range(0, 100):
	    tabCounter[j] /= 100


	histData = []
	for i in range(1, 9):
	    histData.append(tabCounter[i])

	#arrayPrint(histData)
	histData2 = []
	cnt=1

	for j in range(0, len(histData)):
	    for i in range(0, int(histData[j])):
	        histData2.append(cnt)
	    cnt+=1
	return histData

def stringToArray(signal):
	array = []
	for i in range(0, len(signal)):
		array.append(int(signal[i]))
	return array

print(' ')
randSequence = []
for i in range(0, 1000):
	    randSequence.append(random.randint(0, 1))	    
randSequence7 = randSequence
randSequence43 = randSequence
randSequence743 = randSequence

xaxe = 'Dlugosc ciagu'
yaxe = 'Liczba wystapien'
data0 = [histData(polynomialScrambler7(randSequence7), 0), histData(polynomialScrambler43(randSequence43), 0), histData(polynomialScrambler7(polynomialScrambler43(randSequence743)), 0)]

randSequence7 = randSequence
randSequence43 = randSequence
randSequence743 = randSequence

data1 = [histData(polynomialScrambler7(randSequence7), 1), histData(polynomialScrambler43(randSequence43), 1), histData(polynomialScrambler7(polynomialScrambler43(randSequence743)), 1)]
titles = ['x^7+x^6+1, 0', 'x^43+1, 0', 'x^7 na x^43, 0', 'x^7+x^6+1, 1', 'x^43+1, 1', 'x^7 na x^43, 1']
'''
plt.grid()
_ = plt.hist(data0, 7, color = ['skyblue', 'lightgray', 'c'], edgecolor='white', linewidth=5)
plt.show()
plt.grid()
_ = plt.hist(data1, 7, color = ['skyblue', 'lightgray', 'c'], edgecolor='white', linewidth=5)
plt.show()
'''


print(' ')
print(' ')

rS = randSequence
print('Test Chi-Kwadrat dla scramblera X7')
scrambledArray=polynomialScrambler7(rS)
mean = np.mean(scrambledArray)
print('Średnia: '+ str(mean))

sum0=0
for i in range(0, len(scrambledArray)):
	sum0+=(scrambledArray[i]-mean)*(scrambledArray[i]-mean)
sum0=sum0/len(scrambledArray)

rS = randSequence
rS1 = randSequence
print(spy.chisquare(polynomialScrambler7(rS), np.mean(polynomialScrambler7(rS1))))
print('Błąd średniokwadratowy dla scramblera X7')
print(sum0)
print(' ')



rS = randSequence
print('Test Chi-Kwadrat dla scramblera X43')
scrambledArray=polynomialScrambler43(rS)
mean = np.mean(scrambledArray)
print('Średnia: ' + str(mean))

sum1=0
for i in range(0, len(scrambledArray)):
	sum1+=(scrambledArray[i]-mean)*(scrambledArray[i]-mean)
sum1=sum1/len(scrambledArray)

rS = randSequence
rS1 = randSequence
print(spy.chisquare(polynomialScrambler43(rS), np.mean(polynomialScrambler43(rS1))))
print('Błąd średniokwadratowy dla scramblera X43')
print(sum1)
print(' ')




rS = randSequence
print('Test Chi-Kwadrat dla scramblera X7 nałożonego na X43')
scrambledArray = polynomialScrambler7(polynomialScrambler43(rS))
mean = np.mean(scrambledArray)
print('Średnia: ' + str(mean))

sum2=0
for i in range(0, len(scrambledArray)):
	sum2+=(scrambledArray[i]-mean)*(scrambledArray[i]-mean)
sum2=sum2/len(scrambledArray)

rS = randSequence
rS1 = randSequence
print(spy.chisquare(polynomialScrambler7(polynomialScrambler43(rS)), np.mean(polynomialScrambler7(polynomialScrambler43(rS1)))))
print('Błąd średniokwadratowy dla scramblera X7 nałożonego na X43')
print(sum2)
print(crc32([1,1,0,1,0,0,1,1,1,0,1,1,1,0]))

tab = [1, 0, 1, 0, 1, 1, 1]
print(polynomialScrambler17(tab))
print(polynomialScrambler17(tab))
#image to hist

img = cv2.imread('lolek.jpg', 0)
#img = cv2.resize(img, (0, 0), None, .5, .5)
#plt.hist(img.ravel(), 256, [0,256], color=['black'])
#plt.show()
y = len(img)
x = len(img[0])
img_rav = img.ravel()
signal = ""
for i in range(len(img_rav)):
	signal = signal + "{0:08b}".format(img_rav[i])

signal = polynomialScrambler7(polynomialScrambler43(stringToArray(signal)))

for i in range(int(len(signal)/8)):
	bin = ""
	for j in range(8):
		bin = bin + str(signal[8*i+j])
	img_rav[i] = bin
print(img_rav)

plt.hist(img_rav, 256, [0,256], color=['black'])
plt.show()

for i in range(y):
	for j in range(x):
		img[i][j] = img_rav[i*j+j]
cv2.imshow('po scramblowaniu', img)
cv2.waitKey(0)

def binToDec(x):
	return int(x, 2)

def addArray(signal):
	for i in range(tab_len):
		tablica[i] = binToDec(tablica[i])
	tmp = []
	for i in range(tab_len):
		tmp.append(tablica[i])
	return tmp

sig = []
for i in range(0, 128):
	g = random.randint(0, 1)
	sig.append(g)
print(sig)

def getSynch(x):
	synch = [1,1,1,1,1,1,1,1]
	return synch[x]

def createSignal(signal):
	createdSignal = []
	x=int(len(sig)/8)
	for i in range(8):
		createdSignal.append([])
		for j in range(8):
			createdSignal[i].append(getSynch(j))
		for k in range(x):
			createdSignal[i].append(signal[i*x+k])
		createdSignal[i] = createdSignal[i] + crc32(signal[x*i:(x*(i+1))])


	return createdSignal


tablica = ["00001111","00001000"]
tab_len = len(tablica)

tmp =  addArray(tablica)

temp2 = np.array(createSignal(sig))

for i in range(8):	
	print(temp2[i])


b = temp2.ravel()	


print(b)

'''
img = cv2.imread("lolek.jpg", 0)
cv2.imshow("image",img)
cv2.waitKey()
plt.hist(img.ravel(),256,[0,256])
plt.show()

plt.hist(tmp)
plt.show()
'''
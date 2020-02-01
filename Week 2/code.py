'''
-------------------------------------
 Assignment 2 - EE2703 (Jan-May 2020)
 Done by Akilesh Kannan (EE18B122)
 Created on 18/01/20
 Last Modified on 27/01/20
-------------------------------------
'''

# importing necessary libraries
import sys
import os
import math
import cmath
import numpy as np
import pandas as pd

# To improve readability
CIRCUIT_START = ".circuit"
CIRCUIT_END = ".end"
RESISTOR = "R"
CAPACITOR = "C"
INDUCTOR = "L"
IVS = "V"
ICS = "I"
VCVS = "E"
VCCS = "G"
CCVS = "H"
CCCS = "F"
PI = np.pi

# Classes for each circuit component
class resistor:
    def __init__(self, name, n1, n2, val):
        self.name = name
        self.value = enggToMath(val)
        self.node1 = n1
        self.node2 = n2

class inductor:
    def __init__(self, name, n1, n2, val):
        self.name = name
        self.value = enggToMath(val)
        self.node1 = n1
        self.node2 = n2

class capacitor:
    def __init__(self, name, n1, n2, val):
        self.name = name
        self.value = enggToMath(val)
        self.node1 = n1
        self.node2 = n2

class voltageSource:
    def __init__(self, name, n1, n2, val, phase=0):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.phase = float(phase)

class currentSource:
    def __init__(self, name, n1, n2, val, phase=0):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.phase = float(phase)

class vcvs:
    def __init__(self, name, n1, n2, n3, n4, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.node3 = n3
        self.node4 = n4

class vccs:
    def __init__(self, name, n1, n2, n3, n4, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.node3 = n3
        self.node4 = n4

class ccvs:
    def __init__(self, name, n1, n2, vName, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.vSource = vName

class cccs:
    def __init__(self, name, n1, n2, vName, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.vSource = vName

# Convert a number in engineer's format to math
def enggToMath(enggNumber):
    lenEnggNumber = len(enggNumber)
    if enggNumber[lenEnggNumber-1] == 'k':
        base = int(enggNumber[0:lenEnggNumber-1])
        return base*1e3
    elif enggNumber[lenEnggNumber-1] == 'm':
        base = int(enggNumber[0:lenEnggNumber-1])
        return base*1e-3
    elif enggNumber[lenEnggNumber-1] == 'u':
        base = int(enggNumber[0:lenEnggNumber-1])
        return base*1e-6
    elif enggNumber[lenEnggNumber-1] == 'n':
        base = int(enggNumber[0:lenEnggNumber-1])
        return base*1e-9
    elif enggNumber[lenEnggNumber-1] == 'M':
        base = int(enggNumber[0:lenEnggNumber-1])
        return base*1e6
    else:
        try:
            return float(enggNumber)
        except:
            sys.exit("Please check the component values given. Supported engineer units are: M, k, m, u, n\nYou can also enter values in exponential format (eg. 1e3 = 1000).")

if __name__ == "__main__":
    # checking number of command line arguments
    if len(sys.argv)!=2 :
        sys.exit("Invalid number of arguments!")
    else:
        try:
            circuitFile = sys.argv[1]
            circuitFreq = 1e-100
            circuitComponents = { RESISTOR: [], CAPACITOR: [], INDUCTOR: [], IVS: [], ICS: [], VCVS: [], VCCS: [], CCVS: [], CCCS: [] }
            circuitNodes = []
            # checking if given netlist file is of correct type
            if (not circuitFile.endswith(".netlist")):
                print("Wrong file type!")
            else:
                netlistFileLines = []
                with open (circuitFile, "r") as f:
                    for line in f.readlines():
                        netlistFileLines.append(line.split('#')[0].split('\n')[0])
                        # Getting frequency, if any
                        if(line[:3] == '.ac'):
                            circuitFreq = float(line.split()[2])
                try:
                    # Finding the location of the identifiers
                    identifier1 = netlistFileLines.index(CIRCUIT_START)
                    identifier2 = netlistFileLines.index(CIRCUIT_END)
                    circuitBody = netlistFileLines[identifier1+1:identifier2]
                    for line in circuitBody:
                        lineTokens = line.split()
                        try:
                            if lineTokens[1] not in circuitNodes:
                                circuitNodes.append(lineTokens[1])
                            if lineTokens[2] not in circuitNodes:
                                circuitNodes.append(lineTokens[2])
                        except IndexError:
                            continue
                        # Resistor
                        if lineTokens[0][0] == RESISTOR:
                            circuitComponents[RESISTOR].append(resistor(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[3]))
                        # Capacitor
                        elif lineTokens[0][0] == CAPACITOR:
                            circuitComponents[CAPACITOR].append(capacitor(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[3]))
                        # Inductor
                        elif lineTokens[0][0] == INDUCTOR:
                            circuitComponents[INDUCTOR].append(inductor(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[3]))
                        # Voltage Source
                        elif lineTokens[0][0] == IVS:
                            if len(lineTokens) == 5: # DC Source
                                circuitComponents[IVS].append(voltageSource(lineTokens[0], lineTokens[1], lineTokens[2], float(lineTokens[4])))
                            elif len(lineTokens) == 6: # AC Source
                                if circuitFreq == 1e-100:
                                    sys.exit("Frequency of AC Source not specified!!")
                                circuitComponents[IVS].append(voltageSource(lineTokens[0], lineTokens[1], lineTokens[2], float(lineTokens[4])/2, lineTokens[5]))
                        # Current Source
                        elif lineTokens[0][0] == ICS:
                            if len(lineTokens) == 5: # DC Source
                                circuitComponents[ICS].append(currentSource(lineTokens[0], lineTokens[1], lineTokens[2], float(lineTokens[4])))
                            elif len(lineTokens) == 6: # AC Source
                                if circuitFreq == 1e-100:
                                    sys.exit("Frequency of AC Source not specified!!")
                                circuitComponents[ICS].append(currentSource(lineTokens[0], lineTokens[1], lineTokens[2], float(lineTokens[4])/2, lineTokens[5]))
                        # VCVS
                        elif lineTokens[0][0] == VCVS:
                            circuitComponents[VCVS].append(vcvs(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[3], lineTokens[4], lineTokens[5]))
                        # VCCS
                        elif lineTokens[0][0] == VCCS:
                            circuitComponents[VCCS].append(vcvs(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[3], lineTokens[4], lineTokens[5]))
                        # CCVS
                        elif lineTokens[0][0] == CCVS:
                            circuitComponents[CCVS].append(ccvs(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[3], lineTokens[4]))
                        # CCCS
                        elif lineTokens[0][0] == CCCS:
                            circuitComponents[CCCS].append(cccs(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[3], lineTokens[4]))
                        # Erroneous Component Name
                        else:
                            sys.exit("Wrong Component Given. ABORT!")
                    try:
                        circuitNodes.remove('GND')
                        circuitNodes = ['GND'] + circuitNodes
                    except:
                        sys.exit("No ground node specified in the circuit!!")
                    nodeNumbers = {circuitNodes[i]:i for i in range(len(circuitNodes))}
                    numNodes = len(circuitNodes)
                    numVS = len(circuitComponents[IVS])+len(circuitComponents[VCVS])+len(circuitComponents[CCVS])
                    matrixM = np.zeros((numNodes+numVS, numNodes+numVS), np.complex)
                    matrixB = np.zeros((numNodes+numVS,), np.complex)
                    # GND Equation
                    matrixM[0][0] = 1.0
                    # Resistor Equations
                    for r in circuitComponents[RESISTOR]:
                        if r.node1 != 'GND':
                            matrixM[nodeNumbers[r.node1]][nodeNumbers[r.node1]] += 1/r.value
                            matrixM[nodeNumbers[r.node1]][nodeNumbers[r.node2]] -= 1/r.value
                        if r.node2 != 'GND':
                            matrixM[nodeNumbers[r.node2]][nodeNumbers[r.node1]] -= 1/r.value
                            matrixM[nodeNumbers[r.node2]][nodeNumbers[r.node2]] += 1/r.value
                    # Capacitor Equations
                    for c in circuitComponents[CAPACITOR]:
                        if c.node1 != 'GND':
                            matrixM[nodeNumbers[c.node1]][nodeNumbers[c.node1]] += complex(0, 2*PI*circuitFreq*c.value)
                            matrixM[nodeNumbers[c.node1]][nodeNumbers[c.node2]] -= complex(0, 2*PI*circuitFreq*c.value)
                        if c.node2 != 'GND':
                            matrixM[nodeNumbers[c.node2]][nodeNumbers[c.node1]] -= complex(0, 2*PI*circuitFreq*c.value)
                            matrixM[nodeNumbers[c.node2]][nodeNumbers[c.node2]] += complex(0, 2*PI*circuitFreq*c.value)
                    # Inductor Equations
                    for l in circuitComponents[INDUCTOR]:
                        if l.node1 != 'GND':
                            matrixM[nodeNumbers[l.node1]][nodeNumbers[l.node1]] += complex(0, -1.0/(2*PI*circuitFreq*l.value))
                            matrixM[nodeNumbers[l.node1]][nodeNumbers[l.node2]] -= complex(0, -1.0/(2*PI*circuitFreq*l.value))
                        if l.node2 != 'GND':
                            matrixM[nodeNumbers[l.node2]][nodeNumbers[l.node1]] -= complex(0, -1.0/(2*PI*circuitFreq*l.value))
                            matrixM[nodeNumbers[l.node2]][nodeNumbers[l.node2]] += complex(0, -1.0/(2*PI*circuitFreq*l.value))
                    # Voltage Source Equations
                    for i in range(len(circuitComponents[IVS])):
                        if circuitComponents[IVS][i].node1 != 'GND':
                            matrixM[nodeNumbers[circuitComponents[IVS][i].node1]][numNodes+i] = 1.0
                        if circuitComponents[IVS][i].node2 != 'GND':
                            matrixM[nodeNumbers[circuitComponents[IVS][i].node2]][numNodes+i] = -1.0
                        matrixM[numNodes+i][nodeNumbers[circuitComponents[IVS][i].node1]] = -1.0
                        matrixM[numNodes+i][nodeNumbers[circuitComponents[IVS][i].node2]] = +1.0
                        matrixB[numNodes+i] = cmath.rect(circuitComponents[IVS][i].value, circuitComponents[IVS][i].phase*PI/180)
                    # Current Source Equations
                    for i in circuitComponents[ICS]:
                        if i.node1 != 'GND':
                            matrixB[nodeNumbers[i.node1]] = -1*i.value
                        if i.node2 != 'GND':
                            matrixB[nodeNumbers[i.node2]] = i.value
                    # VCVS Equations
                    for i in range(len(circuitComponents[VCVS])):
                        if circuitComponents[VCVS][i].node1 != 'GND':
                            matrixM[nodeNumbers[circuitComponents[VCVS][i].node1]][numNodes+len(circuitComponents[IVS])+i] = 1.0
                        if circuitComponents[VCVS][i].node2 != 'GND':
                            matrixM[nodeNumbers[circuitComponents[VCVS][i].node2]][numNodes+len(circuitComponents[IVS])+i] = -1.0
                        matrixM[numNodes+len(circuitComponents[IVS])+i][nodeNumbers[circuitComponents[VCVS][i].node1]] = 1.0
                        matrixM[numNodes+len(circuitComponents[IVS])+i][nodeNumbers[circuitComponents[VCVS][i].node2]] = -1.0
                        matrixM[numNodes+len(circuitComponents[IVS])+i][nodeNumbers[circuitComponents[VCVS][i].node3]] = -1.0*circuitComponents[VCVS][i].value
                        matrixM[numNodes+len(circuitComponents[IVS])+i][nodeNumbers[circuitComponents[VCVS][i].node4]] = 1.0*circuitComponents[VCVS][i].value
                    # CCVS Equations
                    for i in range(len(circuitComponents[CCVS])):
                        if circuitComponents[VCVS][i].node1 != 'GND':
                            matrixM[nodeNumbers[circuitComponents[CCVS][i].node1]][numNodes+len(circuitComponents[IVS])+len(circuitComponents[VCVS])+i] = 1.0
                        if circuitComponents[VCVS][i].node2 != 'GND':
                            matrixM[nodeNumbers[circuitComponents[VCVS][i].node2]][numNodes+len(circuitComponents[IVS])+len(circuitComponents[VCVS])+i] = -1.0
                        matrixM[numNodes+len(circuitComponents[IVS])+len(circuitComponents[VCVS])+i][nodeNumbers[circuitComponents[CCVS][i].node1]] = 1.0
                        matrixM[numNodes+len(circuitComponents[IVS])+len(circuitComponents[VCVS])+i][nodeNumbers[circuitComponents[CCVS][i].node2]] = -1.0
                        matrixM[numNodes+len(circuitComponents[IVS])+len(circuitComponents[VCVS])+i][numNodes+len(circuitComponents[IVS])+len(circuitComponents[VCVS])+i] = -1.0*circuitComponents[CCVS][i].value
                    # VCCS Equations
                    for vccs in circuitComponents[VCCS]:
                        matrixM[nodeNumbers[vccs.node1]][nodeNumbers[vccs.node4]]+=vccs.value
                        matrixM[nodeNumbers[vccs.node1]][nodeNumbers[vccs.node3]]-=vccs.value
                        matrixM[nodeNumbers[vccs.node2]][nodeNumbers[vccs.node4]]-=vccs.value
                        matrixM[nodeNumbers[vccs.node3]][nodeNumbers[vccs.node3]]+=vccs.value
                    # CCCS Equations
                    for cccs in circuitComponents[CCCS]:
                        matrixM[nodeNumbers[cccs.node1]][numNodes+circuitComponents[IVS].index(cccs.vSource)]-=cccs.value
                        matrixM[nodeNumbers[cccs.node2]][numNodes+circuitComponents[IVS].index(cccs.vSource)]+=cccs.value
                    try:
                        x = np.linalg.solve(matrixM, matrixB)
                        circuitCurrents = []
                        # Data for printing clear data as output
                        for v in circuitComponents[IVS]:
                            circuitCurrents.append("current in "+v.name)
                        for v in circuitComponents[VCVS]:
                            circuitCurrents.append("current in "+v.name)
                        for v in circuitComponents[CCVS]:
                            circuitCurrents.append("current in "+v.name)
                        # Printing data output in table format
                        print(pd.DataFrame(x, circuitNodes+circuitCurrents, columns=['Voltage / Current']))
                    except np.linalg.LinAlgError:
                        sys.exit("Singular Matrix Formed! Please check if you have entered the circuit definition correctly!")
                except ValueError:
                    sys.exit("Netlist does not abide to given format!")
        except FileNotFoundError:
            sys.exit("Given file does not exist!")

'''
-------------------------------------
 Assignment 2 - EE2703 (Jan-May 2020)
 Done by Akilesh Kannan (EE18B122)
 Created on 18/01/20
 Last Modified on 23/01/20
-------------------------------------
'''

# importing necessary libraries
import sys
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
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
    def printComponent(self):
        print("Name: {0}\nValue: {1}\nNode 1: {2}\nNode 2: {3}\n\n".format(self.name, self.value, self.node1, self.node2))

class inductor:
    def __init__(self, name, n1, n2, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
    def printComponent(self):
        print("Name: {0}\nValue: {1}\nNode 1: {2}\nNode 2: {3}\n\n".format(self.name, self.value, self.node1, self.node2))

class capacitor:
    def __init__(self, name, n1, n2, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
    def printComponent(self):
        print("Name: {0}\nValue: {1}\nNode 1: {2}\nNode 2: {3}\n\n".format(self.name, self.value, self.node1, self.node2))

class voltageSource:
    def __init__(self, name, n1, n2, val, phase=0):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.phase = float(phase)
    def printComponent(self):
        print("Name: {0}\nValue: {1}\nNode 1: {2}\nNode 2: {3}\nPhase: {4}\n\n".format(self.name, self.value, self.node1, self.node2, self.phase))

class currentSource:
    def __init__(self, name, n1, n2, val, phase=0):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.phase = float(phase)
    def printComponent(self):
        print("Name: {0}\nValue: {1}\nNode 1: {2}\nNode 2: {3}\nPhase: {4}\n\n".format(self.name, self.value, self.node1, self.node2, self.phase))

class vcvs:
    def __init__(self, name, n1, n2, n3, n4, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.node3 = n3
        self.node4 = n4
    def printComponent(self):
        print("Name: {0}\nValue: {1}\nNode 1: {2}\nNode 2: {3}\nVoltage Source Node 1: {4}\nVoltage Source Node 2: {5}\n\n".format(self.name, self.value, self.node1, self.node2, self.node3, self.node4))

class vccs:
    def __init__(self, name, n1, n2, n3, n4, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.node3 = n3
        self.node4 = n4
    def printComponent(self):
        print("Name: {0}\nValue: {1}\nNode 1: {2}\nNode 2: {3}\nVoltage Source Node 1: {4}\nVoltage Source Node 2: {5}\n\n".format(self.name, self.value, self.node1, self.node2, self.node3, self.node4))

class ccvs:
    def __init__(self, name, n1, n2, vName, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.vSource = vName
    def printComponent(self):
        print("Name: {0}\nValue: {1}\nNode 1: {2}\nNode 2: {3}\nVoltage Source: {4}\n\n".format(self.name, self.value, self.node1, self.node2, self.vSource))

class cccs:
    def __init__(self, name, n1, n2, vName, val):
        self.name = name
        self.value = float(val)
        self.node1 = n1
        self.node2 = n2
        self.vSource = vName
    def printComponent(self):
        print("Name: {0}\nValue: {1}\nNode 1: {2}\nNode 2: {3}\nVoltage Source: {4}\n\n".format(self.name, self.value, self.node1, self.node2, self.vSource))

# Convert a number in engineer's format to math
def enggToMath(enggNumber):
    lenEnggNumber = len(enggNumber)
    if enggNumber[lenEnggNumber-1] == 'k':
        base = int(enggNumber[0:lenEnggNumber-1])
        return base*1000
    elif enggNumber[lenEnggNumber-1] == 'm':
        base = int(enggNumber[0:lenEnggNumber-1])
        return base*0.001
    else:
        return float(enggNumber)

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
                                circuitComponents[IVS].append(voltageSource(lineTokens[0], lineTokens[1], lineTokens[2], float(lineTokens[4])/(2*math.sqrt(2)), lineTokens[5]))

                        # Current Source
                        elif lineTokens[0][0] == ICS:
                            if len(lineTokens) == 5: # DC Source
                                circuitComponents[ICS].append(currentSource(lineTokens[0], lineTokens[1], lineTokens[2], float(lineTokens[4])))
                            elif len(lineTokens) == 6: # AC Source
                                circuitComponents[ICS].append(currentSource(lineTokens[0], lineTokens[1], lineTokens[2], float(lineTokens[4])/(2*math.sqrt(2)), lineTokens[5]))

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

                    # # Printing nodes in the circuit
                    # print("\nThe nodes in the circuit are: ", end='')
                    # for x in circuitNodes:
                    #     print(x, end=' ')
                    # # Printing the circuit frequency
                    # print("\n\nThe circuit frequency is: "+str(circuitFreq))
                    #
                    # # Printing all circuit components
                    # print("\nThe circuit components are:\n")
                    # for x in circuitComponents:
                    #     for y in circuitComponents[x]:
                    #         y.printComponent()

                    try:
                        circuitNodes.remove('GND')
                        circuitNodes = ['GND'] + circuitNodes
                    except:
                        sys.exit("No ground node specified in the circuit!!")

                    nodeNumbers = {circuitNodes[i]:i for i in range(len(circuitNodes))}

                    matrixM = np.zeros((len(circuitNodes)+len(circuitComponents[IVS]), len(circuitNodes)+len(circuitComponents[IVS])), np.complex)
                    matrixB = np.zeros((len(circuitNodes)+len(circuitComponents[IVS]),), np.complex)

                    # GND Equation
                    matrixM[0][0] = 1.0

                    # Equations for source voltages
                    for i in range(len(circuitComponents[IVS])):
                        source = circuitComponents[IVS][i]
                        l = len(circuitNodes)
                        matrixM[l+i][nodeNumbers[source.node1]] = -1.0
                        matrixM[l+i][nodeNumbers[source.node2]] = 1.0
                        matrixB[l+i] = cmath.rect(source.value, source.phase*PI/180)

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

                    numNodes = len(circuitNodes)

                    # Voltage Source Equations
                    for vol in circuitComponents[IVS]:
                        if(vol.node1 != 'GND'):
                            matrixM[nodeNumbers[vol.node1]][numNodes] = -1.0
                        if(vol.node2 != 'GND'):
                            matrixM[nodeNumbers[vol.node2]][numNodes] = 1.0

                    # Current source equations
                    for source in circuitComponents[ICS]:
                        if(source.node1 != 'GND'):
                            matrixB[nodeNumbers[source.node1]] += cmath.rect(
                                source.value, source.phase*cmath.pi/180)
                        if(source.node2 != 'GND'):
                            matrixB[nodeNumbers[source.node2]] -= cmath.rect(
                                source.value, source.phase*cmath.pi/180)

                    # for i in range(len(circuitNodes)+len(circuitComponents[IVS])):
                    #     for j in matrixM[i]:
                    #         print(j, end=' ')
                    #     print('')

                    try:
                        x = np.linalg.solve(matrixM, matrixB)

                        circuitCurrents = []
                        # Data for printing clear data as output
                        for v in circuitComponents[IVS]:
                            circuitCurrents.append("I in "+v.name)

                        # Printing data output
                        print(pd.DataFrame(x, columns=['Value'], index=circuitNodes+circuitCurrents))
                        
                    except np.linalg.LinAlgError:
                        print("Singular Matrix Formed! Please check if you have entered the circuit definition correctly!")

                except ValueError:
                    sys.exit("Netlist does not abide to given format!")
        except FileNotFoundError:
            sys.exit("Given file does not exist!")

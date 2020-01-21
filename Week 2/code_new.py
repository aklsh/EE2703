'''
-------------------------------------
 Assignment 2 - EE2703 (Jan-May 2020)
 Done by Akilesh Kannan (EE18B122)
 Created on 18/01/20
 Last Modified on 20/01/20
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

# Print the Circuit Definition in the required format
def printCktDefn(tokenedSPICELines):
    for x in tokenedSPICELines:
        for y in x:
            print(y, end=' ')
        print('')
    print('')
    return

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
                SPICELines = []
                with open (circuitFile, "r") as f:
                    for line in f.readlines():
                        SPICELines.append(line.split('#')[0].split('\n')[0])
                try:
                    # Finding the location of the identifiers
                    identifier1 = SPICELines.index(CIRCUIT_START)
                    identifier2 = SPICELines.index(CIRCUIT_END)

                    circuitBody = SPICELines[identifier1+1:identifier2]
                    for line in circuitBody:
                        lineTokens = line.split()

                        if lineTokens[1] not in circuitNodes:
                            circuitNodes.append(lineTokens[1])
                        if lineTokens[2] not in circuitNodes:
                            circuitNodes.append(lineTokens[2])

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
                                circuitComponents[IVS].append(voltageSource(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[4]))
                            elif len(lineTokens) == 6: # AC Source
                                circuitComponents[IVS].append(voltageSource(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[4], lineTokens[5]))

                        # Current Source
                        elif lineTokens[0][0] == ICS:
                            if len(lineTokens) == 5: # DC Source
                                circuitComponents[ICS].append(currentSource(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[4]))
                            elif len(lineTokens) == 6: # AC Source
                                circuitComponents[ICS].append(currentSource(lineTokens[0], lineTokens[1], lineTokens[2], lineTokens[4], lineTokens[5]))

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

                    # Debugging Print Statements
                    for x in circuitComponents:
                        for y in circuitComponents[x]:
                            y.printComponent()

                except ValueError:
                    sys.exit("Netlist does not abide to given format!")
        except FileNotFoundError:
            sys.exit("Given file does not exist!")

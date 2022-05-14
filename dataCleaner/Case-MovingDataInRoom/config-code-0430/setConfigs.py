import xml.etree.ElementTree as ET
import os.path
import sys
import random

################################################################@
""" Local classes """
################################################################@

""" Family
    The Family class is used to handle any family in the config files
"""
class Family:
    def __init__(self,name):
        self.name=name
        self.scenario=[]

""" Scenario
    The Scenario class is used to handle the scenario and zones inside
"""
class Scenario:
    def __init__(self,name):
        self.name = name
        self.zones = []

""" Zone
    The Zone class is used to handle boundary parameters and its method
"""
class Zone:
    def __init__(self,name,place,priority,method,lx,ux,ly,uy):
        self.name = name
        self.place = place
        self.priority = priority
        self.outdoor = [0,0]
        self.ux = ux
        self.lx = lx
        self.uy = uy
        self.ly = ly
        self.method = method

    def setOutDoor(self,x,y):
        self.outdoor = [x,y]

""" Station
    The Station class is used to handle station info
"""
class Station:
    def __init__(self,name,x,y):
        self.name = name
        self.x = x
        self.y = y

################################################################@
""" Local methods """
################################################################@

""" parseFamily
    Method to parse the whole Family
        xmlFile : the path to the xml file
"""
def parseFamily(xmlFile):
    if os.path.isfile(xmlFile):
        tree = ET.parse(xmlFile)
        root = tree.getroot()
        for f in root.findall('family'):
            family = Family(f.get('name'))
            families.append(family)
            outdoor = f.find('outdoor')
            outx = float(outdoor.get('x'))
            outy = float(outdoor.get('y'))
            for sc in f.findall('scenario'):
                scenario = Scenario(sc.get('name'))
                family.scenario.append(scenario)
                for z in sc.findall('zone'):
                    zone = Zone(z.get('name'),z.get('place'),int(z.get('priority')),z.get('method'),
                            float(z.get('xlow')),float(z.get('xup')),float(z.get('ylow')),float(z.get('yup')))
                    zone.setOutDoor(outx,outy)
                    scenario.zones.append(zone)
    else:
        print("File not found: "+xmlFile)

""" traceFamily
    Method to trace the family to the console
"""
def traceFamily(families):
    #print("Stations:")
    #for node in nodes:
    #    if not node.isSwitch():
    #        print ("\t" + node.name)
            
    print("\nTrace familes:")
    for family in families:
        print ("\t" + family.name)
        for scenario in family.scenario:
            print ("\t\tScenario=" + scenario.name)
            for zone in scenario.zones:
                print ("\t\t\t" + zone.name+": "+zone.place + "\n\t\t\t\t{ x : [" + str(zone.lx) + ","+ str(zone.ux) + "]" +
                        ",y : ["+ str(zone.ly) + "," + str(zone.uy) + "]" + " => " +zone.method)

################################################################@
""" Global data """
################################################################@
families = [] # families

parseFamily('config.xml')
#traceFamily(families)

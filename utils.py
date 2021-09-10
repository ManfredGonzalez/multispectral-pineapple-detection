import bs4
from bs4 import BeautifulSoup
def getXMPData(filePath):
    fin = open( filePath, "rb")
    img = fin.read()
    imgAsString=str(img)
    xmp_start = imgAsString.find('<x:xmpmeta')
    xmp_end = imgAsString.find('</x:xmpmeta')
    if xmp_start != xmp_end:
        xmpString = imgAsString[xmp_start:xmp_end+12]
    
    xmpAsXML = BeautifulSoup(xmpString, "xml")
    info = str(xmpAsXML.Description).split('drone-dji:')
    info_dji = info[1:len(info)-1]
    keys = []
    values = []
    for line in info_dji:
        line = line.replace('"',"")
        line = line.replace('\\n',"")
        line = line.replace(' ',"")
        line = line.split('=')
        keys.append(line[0])
        values.append(line[1])
    dictionary = dict(zip(keys,values))
    return dictionary


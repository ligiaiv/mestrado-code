PATH = "/media/ligia/DA8CB0F18CB0C8F1/Users/ligia/Escola/MestradoBR/"
FILE = "MERGED.csv"



#Define Filters

#Language Filter

def lang_filter(cells,LANG):
    return cells[5] == LANG
def rt_filter(cells):
    start = cells[0]
    return not(start.startwith("RT ") or start.startwith('"RT '))

filters = [rt_filter,lang_filter]
def lang_filter(PATH,FILE,LANG):
    is_in = True
    for f in filters:
        is_in = is_in and f(cells)
    print("Filtering "+LANG+" tweets")
# N = 1000
# LANG = "en"
    inFile = open(PATH +FILE,'r')
    outFile = open(PATH+"filtered_"+LANG+"_"+FILE, 'w')

    i = 0
    for line in inFile:
        cells = line.split("|")
        
        if cells[5] == LANG:
            outFile.write(line)

        i+=1
        if (i % 1000) == 0:
            print(str(i),end="\r", flush=True)
    inFile.close()
    outFile.close()
lang_filter(PATH,FILE,"pt")
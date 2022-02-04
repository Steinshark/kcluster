file = open("docword.nytimes.txt",'r')
file_write = open("newData","w")

for line in file.readlines()[:100000]:
    file_write.write(line)
file_write.close()

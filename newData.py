file = open("docword.nytimes.txt",'r')
file_write = open("newData","w")


size  = int(input("num docs: "))
file_write.write(f"{size}\n")
file_write.write(f"{102660}\n")
file_write.write(f"{69679427}\n")

file.readline()
file.readline()
file.readline()

for line in file:
    doc, word, count = line.split()
    if int(doc) -1  == size:
        break

    file_write.write(f"{doc} {word} {count}\n")
file.close()
file_write.close()

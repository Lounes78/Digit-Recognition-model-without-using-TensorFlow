def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb") #Open in Binary Mode
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16) #Skip the metadata
    l.read(8)  #Skip the metadata
    images = []

    for i in range(n):
        image = [ord(l.read(1))] #Read the label
        for j in range(28*28):
            image.append(ord(f.read(1))) #Getting back the numerical value of each pixel (0 - 255)
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n") # Write the value of each pixel into the output file seperated with a comma (csv)
    f.close()
    o.close()
    l.close()

#Don't forget to replace with the correct file path

convert("Binary_data/train-images.idx3-ubyte", "Binary_data/train-labels.idx1-ubyte",
        "mnist_train.csv", 60000)
convert("Binary_data/t10k-images.idx3-ubyte", "Binary_data/t10k-labels.idx1-ubyte",
        "mnist_test.csv", 10000)
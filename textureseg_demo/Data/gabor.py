# ----------------------------------------------------------
# Adam Lefaivre (001145679)
# Cpsc 5990
# Final Program Project
# Dr. Howard Cheng
# ----------------------------------------------------------

import sklearn.cluster as clstr
from PIL import Image, ImageOps, ImageDraw
import os, glob
import matplotlib.pyplot as pyplt
import scipy.cluster.vq as vq
import cv2
import numpy as np
import math
import sys
sys.path.insert(1, './')
del sys
import argparse
import os.path
import matplotlib.pyplot as plt
# A simple convolution function that returns the filtered images.
def getFilterImages(filters, img):
    featureImages = []
    for filter in filters:
        kern, params = filter
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        featureImages.append(fimg)
    return featureImages

# Apply the R^2 threshold technique here, note we find energy in the spatial domain.
def filterSelection(featureImages, threshold, img, howManyFilterImages):

    idEnergyList = []
    id = 0
    height, width = img.shape
    for featureImage in featureImages:
        thisEnergy = 0.0
        for x in range(height):
            for y in range(width):
                thisEnergy += pow(np.abs(featureImage[x][y]), 2)
        idEnergyList.append((thisEnergy, id))
        id += 1
    E = 0.0
    for E_i in idEnergyList:
        E += E_i[0]
    sortedlist = sorted(idEnergyList, key=lambda energy: energy[0], reverse = True)

    tempSum = 0.0
    RSquared = 0.0
    added = 0
    outputFeatureImages = []
    while ((RSquared < threshold) and (added < howManyFilterImages)):
        tempSum += sortedlist[added][0]
        RSquared = (tempSum/E)
        outputFeatureImages.append(featureImages[sortedlist[added][1]])
        added += 1
    return outputFeatureImages

# This is where we create the gabor kernel
# Feel free to uncomment the other list of theta values for testing.
def build_filters(lambdas, ksize, gammaSigmaPsi):

    filters = []
    thetas = []

    # Thetas 1
    # -------------------------------------
    thetas.extend([0, 45, 90, 135])

    # Thetas2
    # -------------------------------------
    #thetas.extend([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])

    thetasInRadians = [np.deg2rad(x) for x in thetas]

    for lamb in lambdas:
        for theta in thetasInRadians:
            params = {'ksize': (ksize, ksize), 'sigma': gammaSigmaPsi[1], 'theta': theta, 'lambd': lamb,
                   'gamma':gammaSigmaPsi[0], 'psi': gammaSigmaPsi[2], 'ktype': cv2.CV_64F}
            kern = cv2.getGaborKernel(**params)
            kern /= 1.5 * kern.sum()
            filters.append((kern, params))
    return filters

# Here is where we convert radial frequencies to wavelengths.
# Feel free to uncomment the other list of lambda values for testing.
def getLambdaValues(img):
    height, width = img.shape

    #calculate radial frequencies.
    max = (width/4) * math.sqrt(2)
    min = 4 * math.sqrt(2)
    temp = min
    radialFrequencies = []

    # Lambda 1
    # -------------------------------------
    while(temp < max):
        radialFrequencies.append(temp)
        temp = temp * 2

    # Lambda 2
    # -------------------------------------
    # while(temp < max):
    #     radialFrequencies.append(temp)
    #     temp = temp * 1.5

    radialFrequencies.append(max)
    lambdaVals = []
    for freq in radialFrequencies:
        lambdaVals.append(width/freq)
    return lambdaVals

# The activation function with gaussian smoothing
def nonLinearTransducer(img, gaborImages, L, sigmaWeight, filters):

    alpha_ = 0.25
    featureImages = []
    count = 0
    for gaborImage in gaborImages:

        # Spatial method of removing the DC component
        avgPerRow = np.average(gaborImage, axis=0)
        avg = np.average(avgPerRow, axis=0)
        gaborImage = gaborImage.astype(float) - avg

        #gaborImage = cv2.normalize(gaborImage, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Normalization sets the input to the active range [-2,2] this becomes [-8,8] with alpha_
        if int(cv2.__version__[0]) >= 3:
            gaborImage = cv2.normalize(gaborImage, gaborImage, alpha=-8, beta=8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            gaborImage = cv2.normalize(gaborImage, alpha=-8, beta=8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        height, width = gaborImage.shape
        copy = np.zeros(img.shape)
        for row in range(height):
            for col in range(width):
                #centralPixelTangentCalculation_bruteForce(gaborImage, copy, row, col, alpha, L)
                copy[row][col] = math.fabs(math.tanh(alpha_ * (gaborImage[row][col])))

        # now apply smoothing
        copy, destroyImage = applyGaussian(copy, L, sigmaWeight, filters[count])
        if(not destroyImage):
            featureImages.append(copy)
        count += 1

    return featureImages

# I implemented this just for completeness
# It just applies the tanh function and smoothing as spatial convolution
def centralPixelTangentCalculation_bruteForce(img, copy, row, col, alpha, L):
    height, width = img.shape
    windowHeight, windowWidth, inita, initb = \
        getRanges_for_window_with_adjust(row, col, height, width, L)

    sum = 0.0
    for a in range(windowHeight + 1):
        for b in range(windowWidth + 1):
            truea = inita + a
            trueb = initb + b
            sum += math.fabs(math.tanh(alpha * (img[truea][trueb])))

    copy[row][col] = sum/pow(L, 2)

# Apply Gaussian with the central frequency specification
def applyGaussian(gaborImage, L, sigmaWeight, filter):

    height, N_c = gaborImage.shape

    nparr = np.array(filter[0])
    u_0 = nparr.mean(axis=0)
    u_0 = u_0.mean(axis=0)

    destroyImage = False
    sig = 1
    if (u_0 < 0.000001):
        print("div by zero occured for calculation:")
        print("sigma = sigma_weight * (N_c/u_0), sigma will be set to zero")
        print("removing potential feature image!")
        destroyImage = True
    else:
        sig = sigmaWeight * (N_c / u_0)

    return cv2.GaussianBlur(gaborImage, (L, L), sig), destroyImage

# Remove feature images with variance lower than 0.0001
def removeFeatureImagesWithSmallVariance(featureImages, threshold):
    toReturn =[]
    for image in featureImages:
        if(np.var(image) > threshold):
            toReturn.append(image)

    return toReturn

# This is the function that checks boundaries when performing spatial convolution.
def getRanges_for_window_with_adjust(row, col, height, width, W):

    mRange = []
    nRange = []

    mRange.append(0)
    mRange.append(W-1)

    nRange.append(0)
    nRange.append(W-1)

    initm = int(round(row - math.floor(W / 2)))
    initn = int(round(col - math.floor(W / 2)))

    if (initm < 0):
        mRange[1] += initm
        initm = 0

    if (initn < 0):
        nRange[1] += initn
        initn = 0

    if(initm + mRange[1] > (height - 1)):
        diff = ((initm + mRange[1]) - (height - 1))
        mRange[1] -= diff

    if(initn + nRange[1] > (width-1)):
        diff = ((initn + nRange[1]) - (width - 1))
        nRange[1] -= diff

    windowHeight = mRange[1] - mRange[0]
    windowWidth = nRange[1] - nRange[0]

    return int(round(windowHeight)), int(round(windowWidth)), int(round(initm)), int(round(initn))

# Used to normalize data before clustering occurs.
# Whiten sets the variance to be 1 (unit variance),
# spatial weighting also takes place here.
# The mean can be subtracted if specified by the implementation.
def normalizeData(featureVectors, setMeanToZero, spatialWeight=1):

    means = []
    for col in range(0, len(featureVectors[0])):
        colMean = 0
        for row in range(0, len(featureVectors)):
            colMean += featureVectors[row][col]
        colMean /= len(featureVectors)
        means.append(colMean)

    for col in range(2, len(featureVectors[0])):
        for row in range(0, len(featureVectors)):
            featureVectors[row][col] -= means[col]
    copy = vq.whiten(featureVectors)
    if (setMeanToZero):
        for row in range(0, len(featureVectors)):
            for col in range(0, len(featureVectors[0])):
                copy[row][col] -= means[col]

    for row in range(0, len(featureVectors)):
        copy[row][0] *= spatialWeight
        copy[row][1] *= spatialWeight

    return copy

# Create the feature vectors and add in row and column data
def constructFeatureVectors(featureImages, img):

    featureVectors = []
    height, width = img.shape
    for row in range(height):
        for col in range(width):
            featureVector = []
            featureVector.append(row)
            featureVector.append(col)
            for featureImage in featureImages:
                featureVector.append(featureImage[row][col])
            featureVectors.append(featureVector)

    return featureVectors

# An extra function if we are looking to save our feature vectors for later
def printFeatureVectors(outDir, featureVectors):

    f = open(outDir, 'w')
    for vector in featureVectors:
        for item in vector:
            f.write(str(item) + " ")
        f.write("\n")
    f.close()

# If we want to read in some feature vectors instead of creating them.
def readInFeatureVectorsFromFile(dir):
    list = [line.rstrip('\n') for line in open(dir)]
    list = [i.split() for i in list]
    newList = []
    for row in list:
        newRow = []
        for item in row:
            floatitem = float(item)
            newRow.append(floatitem)
        newList.append(newRow)

    return newList

# Print the intermediate results before clustering occurs
def printFeatureImages(featureImages, naming, printlocation):

    i =0
    for image in featureImages:
        # Normalize to intensity values
        imageToPrint = cv2.normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imwrite(printlocation + "\\" + naming + str(i) + ".png", imageToPrint)
        i+=1

# Print the final result, the user can also choose to make the output grey
def printClassifiedImage(labels, k, img, outdir, greyOutput):

    if(greyOutput):
        labels = labels.reshape(img.shape)
        for row in range(0, len(labels)):
            for col in range(0, len(labels[0])):
                outputIntensity = (255/k)*labels[row][col]
                labels[row][col] = outputIntensity
        cv2.imwrite(outdir, labels.reshape(img.shape))
    else:
        pyplt.imsave(outdir, labels.reshape(img.shape))
    with open(outdir[:-4]+"_segments.txt", "w") as f:
        np.savetxt(f, labels.reshape(img.shape))
    return labels
# Call the k means algorithm for classification
def clusterFeatureVectors(featureVectors, k):

    kmeans = clstr.KMeans(n_clusters=k)
    kmeans.fit(featureVectors)
    labels = kmeans.labels_

    return labels

# To clean up old filter and feature images if the user chose to print them.
def deleteExistingSubResults(outputPath):
    for filename in os.listdir(outputPath):
        if (filename.startswith("filter") or filename.startswith("feature")):
            os.remove(filename)

# Checks user input (i.e. cannot have a negative mask size value)
def check_positive_int(n):
    int_n = int(n)
    if int_n < 0:
         raise argparse.ArgumentTypeError("%s is negative" % n)
    return int_n

# Checks user input (i.e. cannot have a negative weighting value)
def check_positive_float(n):
    float_n = float(n)
    if float_n < 0:
         raise argparse.ArgumentTypeError("%s is negative " % n)
    return float_n

#--------------------------------------------------------------------------
# All of the functions below were left here to demonstrate how I went about
# cropping the input images. I left them here, in the case that Brodatz
# textures were downloaded and cropped as new input images.
#--------------------------------------------------------------------------

def cropTexture(x_offset, Y_offset, width, height, inDir, outDir):

    box = (x_offset, Y_offset, width, height)
    image = Image.open(inDir)
    crop = image.crop(box)
    crop.save(outDir, "PNG")

def deleteCroppedImages():
    for filename in glob.glob(brodatz + "*crop*"):
        os.remove(filename)

def concatentationOfBrodatzTexturesIntoRows(pathsToImages, outdir, axisType):
    images = []
    for thisImage in pathsToImages:
        images.append(cv2.imread(thisImage, cv2.CV_LOAD_IMAGE_GRAYSCALE))
    cv2.imwrite(outdir, np.concatenate(images, axis=axisType))

    outimg = cv2.imread(outdir, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    return outimg

def createGrid(listOfBrodatzInts, outName, howManyPerRow):

    listOfRowOutputs = []
    for i in range(len(listOfBrodatzInts)):
        brodatzCropInput = brodatz + "D" + str(listOfBrodatzInts[i]) + ".png"
        brodatzCropOutput = brodatz + "cropD" + str(listOfBrodatzInts[i]) + ".png"
        # 128x128 crops, in order to generate a 512x512 image
        cropTexture(256, 256, 384, 384, brodatzCropInput, brodatzCropOutput)
        listOfRowOutputs.append(brodatzCropOutput)
    subOuts = [listOfRowOutputs[x:x + howManyPerRow] for x in xrange(0,len(listOfRowOutputs), howManyPerRow)]
    dests = []
    for i in range(len(subOuts)):
        dest = brodatz + "cropRow" + str(i) + ".png"
        dests.append(dest)
        concatentationOfBrodatzTexturesIntoRows(subOuts[i], brodatz + "cropRow" + str(i) + ".png", 1)
    concatentationOfBrodatzTexturesIntoRows(dests, brodatz + outName, 0)

    # Destroy all sub crops (we can make this optional if we want!)
    deleteCroppedImages()

def createGridWithCircle(listOfBrodatzInts, circleInt, outName):

    listOfRowOutputs = []
    for i in range(len(listOfBrodatzInts)):
        brodatzCropInput = brodatz + "D" + str(listOfBrodatzInts[i]) + ".png"
        brodatzCropOutput = brodatz + "cropD" + str(listOfBrodatzInts[i]) + ".png"
        # 128x128 crops, in order to generate a 256x256 image
        cropTexture(256, 256, 384, 384, brodatzCropInput, brodatzCropOutput)
        listOfRowOutputs.append(brodatzCropOutput)
    subOuts = [listOfRowOutputs[x:x + 2] for x in xrange(0, len(listOfRowOutputs), 2)]
    dests = []
    for i in range(len(subOuts)):
        dest = brodatz + "cropRow" + str(i) + ".png"
        dests.append(dest)
        concatentationOfBrodatzTexturesIntoRows(subOuts[i], brodatz + "cropRow" + str(i) + ".png", 1)
    concatentationOfBrodatzTexturesIntoRows(dests, brodatz + "Nat5crop.png", 0)

    size = (128, 128)
    mask = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=0)
    im = Image.open(brodatz + "D" + str(circleInt) + ".png")
    output = ImageOps.fit(im, mask.size, centering=(0.5, 0.5))
    output.paste(0, mask=mask)
    output.save(brodatz + 'circlecrop.png', transparency=0)

    img = Image.open(brodatz + 'circlecrop.png').convert("RGBA")
    img_w, img_h = img.size
    background = Image.open(brodatz + "Nat5crop.png")
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
    background.paste(output, offset, img)
    background.save(brodatz + outName, format="png")
    deleteCroppedImages()

def createTexturePair(pair, outName):
    pathsToTemp = [brodatz + "D" + str(pair[0]) + ".png", brodatz + "D" + str(pair[1]) + ".png"]
    cropTexture(256, 256, 384, 384, pathsToTemp[0], brodatz + "outcrop1.png")
    cropTexture(256, 256, 384, 384, pathsToTemp[1], brodatz + "outcrop2.png")
    cropsToConcat = [brodatz + "outcrop1.png", brodatz + "outcrop2.png"]
    concatentationOfBrodatzTexturesIntoRows(cropsToConcat, outName, 1)
    deleteCroppedImages()


# Our main driver function to return the segmentation of the input image.
def runGabor(args):

    infile = args.infile
    if(not os.path.isfile(infile)):
        print(infile, " is not a file!")
        exit(0)

    outfile = args.outfile
    printlocation = os.path.dirname(os.path.abspath(outfile))
    deleteExistingSubResults(printlocation)

    M_transducerWindowSize = args.M
    if((M_transducerWindowSize % 2) == 0):
        print('Gaussian window size not odd, using next odd number')
        M_transducerWindowSize += 1

    k_clusters = args.k
    k_gaborSize = args.gk

    spatialWeight = args.spw
    gammaSigmaPsi = []
    gammaSigmaPsi.append(args.gamma)
    gammaSigmaPsi.append(args.sigma)
    gammaSigmaPsi.append(args.psi)
    variance_Threshold = args.vt
    howManyFeatureImages = args.fi
    R_threshold = args.R
    sigmaWeight = args.siw
    greyOutput = args.c
    printIntermediateResults = args.i
    if infile[-3:]!='txt':
        if int(cv2.__version__[0]) >= 3:
            img = cv2.imread(infile, 0)
        else:
            img = cv2.imread(infile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        test = np.loadtxt(infile)
        img = (test/test.max()*255).astype('uint8')

    lambdas = getLambdaValues(img)
    filters = build_filters(lambdas, k_gaborSize, gammaSigmaPsi)

    print("Gabor kernels created, getting filtered images")
    filteredImages = getFilterImages(filters, img)
    filteredImages = filterSelection(filteredImages, R_threshold, img, howManyFeatureImages)
    if(printIntermediateResults):
        printFeatureImages(filteredImages, "filter", printlocation)

    print("Applying nonlinear transduction with Gaussian smoothing")
    featureImages = nonLinearTransducer(img, filteredImages, M_transducerWindowSize, sigmaWeight, filters)
    featureImages = removeFeatureImagesWithSmallVariance(featureImages, variance_Threshold)

    if (printIntermediateResults):
        printFeatureImages(featureImages, "feature", printlocation)

    featureVectors = constructFeatureVectors(featureImages, img)
    featureVectors = normalizeData(featureVectors, False, spatialWeight=spatialWeight)

    print("Clustering...")
    labels = clusterFeatureVectors(featureVectors, k_clusters)
    labels = printClassifiedImage(labels, k_clusters, img, outfile, greyOutput)
    plt.subplot(121)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(labels.reshape(img.shape))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
# For running the program on the command line
# def main2(infile, outfile, k, gk, M, spw=1, gamma=1, sigma=1):
def main():

    # initialize
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("-infile", required=True)
    parser.add_argument("-outfile", required=True)

    parser.add_argument('-k', help='Number of clusters', type=check_positive_int, required=True)
    parser.add_argument('-gk', help='Size of the gabor kernel', type=check_positive_int, required=True)
    parser.add_argument('-M', help='Size of the gaussian window', type=check_positive_int, required=True)

    # Optional arguments
    parser.add_argument('-spw', help='Spatial weight of the row and columns for clustering, DEFAULT = 1', nargs='?', const=1,
                        type=check_positive_float, default=1, required=False)
    parser.add_argument('-gamma', help='Spatial aspect ratio, DEFAULT = 1', nargs='?', const=1, default=1,
                        type=check_positive_float, required=False)
    parser.add_argument('-sigma', help='Spread of the filter, DEFAULT = 1', nargs='?', const=1, default=1,
                        type=check_positive_float, required=False)
    parser.add_argument('-psi', help='Offset phase, DEFAULT = 0', nargs='?', const=0, default=0,
                        type=check_positive_float, required=False)
    parser.add_argument('-vt', help='Variance Threshold, DEFAULT = 0.0001', nargs='?', const=0.0001, default=0.0001,
                        type=check_positive_float, required=False)
    parser.add_argument('-fi', help='Maximum number of feature images wanted, DEFAULT = 100', nargs='?', const=100, default=100,
                        type=check_positive_int, required=False)
    parser.add_argument('-R', help='Energy R threshold, DEFAULT = 0.95', nargs='?', const=0.95, default=0.95,
                        type=check_positive_float, required=False)
    parser.add_argument('-siw', help='Sigma weight for gaussian smoothing, DEFAULT = 0.5', nargs='?', const=0.5, default=0.5,
                        type=float, required=False)
    parser.add_argument('-c', help='Output grey? True/False, DEFAULT = False', nargs='?', const=False, default=False,
                        type=bool, required=False)
    parser.add_argument('-i', help='Print intermediate results (filtered/feature images)? True/False, DEFAULT = False', nargs='?', const=False, default=False,
                        type=bool, required=False)

    args = parser.parse_args()
    runGabor(args)
    
if __name__ == "__main__":
    segments = main()

# def mahala(xx, yy):
#     e = xx-yy
#     X = np.vstack([xx,yy])
#     V = np.cov(X.T) 
#     p = np.linalg.inv(V)
#     D = np.sqrt(np.sum(np.dot(e,p) * e, axis = 1))
#     return D

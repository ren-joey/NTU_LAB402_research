import os
import csv
import six
import numpy as np
from pathlib import Path
from PIL import Image
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, shape2D
import SimpleITK as sitk

settings = {
    'binWidth': 25,
    'interpolator': sitk.sitkBSpline,
    'resampledPixelSpacing': None
}

applyLog = False
applyWavelet = False

def some_test():
    # a = np.random.choice([0, 1, 2, 3], (10, 10))
    # b = np.zeros((10, 10, 3), dtype=np.uint8)
    # print(a)
    # for i, c in enumerate([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]):
    #     b[a == i] = c
    # print()
    # print(b)
    # b = Image.fromarray(b)
    # b = b.resize((10, 10), resample=Image.NEAREST)
    # b = np.asarray(b)
    # d = np.ones((10, 10), dtype=np.uint8)
    # print(b.ndim)
    # for i, c in enumerate([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]):
    #     d[(b == c).all(-1)] = i
    # print()
    # print(d)
    raise

def ct_radiomics(in_path, out_path, id_range):
    in_path = [
        Path(in_path[0]),
        Path(in_path[1])
    ]
    csv_path = Path(out_path, 'radiomics.csv')
    if csv_path.is_file():
        # confirm = input('remove current file (Y/N)?')
        os.remove(csv_path)
    f = open(csv_path, 'w', newline='')
    writer = csv.writer(f)
    header_data = ['id']
    row_data = []

    for id in range(1, id_range + 1):
        # https://github.com/AIM-Harvard/pyradiomics/blob/master/examples/helloFeatureClass.py

        img_paths = [
            f'{in_path[0]}/{id}.png',
            f'{in_path[1]}/{id}.png'
        ]
        o_mask = np.array(Image.open(img_paths[1]))
        mask = np.zeros(o_mask.shape[:2], dtype=np.int64)
        for i, c in enumerate([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]):
            mask[(o_mask == c).all(-1)] = i
        image = sitk.ReadImage(img_paths[0])
        mask = sitk.GetImageFromArray(mask)
        each_row_data = [id]

        interpolator = settings.get('interpolator')
        resampledPixelSpacing = settings.get('resampledPixelSpacing')
        if interpolator is not None and resampledPixelSpacing is not None:
            image, mask = imageoperations.resampleImage(image, mask, **settings)

        bb, correctedMask = imageoperations.checkMask(image, mask)
        if correctedMask is not None:
            mask = correctedMask
        image, mask = imageoperations.cropToTumorMask(image, mask, bb)

        #
        # Show the first order feature calculations
        #
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
        # firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableAllFeatures()

        # print('Will calculate the following first order features: ')
        # for f in firstOrderFeatures.enabledFeatures.keys():
        #     print('  ', f)
        #     print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)

        print('Calculating first order features...')
        results = firstOrderFeatures.execute()
        print('done')

        # print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
            if id == 1:
                header_data.append(key)
            each_row_data.append(val)

            # print('  ', key, ':', val)

        #
        # Show Shape features
        #
        shapeFeatures = shape2D.RadiomicsShape2D(image, mask, **settings)
        shapeFeatures.enableAllFeatures()

        # print("Will calculate the following Shape features: ")
        # for f in shapeFeatures.enabledFeatures.keys():
        #     print("  ", f)
        #     print(getattr(shapeFeatures, "get%sFeatureValue" % f).__doc__)

        print("Calculating Shape features...")
        results = shapeFeatures.execute()
        print("done")

        # print("Calculated Shape features: ")
        for key, val in six.iteritems(results):
            if id == 1:
                header_data.append(key)
            each_row_data.append(val)

            # print("  ", key, ":", val)

        #
        # Show GLCM features
        #
        glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
        glcmFeatures.enableAllFeatures()

        # print("Will calculate the following GLCM features: ")
        # for f in glcmFeatures.enabledFeatures.keys():
        #     print("  ", f)
        #     print(getattr(glcmFeatures, "get%sFeatureValue" % f).__doc__)

        print("Calculating GLCM features...")
        results = glcmFeatures.execute()
        print("done")

        # print("Calculated GLCM features: ")
        for key, val in six.iteritems(results):
            if id == 1:
                header_data.append(key)
            each_row_data.append(val)

            # print("  ", key, ":", val)

        #
        # Show GLRLM features
        #
        glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
        glrlmFeatures.enableAllFeatures()

        # print("Will calculate the following GLRLM features: ")
        # for f in glrlmFeatures.enabledFeatures.keys():
        #     print("  ", f)
        #     print(getattr(glrlmFeatures, "get%sFeatureValue" % f).__doc__)

        print("Calculating GLRLM features...")
        results = glrlmFeatures.execute()
        print("done")

        # print("Calculated GLRLM features: ")
        for key, val in six.iteritems(results):
            if id == 1:
                header_data.append(key)
            each_row_data.append(val)

            # print("  ", key, ":", val)

        #
        # Show GLSZM features
        #
        glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
        glszmFeatures.enableAllFeatures()

        # print("Will calculate the following GLSZM features: ")
        # for f in glszmFeatures.enabledFeatures.keys():
        #     print("  ", f)
        #     print(getattr(glszmFeatures, "get%sFeatureValue" % f).__doc__)

        print("Calculating GLSZM features...")
        results = glszmFeatures.execute()
        print("done")

        # print("Calculated GLSZM features: ")
        for key, val in six.iteritems(results):
            if id == 1:
                header_data.append(key)
            each_row_data.append(val)

            # print("  ", key, ":", val)

        #
        # Show FirstOrder features, calculated on a LoG filtered image
        #
        if applyLog:
            sigmaValues = numpy.arange(5.0, 0.0, -0.5)[::1]
            for logImage, imageTypeName, inputKwargs in imageoperations.getLoGImage(
                image, mask, sigma=sigmaValues
            ):
                logFirstorderFeatures = firstorder.RadiomicsFirstOrder(
                    logImage, mask, **inputKwargs
                )
                logFirstorderFeatures.enableAllFeatures()
                results = logFirstorderFeatures.execute()
                for key, val in six.iteritems(results):
                    if id == 1:
                        header_data.append(key)
                    each_row_data.append(val)

                    # laplacianFeatureName = "%s_%s" % (imageTypeName, key)
                    # print("  ", laplacianFeatureName, ":", val)
        #
        # Show FirstOrder features, calculated on a wavelet filtered image
        #
        if applyWavelet:
            for (
                decompositionImage,
                decompositionName,
                inputKwargs,
            ) in imageoperations.getWaveletImage(image, mask):
                waveletFirstOrderFeaturs = firstorder.RadiomicsFirstOrder(
                    decompositionImage, mask, **inputKwargs
                )
                waveletFirstOrderFeaturs.enableAllFeatures()
                results = waveletFirstOrderFeaturs.execute()
                print("Calculated firstorder features with wavelet ", decompositionName)
                for key, val in six.iteritems(results):
                    if id == 1:
                        header_data.append(key)
                    each_row_data.append(val)

                    # waveletFeatureName = "%s_%s" % (str(decompositionName), key)
                    # print("  ", waveletFeatureName, ":", val)

        row_data.append(each_row_data)

    writer.writerow(header_data)
    writer.writerows(row_data)
    f.close()


if __name__ == '__main__':
    in_path = [
        '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/xdata',
        '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/ydata'
        # '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_segment/all/ydata'
    ]
    out_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/RT_spine_NESMS_info'
    id_range = 911
    ct_radiomics(in_path, out_path, id_range)
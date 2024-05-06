import pydicom
import numpy as np
from scipy import ndimage
from skimage import transform
import os
import errno
import warnings

warnings.filterwarnings('ignore')

def readDicomImg(filename):

    ds = pydicom.read_file(filename)
    scale_z = ds.SpacingBetweenSlices
    scale_y,scale_x = ds.PixelSpacing
    
    img = ds.pixel_array
    img = np.array(img).astype('uint8')

    #scale = 0.25 mm/pixel
    # z = int(img.shape[0] * scale_z * 4) + 1
    # y = int(img.shape[1] * scale_y * 4) + 1
    # x = int(img.shape[2] * scale_x * 4) + 1
    # print(x,y,z)

    return img, scale_x, scale_y, scale_z
    
def extractDicomReport(filename):
    info = pydicom.read_file(filename)
    report = info[0x0040,0xa730].value._list
    outputstr = ''

    for t in report:
        meaning = t[0x0040,0xa043].value._list[0][0x0008,0x0104].value
        if([0x0008,0x1199] in t):
            uid = t[0x0008,0x1199].value._list[0][0x0008,0x1155].value

        if meaning.find('Case')!=-1:
            case = meaning
        else:
            case = ''
        if([0x0040,0xa730] in t):
            contents = t[0x0040,0xa730].value._list
            outputstr += '\'' + filename + '\',' + uid + ',' + case
            for content in contents:
                outputstr += ',\'' + content[0x0040,0xa160].value + '\''
            outputstr += '\n'

    return uid, outputstr

def extractVOI(img,bbox,z1,z2):
    
    x1, y1, x2, y2 = bbox  #x1 < x2 , y1 < y2

    img = img[max(z1-3,0):min(z2+3,img.shape[0]), max(y1-10,0):min(y2+10,img.shape[1]), max(x1-10,0):min(x2+10,img.shape[2])]
    
    return img

def createbbox(points, scale_x, scale_y):
    points = points.split('|')
    coordinates = []
    for p in points:
        point = p.split(' ')
        for s in point:
            x, y = s.split(',')
            x = float(x) * scale_y / scale_x #從太豪report還原成正常dicom size
            y = float(y)
            coordinates.append([x, y])
    coordinates = np.array(coordinates)
    coordinates = np.round(coordinates)
    x1, y1 = np.min(coordinates,axis=0)
    x2, y2 = np.max(coordinates,axis=0)
    
    return (int(x1), int(y1), int(x2), int(y2))


def main(root_dirname):
    scale = 1

    with open('./data.txt', 'w+') as to_f:

        for inx, child in enumerate(os.listdir(root_dirname)):
            print('---', inx, ' / ', len(os.listdir(root_dirname)))
            child_dirname = os.path.join(root_dirname, child)
            # chlid = id folder
            if os.path.isdir(child_dirname):
                for loc in os.listdir(child_dirname):
                    dirname = os.path.join(child_dirname, loc)
                    # dir = location folder
                    if os.path.isdir(dirname):
                        for f in os.listdir(dirname):
                            if '.dcm' in f:
                                if f[0] != '1':
                                    report_dir = os.path.join(dirname, f)
                                    _, report = extractDicomReport(report_dir)
                                else:
                                    dicom_dir = os.path.join(dirname, f)
                                    img, scale_x, scale_y, scale_z = readDicomImg(dicom_dir)

                        #np.save(dirname+'/ori_dicom.npy', img)
                        #scale_img = ndimage.interpolation.zoom(img, [scale_z*scale, scale_y*scale, scale_x*scale], mode='nearest')
                        #np.save(dirname+'/resize_dicom.npy', scale_img)

                        report_str = report.splitlines()
                        for i, note in enumerate(report_str):
                            note = note.split("\'")
                            points = note[23]
                            slices = note[-2].split('-')
                            bbox = createbbox(points, scale_x, scale_y)
                            #print(bbox)
                            x1, y1, x2, y2 = bbox
                            z1 = int(slices[0])
                            z2 = int(slices[1])

                            ex_img = extractVOI(img, bbox, z1, z2)
                            z, y, x = ex_img.shape
                            print(child, ex_img.shape)
                            scale_voi = ndimage.interpolation.zoom(ex_img, [scale_z*scale, scale_y*scale, scale_x*scale], mode='nearest')  #可以嘗試更換resize方法
                            # scale_voi = transform.resize(ex_img, (z*scale_z*scale, y*scale_y*scale, x*scale_x*scale))   #同上

                            #np.save(dirname+'/ori_volume'+str(i+1)+'.npy', ex_img)
                            #np.save(dirname+'/resize_volume'+str(i+1)+'.npy', scale_voi)
                            np.save('D:/3D_ABUS/volume/'+str(child)+'_'+str(loc)+'_voi_'+str(i+1)+'.npy', scale_voi)

                            to_f.write(str(child) +' / '+ str(loc) +' / '+ str(z1)+','+str(y1)+','+str(x1) +' / '+ str(z2)+','+str(y2)+','+str(x2) + '  '+ str(scale_voi.shape) + '\n')
    

        print('------', child, '- done ------  ')


if __name__ == '__main__':
    # filename = 'D:/3D_ABUS/VOI/911/RAP/1.3.6.1.4.1.47779.1.002.dcm'

    # a = readDicomImg(filename)[0]
    # print(a.shape)

    #filename = 'D:/3D_ABUS/VOI/829/LAP/SR_Chen^Shaoyi_829_201901281828.dcm'
    #uid, out = extractDicomReport(filename)
    
    dirname = 'D:/3D_ABUS/VOI'
    main(dirname)

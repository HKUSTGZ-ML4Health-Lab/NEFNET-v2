import nibabel as nib
import numpy as np
import glob
import os

def document_write():
    with open('','w') as file:
        for item in list:
            file.write(f'{item}\n')

def standard12():
    file_path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/seg.nii.gz'
    a12_0 = np.where(labels == 0)[0][0]
    a12_1 = np.where(labels == 1)[0][0]
    a12_2 = np.where(labels == 2)[0][0]
    a12_3 = np.where(labels == 3)[0][0]
    a12_4 = np.where(labels == 4)[0][0]
    a12_5 = np.where(labels == 5)[0][0]
    list = [a12_0, a12_1, a12_2, a12_3, a12_4, a12_5]
    sorted_list = sorted(list)
    v1av = np.mean(standard12[sorted_list[0]:sorted_list[1], :], axis=0)
    v2av = np.mean(standard12[sorted_list[1]:sorted_list[2], :], axis=0)
    v3av = np.mean(standard12[sorted_list[2]:sorted_list[3], :], axis=0)
    v4av = np.mean(standard12[sorted_list[3]:sorted_list[4], :], axis=0)
    v5av = np.mean(standard12[sorted_list[4]:sorted_list[5], :], axis=0)
    v6av = np.mean(standard12[sorted_list[5]:, :], axis=0)
    # v1av[1] = 120
    # v2av[1] = 120
    # v3av[1] = 120
    # v4av[1] = 150
    # v5av[1] = 210
    # v6av[1] = 287
    v1_x = angle_with_x_axis(v1av - heart_av)
    v2_x = angle_with_x_axis(v2av - heart_av)
    v3_x = angle_with_x_axis(v3av - heart_av)
    v4_x = angle_with_x_axis(v4av - heart_av)
    v5_x = angle_with_x_axis(v5av - heart_av)
    v6_x = angle_with_x_axis(v6av - heart_av)

def theta_with_z_axis(point):
    y, x, z = point
    y = -y
    vector_magnitude = np.sqrt(x**2 + y**2 + z**2)
    cos_theta = z / vector_magnitude
    theta = np.arccos(cos_theta)
    # 将弧度转换为角度
    angle_degrees = np.degrees(theta)
    return angle_degrees

def alpha_with_x_axis(point):
    y, x, z = point
    y = -y
    vector_magnitude = np.sqrt(x**2 + y**2 + z**2)
    cos_theta = x / vector_magnitude
    theta = np.arccos(cos_theta)
    # 将弧度转换为角度
    angle_degrees = np.degrees(theta)
    return angle_degrees
def axis():
    import scipy.io
    mat = scipy.io.loadmat('/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/ep0.mat')
    ecg = mat['bspm']
    #----------------CT-----------------
    import nibabel as nib
    import numpy as np
    import glob
    file_path = glob.glob('F:\Master_file\Data\Totalsegmentator_dataset\*')
    path = os.path.join(file_path[0], 'segmentation.nii.gz')
    img = nib.load(path)
    # 获取图像数据
    data = img.get_fdata()

    # standard12 = np.argwhere(data == 2)
    bspm = np.argwhere(data == 3)
    ref = np.argwhere(data == 4)
    ref = np.mean(ref,axis=0)
    heart = np.argwhere(data == 1)
    heart_av = np.mean(heart,axis=0)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=54, random_state=42)
    kmeans.fit(bspm)
    labels = kmeans.labels_

    a_bspm = []
    for i in range(54):
        a_bspm.append(np.where(labels == i)[0][0])
    a_bspm = sorted(a_bspm)

    bspm_av = []
    for i in range(53):
        bspm_av.append(np.mean(bspm[a_bspm[i]:a_bspm[i + 1], :], axis=0))
    bspm_av.append(np.mean(bspm[a_bspm[53]:, :], axis=0))

def axis2():
    file_path = glob.glob('F:\Master_file\Data\Totalsegmentator_dataset\*')


    path = os.path.join(file_path[0], 'segmentation.nii.gz')
    img = nib.load(path)
    # 获取图像数据
    data = img.get_fdata()

    heart = np.argwhere(data == 1)
    heart_av = np.mean(heart, axis=0)
    print(path[-26:])

    bspm_av = [[266, 137, 267], [266, 137, 226], [266, 138, 189],
               [235, 197, 394], [235, 207, 355], [235, 206, 309], [235, 208, 267], [235, 211, 227], [235, 216, 188],
               [202, 208, 394], [202, 219, 356], [202, 223, 307], [202, 228, 267], [202, 232, 227], [202, 238, 188],
               [169, 205, 394], [169, 217, 356], [169, 225, 306], [169, 234, 266], [169, 243, 227], [169, 247, 188],
               [136, 206, 394], [136, 217, 356], [136, 225, 306], [136, 234, 266], [136, 243, 227], [136, 247, 188],
               [102, 210, 394], [102, 220, 357], [102, 224, 308], [102, 228, 267], [102, 232, 228], [102, 237, 188],
               [69, 198, 394], [69, 208, 356], [69, 206, 309], [69, 201, 267], [69, 211, 227], [69, 216, 188],
               [45, 137, 267], [48, 137, 226], [42, 137, 189],

               [83, 68, 394], [83, 67, 309], [83, 75, 227],
               [130, 67, 394], [130, 56, 309], [130, 62, 227],
               [177, 65, 394], [177, 55, 309], [177, 61, 227],
               [224, 66, 394], [224, 65, 309], [224, 73, 227]
               ]

    thetas = []
    alphas = []
    for i in bspm_av:
        thetas.append(theta_with_z_axis(i - heart_av))
        alphas.append(alpha_with_x_axis(i - heart_av))

    with open(os.path.join(path[:-20], 'thetas.txt'), 'w') as file:
        # 遍历列表中的每个元素
        for item in thetas:
            # 将每个元素写入文件，每个元素后面跟着一个换行符
            file.write("%s\n" % item)
    with open(os.path.join(path[:-20], 'alphas.txt'), 'w') as file:
        # 遍历列表中的每个元素
        for item in alphas:
            # 将每个元素写入文件，每个元素后面跟着一个换行符
            file.write("%s\n" % item)
    thetas_list.append(thetas)
    alphas_list.append(alphas)

def segmentation():
    import numpy as np
    import nibabel as nib
    img = nib.load('/home/ALL_Lab_Data/zzh/CT/heart_atrium_left.nii.gz')
    data = img.get_fdata()
    data = (data > 0).astype(np.float64)

    roi_files_root = '/home/ALL_Lab_Data/zzh/CT'
    roi_files_labels = ['heart_atrium_right.nii.gz', 'heart_myocardium.nii.gz',
                        'heart_ventricle_left.nii.gz', 'heart_ventricle_right.nii.gz']

    for i in roi_files_labels:
        roi_img = nib.load(os.path.join(roi_files_root, i))
        roi_data = roi_img.get_fdata()
        roi_data = (roi_data > 0).astype(np.float64)
        data = np.logical_or(data, roi_data).astype(np.float64)

    new_img = nib.Nifti1Image(data, img.affine, img.header)
    # 保存合并后的 NIfTI 文件
    nib.save(new_img, os.path.join(roi_files_root,'heart.nii'))

def bspm():
    import scipy.io
    import matplotlib.pyplot as plt
    path = '/home/ALL_Lab_Data/zzh/rawdata/S02/ep0.mat'
    data = scipy.io.loadmat(path)
    ecg = data['bspm']
    RA = ecg[:,3:4]
    LA = ecg[:,33:34]
    LL = ecg[:,38:39]
    mix = (RA + LL + LA)/3

    ecg_or = ecg
    ecg2 = []
    for i in range(ecg.shape[1]):
        ecg2.append(ecg[:,i] - mix[:,0])
    ecg2 = np.asarray(ecg2)

    plt.plot(ecg_or[100:1000, 0])
    plt.show()

if __name__ == '__main__':
    thetas_list, alphas_list = [], []
    print('begin')

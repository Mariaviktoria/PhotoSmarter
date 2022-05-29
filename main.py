from os import listdir
from PIL import Image
import imagehash
import glob
import shutil
import os
import cv2
import pandas as pd

#Variable declarations
global smile_status
hash_list_hex = []
hash_list_int = []
blur_list = []
image_list = []
smile_list =[]
cluster_list =[]
final_list =[]


# get the path/directory
src_dir = os.getcwd() + "/dataset/"
# Check whether the specified path exists or not
isExist = os.path.exists(os.getcwd() + '/output')
if not isExist:
  # Create a new directory because it does not exist
  os.makedirs(os.getcwd() + '/output/')
dst_dir = os.getcwd() + "/output/"

def image_sort():
  images = os.listdir(src_dir)
  for i in range(len(images)):
    start_img = images[i]
    image_list.append(start_img)
    int_val , hex_val = similarity_checker(start_img)
    hash_list_int.append(int_val)
    hash_list_hex.append(hex_val)
    blur_list.append(blur_check(start_img))
    smile_list.append(smile_detect(start_img))
    cluster_list.append('')

  df = pd.DataFrame(list(zip(image_list, hash_list_int, hash_list_hex, blur_list, smile_list, cluster_list)),
                    columns=['Image_Name', 'Hash_Value_Int', 'Hash_Value_Hex', 'Blur_Value', 'Smile_Detect', 'Cluster_Group'])

  df.sort_values(by='Hash_Value_Int', inplace=True, ascending=False)
  df = df.reset_index()
  df = df.drop('index', 1)
  df.to_csv('output.csv')
  return df


#Check for blur in each image
def blur_check(start_img):
  img = cv2.imread(src_dir + start_img)
  laplacian_val = cv2.Laplacian(img, cv2.CV_64F).var()
  #Return the laplacian_val value showing the blur level
  return laplacian_val

#Generate image hash
def similarity_checker(start_img):
  hash_start_hex = imagehash.average_hash(Image.open(src_dir + start_img))
  hash_start_int= int(str(hash_start_hex), 16)
  # cal_cutoff = 11  # maximum bits that could be different between the hashes.
  return hash_start_int, hash_start_hex

#Detect face and smile

def smile_detect(start_img):
  smile_status = False
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
  smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

  # Read the image
  image = cv2.imread(src_dir + start_img)
  # To convert image to monochrome
  try:
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # First detect the face
    faces = face_cascade.detectMultiScale(gray_scale, 1.3, 5)
    for (x, y, w, h) in faces:
      cv2.rectangle(image, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
      roi_gray = gray_scale[y:y + h, x:x + w]
      # roi_color = image[y:y + h, x:x + w]
      # Detect the smile
      smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
      for (sx, sy, sw, sh) in smiles:
        smile_status = True
        # cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
  except:
    pass
  return smile_status

def image_cluster(df_sort):
  cal_cutoff = 11  # maximum bits that could be different between the hashes.
  cluster = 0
  # print(df_sort['Hash_Value_Hex'])
  for i in range(len(df_sort)):
    if df_sort['Cluster_Group'][i] == '':
      for j in range(len(df_sort) - 1):
        if df_sort['Cluster_Group'][j+1] == '':
          # print(df_sort['Hash_Value_Hex'][i])
          # print(df_sort['Hash_Value_Hex'][j + 1])
          if ((df_sort['Hash_Value_Hex'][i]) - ((df_sort['Hash_Value_Hex'][j + 1])) < cal_cutoff):
            # print(True)
            df_sort['Cluster_Group'][i] = cluster
            df_sort['Cluster_Group'][j + 1] = cluster
          else:
            # print(False)
            cluster = cluster + 1
    # print(df_sort['Cluster_Group'])
  # print(df_sort)
  df_sort.to_csv('cluster.csv')
  return df_sort

def final_sort(df_cluster):
  groups = df_cluster.groupby('Cluster_Group')
  for name, group in groups:
    df_temp = group.loc[group['Smile_Detect'] == True]
    if not df_temp.empty:
      df_temp = df_temp.loc[df_temp['Blur_Value'] == (df_temp['Blur_Value'].max())]
    else:
      df_temp = group.loc[group['Blur_Value'] == (group['Blur_Value'].max())]
    final_list.append(df_temp['Image_Name'].values[0])
  print(final_list)
  for value in final_list:
    shutil.move(src_dir + value, dst_dir)

if __name__ == '__main__':
    df_sort = image_sort()
    # df_sort =pd.read_csv("output.csv")
    df_cluster = image_cluster(df_sort)
    df_cluster.to_csv('final_sort.csv')
    # df_cluster = pd.read_csv("cluster.csv")
    final_sort(df_cluster)


import cv2
import csv
from pathlib import Path
import numpy as np
from skimage.feature import graycomatrix, graycoprops

"""
  H → [0, 179]

  Como Hue es un valor circular, lo tratamos como un ángulo
"""
def get_H_mean_std(H):
  H = H * 2 * np.pi / 180.0 # Convertir a ángulo

  # Media de vectores
  mean_sin = np.mean(np.sin(H)) # Media de x
  mean_cos = np.mean(np.cos(H)) # Media de y

  h_mean = np.arctan2(mean_sin, mean_cos) # Obtenemos ángulo

  if(h_mean < 0):
    h_mean += 2 * np.pi # En caso de que esté en otro cuadrante

  R = np.sqrt(mean_sin**2 + mean_cos**2)

  h_std = np.sqrt(-2 * np.log(R + 1e-8))  # epsilon por estabilidad

  # Convertir a grados
  h_mean_deg = h_mean * 180 / (2 * np.pi)
  h_std_deg  = h_std  * 180 / np.pi

  return (h_mean_deg, h_std_deg)

def hsv_features(hsv):
  H,S,V = cv2.split(hsv)
  h_mean, h_std = get_H_mean_std(H)
  
  return (
    h_mean,
    np.mean(S),
    np.mean(V),
    h_std,
    np.std(S),
    np.std(V)
  )

def glcm_features(gray):
  gray_q = (gray / 16).astype(np.uint8)
  
  glcm = graycomatrix(
    gray_q,
    distances=[1],
    angles=[0],
    levels=16,
    symmetric=True,
    normed=True
  )

  return (
    graycoprops(glcm, 'contrast')[0,0],
    graycoprops(glcm, 'dissimilarity')[0,0],
    graycoprops(glcm, 'homogeneity')[0,0],
    graycoprops(glcm, 'ASM')[0,0],
    graycoprops(glcm, 'correlation')[0,0],
  )

def entropy_feature(gray):
  hist = cv2.calcHist([gray], [0], None, [256], [0,256])
  hist = hist / hist.sum()
  entropy = -np.sum(hist * np.log2(hist + 1e-10))

  return entropy

def intensity_features(rgb):
  R,G,B = cv2.split(rgb)

  luminance = 0.299*R + 0.587*G + 0.114*B
  return (
    np.mean(luminance),
    np.std(luminance)
  )

def extract_feature(image_path: Path):
  img = cv2.imread(image_path)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  return (
    *hsv_features(hsv),
    *glcm_features(gray),
    entropy_feature(gray),
    *intensity_features(rgb)
  )

def export_feature(image_path: Path):
  feature = extract_feature(image_path)

  writer.writerow((
    image_path.stem,
    *feature))

with open("dataset.csv", "w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow([
    "id",
    "h_mean",
    "v_mean",
    "s_mean",
    "h_std",
    "v_std",
    "s_std",
    'contrast',
    'dissimilarity',
    'homogeneity',
    'asm',
    'correlation',
    "entropy",
    "intensity_mean",
    "intensity_std"
  ])

  filenames = [p for p in Path("media").iterdir() if p.is_file()]

  for filename in filenames:
    try:
      export_feature(filename)
    except:
      print(f"Error al abrir el archivo {filename}")
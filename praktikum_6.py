import pandas as pd
import numpy as np
import streamlit as st


def baca_file(file):
    file = pd.read_csv(file)
    return file

def buat_array(df):
    x = df[['luas','kamar','usia']]
    x = np.array(x)

    y = df['harga']
    y = np.array(y)
    return x, y

#fungsi eliminasi Gauss
def Eliminasi_Gauss(matrix):
    baris = len(matrix)
    kolom = len(matrix[0])
    pivot = 0

    for col in range(kolom):
        baris_pivot = None

        for row in range(pivot, baris):
            if abs(matrix[row][col]) > 1e-12:
                baris_pivot = row
                break

        if baris_pivot is None:
            continue

        matrix[baris_pivot], matrix[pivot] = matrix[pivot].copy(), matrix[baris_pivot].copy()

        for k in range(pivot + 1, baris):
            faktor = matrix[k][col] / matrix[pivot][col]
            for j in range(col, kolom):
                matrix[k][j] -= faktor * matrix[pivot][j]

        pivot += 1
        if pivot == baris:
            break

    return matrix

#fungsi untuk melakukan subtitusi
def substitusi(aug_matriks):
    n = aug_matriks.shape[0]
    w = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(aug_matriks[i][i]) < 1e-12:
            raise ValueError("Matriks singular, tidak ada solusi unik")
        w[i] = (aug_matriks[i][-1] - np.dot(aug_matriks[i][i + 1:n], w[i + 1:])) / aug_matriks[i][i]
    return w

def hitung_norm_equation(x, y):
    XtX = x.T @ x # perkalian matriks
    Xty = x.T @ y

    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        raise ValueError("woi gaboleh") 

    aug = np.hstack([XtX, Xty.reshape(-1, 1)]).astype(float)
    aug = Eliminasi_Gauss(aug)
    w = substitusi(aug)

    return np.round(w)

def tampilan_hasil(x, y, w):
    XtX = x.T @ x 
    Xty = x.T @ y
    
    y_prediksi = x @ w

    rmse = np.sqrt(np.mean((y_prediksi - y) ** 2))
    r2 = r2 = 1 - (np.sum((y - y_prediksi) ** 2) / np.sum((y - np.mean(y)) ** 2))

    tabel_perbandingan = pd.DataFrame({
        "y_asli":y,
        "y_prediksi":y_prediksi
    })

    return rmse, y_prediksi, r2, XtX, Xty, tabel_perbandingan

def hitung_input(x_input, w):
    y_prediksi = np.dot(x_input,w)
    return y_prediksi

# pembatas bagian UI streamlit
st.title('pertemuan 6')

file = st.file_uploader("silahkah masukan file csv anda")

if file :
    df = baca_file(file)
    x, y = buat_array(df)
    w = hitung_norm_equation(x, y)
    rmse, y_prediksi, r2, XtX, Xty, tabel_perbandingan = tampilan_hasil(x, y, w)

    st.dataframe(df)

    
    col_1, col_2 = st.columns(2)
    with col_1:
        st.write("bobot didapat")
        st.write(w)
        st.dataframe(tabel_perbandingan)
        
    with col_2:
        st.write(f" ini hasil rmse {rmse}")
        st.write(f" ini hasil r2 {r2}")
        st.write(f" ini adalah A {XtX}")
        st.write(f" ini adalah hasil B {Xty}")
        st.write(f"hasil persamaan anda {w[0]}X1 {w[1]}X2 {w[2]}X3")

        kamar = st.number_input("masukan kamar", step=1)
        usia = st.number_input("masukan usia", step=1)
        luas = st.number_input("masukan luas", step=1)

        x_input = np.array([luas, kamar, usia])
        
        if st.button("tekan"):
            y_prediksi = hitung_input(x_input, w)
            st.write(f"hore prediksi model anda adalah {y_prediksi}")